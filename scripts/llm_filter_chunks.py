#!/usr/bin/env python
"""LLM-based chunk filter.

For each sampled chunk, asks a local Ollama model to understand the chunk,
generate a clinical query it can answer, and judge whether the query is
answerable and clinically useful. Chunks passing both checks are kept.

Usage:
    python scripts/llm_filter_chunks.py                  # full run
    python scripts/llm_filter_chunks.py --limit 20       # test run
    python scripts/llm_filter_chunks.py --resume         # continue after interruption
    python scripts/llm_filter_chunks.py --workers 4      # increase concurrency
    python scripts/llm_filter_chunks.py --no-think       # faster smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_INPUT = Path("data/sampled_chunks.jsonl")
DEFAULT_OUTPUT = Path("data/llm_filtered_chunks.jsonl")
DEFAULT_RESULTS = Path("data/llm_filter_results.jsonl")

SYSTEM_PROMPT = """You evaluate text chunks from midwifery and obstetrics clinical guidelines.

Your task has two steps:
1. Generate ONE clinical question that a practicing midwife or nurse would search for when encountering this clinical topic.
   The question must be ≤20 words and specific enough that only one or two guideline sections would answer it — not so broad that any clinical text would be relevant.
2. Given the chunk text and your generated question, evaluate both:
   - answerable_by_chunk: whether the chunk contains enough information to directly answer the question.
   - clinically_useful: whether the question is useful for direct clinical care.

The reason must explain both why the generated question is or is not answerable from the chunk and why it is or is not clinically useful.
Evaluate answerable_by_chunk and clinically_useful independently:
- answerable_by_chunk can be true while clinically_useful is false if the chunk answers a question, but that question is educational, administrative, bibliographic, or otherwise not useful for direct patient care.
- answerable_by_chunk can be false while clinically_useful is true if the generated question is clinically useful, but the chunk lacks enough information to answer it.
Valid decision patterns include:
- {"query": "<question>", "reason": "Chunk answers it and it guides care.", "answerable_by_chunk": true, "clinically_useful": true}
- {"query": "<question>", "reason": "Chunk answers it, but it is not direct care.", "answerable_by_chunk": true, "clinically_useful": false}
- {"query": "<question>", "reason": "Clinically useful, but chunk lacks the answer.", "answerable_by_chunk": false, "clinically_useful": true}
- {"query": null, "reason": "No meaningful clinical or answerable question.", "answerable_by_chunk": false, "clinically_useful": false}

A question is clinically useful if it asks about:
- Diagnosis, assessment, or recognition of a condition
- Clinical management steps, procedures, or protocols
- Drug names, dosages, routes, or contraindications
- Risk factors, complications, or emergency responses
- Evidence-based clinical recommendations for patient care

A question is not clinically useful if the chunk is primarily:
- Educator instructions (ask students, group activities, classroom exercises)
- Student reflection prompts or scenario discussion questions
- Module / chapter structure overviews or table-of-contents listings
- Bibliography or reference lists
- Very sparse incomplete fragments

Respond with JSON only — no prose, no markdown fences:
{"query": "<clinical question>", "reason": "<explanation>", "answerable_by_chunk": true, "clinically_useful": true}

If no meaningful clinical query is possible, respond:
{"query": null, "reason": "<explanation>", "answerable_by_chunk": false, "clinically_useful": false}"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS),
                        help="JSONL file logging every judgment (used for --resume).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N chunks (0 = all).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunk_ids already present in --results.")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent Ollama requests. Default 2.")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-request timeout in seconds. Default 180.")
    parser.add_argument("--num-predict", type=int, default=8192,
                        help="Maximum generated tokens per request. Default 8192.")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Generation temperature. Defaults to 0.6.")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable model thinking for faster smoke tests.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_done_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done.add(rec["chunk_id"])
    return done


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _should_keep(result: dict[str, Any]) -> bool:
    return bool(result["answerable_by_chunk"] and result["clinically_useful"])


def _call_ollama(
    chunk: dict[str, Any],
    model: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    think: bool,
) -> dict[str, Any]:
    """Call Ollama and return the query plus answerability/usefulness judgments."""
    text = chunk.get("text", "").strip()
    breadcrumb = chunk.get("breadcrumb", "").strip()

    if breadcrumb:
        user_content = f"Breadcrumb: {breadcrumb}\n\nChunk:\n{text}"
    else:
        user_content = f"Chunk:\n{text}"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": think,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": num_predict,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read())

    raw = response["message"]["content"].strip()
    parsed = json.loads(raw)
    answerable_by_chunk = _as_bool(parsed.get("answerable_by_chunk", False))
    clinically_useful = _as_bool(parsed.get("clinically_useful", False))

    return {
        "chunk_id": chunk["chunk_id"],
        "query": parsed.get("query") or None,
        "reason": str(parsed.get("reason", "")),
        "answerable_by_chunk": answerable_by_chunk,
        "clinically_useful": clinically_useful,
    }


def _process_one(
    chunk: dict[str, Any],
    model: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    think: bool,
    retries: int = 2,
) -> dict[str, Any]:
    """Attempt to judge one chunk, retrying on transient errors."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _call_ollama(chunk, model, timeout, num_predict, temperature, think)
        except (json.JSONDecodeError, KeyError) as exc:
            last_err = exc
            # Model returned malformed JSON — skip without retry
            break
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(2 ** attempt)

    return {
        "chunk_id": chunk["chunk_id"],
        "query": None,
        "reason": f"error: {type(last_err).__name__}",
        "answerable_by_chunk": False,
        "clinically_useful": False,
    }


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    results_path = Path(args.results)

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    chunks = _read_jsonl(input_path)
    print(f"Loaded {len(chunks)} sampled chunks from {input_path}")

    done_ids: set[str] = set()
    if args.resume:
        done_ids = _load_done_ids(results_path)
        print(f"Resuming: {len(done_ids)} chunks already judged, skipping.")

    todo = [c for c in chunks if c["chunk_id"] not in done_ids]
    if args.limit > 0:
        todo = todo[: args.limit]

    if not todo:
        print("Nothing to process.")
        return 0

    print(
        f"Processing {len(todo)} chunks with model={args.model}, "
        f"workers={args.workers}, timeout={args.timeout}s, "
        f"num_predict={args.num_predict}, think={not args.no_think}"
    )
    print("Output chunks →", output_path)
    print("All judgments →", results_path)
    print()

    results_lock = Lock()
    output_lock = Lock()
    kept = 0
    discarded = 0
    errors = 0
    start_time = time.monotonic()

    output_mode = "a" if args.resume else "w"
    results_fh = results_path.open(output_mode, encoding="utf-8")
    output_fh = output_path.open(output_mode, encoding="utf-8")

    # Build a lookup from chunk_id to full chunk record
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    def _done_callback(result: dict[str, Any]) -> None:
        nonlocal kept, discarded, errors

        chunk_rec = chunk_by_id[result["chunk_id"]]

        with results_lock:
            results_fh.write(json.dumps(result) + "\n")
            results_fh.flush()

        reason = result["reason"]
        if reason.startswith("error:"):
            errors += 1
        elif _should_keep(result):
            kept += 1
            out_rec = dict(chunk_rec)
            out_rec["seed_query"] = result["query"]
            out_rec["llm_answerable_by_chunk"] = result["answerable_by_chunk"]
            out_rec["llm_clinically_useful"] = result["clinically_useful"]
            out_rec["llm_filter_reason"] = result["reason"]
            with output_lock:
                output_fh.write(json.dumps(out_rec) + "\n")
                output_fh.flush()
        else:
            discarded += 1

        total_done = kept + discarded + errors
        elapsed = time.monotonic() - start_time
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = len(todo) - total_done
        eta = remaining / rate if rate > 0 else float("inf")
        eta_str = f"{eta / 60:.0f}m" if eta < 7200 else "—"

        if reason.startswith("error:"):
            status = "ERR"
        elif _should_keep(result):
            status = "KEEP"
        else:
            status = "SKIP"
        query_preview = (result["query"] or "")[:60]
        print(
            f"[{total_done:>4}/{len(todo)}] {status} | "
            f"{rate:.1f}/s ETA {eta_str} | "
            f"{result['chunk_id'][:8]} | {query_preview}"
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    chunk,
                    args.model,
                    args.timeout,
                    args.num_predict,
                    args.temperature,
                    not args.no_think,
                ): chunk
                for chunk in todo
            }
            for future in as_completed(futures):
                result = future.result()
                _done_callback(result)
    finally:
        results_fh.close()
        output_fh.close()

    elapsed = time.monotonic() - start_time
    total = kept + discarded + errors
    print()
    print(f"Done in {elapsed:.0f}s  ({total} judged, {kept} kept, {discarded} discarded, {errors} errors)")
    print(f"Kept rate: {100 * kept / total:.1f}%" if total else "")

    # Write a single consolidated output (re-read and deduplicate in case of resume)
    if not args.resume:
        return 0

    # On resume the output file may have duplicates from earlier runs — deduplicate
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["chunk_id"] not in seen:
                seen.add(rec["chunk_id"])
                deduped.append(rec)

    with output_path.open("w", encoding="utf-8") as fh:
        for rec in deduped:
            fh.write(json.dumps(rec) + "\n")

    print(f"Deduplicated output: {len(deduped)} chunks in {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
