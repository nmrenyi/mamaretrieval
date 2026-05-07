#!/usr/bin/env python
"""LLM-based chunk filter.

For each sampled chunk, asks a local LLM to understand the chunk, generate a
clinical query it can answer, and judge whether the query is answerable and
clinically useful. Chunks passing both checks are kept.

Usage:
    python scripts/llm_filter_chunks.py                  # full run
    python scripts/llm_filter_chunks.py --limit 20       # test run
    python scripts/llm_filter_chunks.py --resume         # continue after interruption
    python scripts/llm_filter_chunks.py --workers 4      # increase concurrency
    python scripts/llm_filter_chunks.py --no-think       # faster smoke test
    python scripts/llm_filter_chunks.py --backend openai --model Qwen/Qwen3.6-27B-FP8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any


OLLAMA_URL = "http://localhost:11434/api/chat"
OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OPENAI_MODEL = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL
DEFAULT_INPUT = Path("data/sampled_chunks.jsonl")
DEFAULT_OUTPUT = Path("data/llm_filtered_chunks.jsonl")
DEFAULT_RESULTS = Path("data/llm_filter_results.jsonl")

SYSTEM_PROMPT = """You evaluate text chunks from midwifery and obstetrics clinical guidelines.

Your task has three steps:

1. Carefully read the chunk and identify:
   - The clinical topic or patient situation it addresses
   - The specific guidance, recommendations, dosages, or procedures it contains
   - Whether its primary purpose is to guide clinical care or to serve an educational, administrative, or structural function
   - How complete the clinical information is — does it provide enough to guide action, or is it partial or introductory?
   This understanding informs your clinical relevance judgment and answerability evaluation.

2. Judge clinical relevance (clinically_useful).
   A chunk IS clinically relevant if it primarily contains:
   - Diagnosis, assessment, or recognition of a condition
   - Clinical management steps, procedures, or protocols
   - Drug names, dosages, routes, or contraindications
   - Risk factors, complications, prevention, or emergency responses
   - Counseling or patient education that directly supports health decisions
   - Evidence-based clinical recommendations for patient care
   Do not reject a chunk merely because it is written in an explanatory style — if it explains clinically relevant facts that support counseling, assessment, prevention, risk recognition, diagnosis, management, or referral, it is clinically relevant.

   A chunk is NOT clinically relevant if it is primarily:
   - Educator instructions, classroom activities, group discussion prompts, role-play directions, or facilitator notes
   - Student reflection prompts, scenario discussion questions, or assessment / competency questions
   - Module / chapter structure overviews or table-of-contents listings
   - Bibliography, citation, copyright, cataloging, acknowledgements, or contact information
   - Professional conduct or organizational advice that does not guide patient counseling, assessment, prevention, or care
   - Very sparse incomplete fragments
   If the chunk is not clinically relevant, stop here and return query=null with clinically_useful=false and answerable_by_chunk=false.

3. Only if clinically relevant: generate ONE clinical question that a practicing midwife or nurse would type into a clinical reference system — a direct question, not a conversational sentence.
   The question must be ≤20 words and specific enough that only one or two guideline sections would answer it.
   Then evaluate answerable_by_chunk: treat as true if the chunk contains the key fact, step, indication, warning, or recommendation needed to answer the question — even if the answer is incomplete.
   If no specific answerable question can be derived from the chunk, use query=null rather than inventing an unrelated clinical question.

The reason must explain your clinical relevance judgment and, if applicable, your answerability assessment, in ≤30 words.

Return exactly one JSON object — no prose, no markdown fences. The four patterns below are options, not all to be returned. Choose whichever applies and write your own reason in ≤30 words:
{"query": "<question ≤20 words>", "reason": "<≤30 words>", "answerable_by_chunk": true, "clinically_useful": true}
{"query": "<question ≤20 words>", "reason": "<≤30 words>", "answerable_by_chunk": true, "clinically_useful": false}
{"query": "<question ≤20 words>", "reason": "<≤30 words>", "answerable_by_chunk": false, "clinically_useful": true}
{"query": null, "reason": "<≤30 words>", "answerable_by_chunk": false, "clinically_useful": false}"""

RESULT_SCHEMA_VERSION = "answerable-clinically-useful-v1"
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS),
                        help="JSONL file logging every judgment (used for --resume).")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                        help="LLM serving backend. Use openai for vLLM/SGLang.")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults to qwen3.5:9b for Ollama and Qwen/Qwen3.6-27B-FP8 for OpenAI-compatible backends.")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"Ollama chat endpoint. Default {OLLAMA_URL}.")
    parser.add_argument("--base-url", default=OPENAI_BASE_URL,
                        help=f"OpenAI-compatible base URL. Default {OPENAI_BASE_URL}.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="OpenAI-compatible API key. vLLM commonly accepts EMPTY.")
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


def _resolve_model(backend: str, model: str | None) -> str:
    if model:
        return model
    if backend == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_OLLAMA_MODEL


def _openai_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    return exc.read().decode("utf-8", errors="replace").strip()


def _load_done_ids(results_path: Path, backend: str, model: str) -> tuple[set[str], int]:
    if not results_path.exists():
        return set(), 0
    done = set()
    stale = 0
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if _matches_current_result_contract(rec, backend, model):
                    done.add(rec["chunk_id"])
                else:
                    stale += 1
    return done, stale


def _matches_current_result_contract(rec: dict[str, Any], backend: str, model: str) -> bool:
    return (
        rec.get("llm_filter_schema_version") == RESULT_SCHEMA_VERSION
        and rec.get("llm_filter_prompt_hash") == PROMPT_HASH
        and rec.get("llm_backend") == backend
        and rec.get("llm_model") == model
        and "answerable_by_chunk" in rec
        and "clinically_useful" in rec
    )


def _matches_current_output_contract(rec: dict[str, Any], backend: str, model: str) -> bool:
    return (
        rec.get("llm_filter_schema_version") == RESULT_SCHEMA_VERSION
        and rec.get("llm_filter_prompt_hash") == PROMPT_HASH
        and rec.get("llm_backend") == backend
        and rec.get("llm_model") == model
        and "seed_query" in rec
        and "llm_answerable_by_chunk" in rec
        and "llm_clinically_useful" in rec
    )


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
    url: str,
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
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read())

    raw = response["message"]["content"].strip()
    parsed = json.loads(raw)
    answerable_by_chunk = _as_bool(parsed.get("answerable_by_chunk", False))
    clinically_useful = _as_bool(parsed.get("clinically_useful", False))

    return {
        "chunk_id": chunk["chunk_id"],
        "llm_filter_schema_version": RESULT_SCHEMA_VERSION,
        "llm_filter_prompt_hash": PROMPT_HASH,
        "llm_backend": "ollama",
        "llm_model": model,
        "query": parsed.get("query") or None,
        "reason": str(parsed.get("reason", "")),
        "answerable_by_chunk": answerable_by_chunk,
        "clinically_useful": clinically_useful,
    }


def _call_openai(
    chunk: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    think: bool,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat endpoint such as vLLM."""
    text = chunk.get("text", "").strip()
    breadcrumb = chunk.get("breadcrumb", "").strip()

    if breadcrumb:
        user_content = f"Breadcrumb: {breadcrumb}\n\nChunk:\n{text}"
    else:
        user_content = f"Chunk:\n{text}"

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": num_predict,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": think},
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        _openai_chat_url(base_url),
        data=json.dumps(request_payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = _read_http_error_body(exc)
        raise RuntimeError(
            f"OpenAI-compatible request failed with HTTP {exc.code}: {detail}"
        ) from exc

    raw = response["choices"][0]["message"]["content"].strip()
    parsed = json.loads(raw)
    answerable_by_chunk = _as_bool(parsed.get("answerable_by_chunk", False))
    clinically_useful = _as_bool(parsed.get("clinically_useful", False))

    return {
        "chunk_id": chunk["chunk_id"],
        "llm_filter_schema_version": RESULT_SCHEMA_VERSION,
        "llm_filter_prompt_hash": PROMPT_HASH,
        "llm_backend": "openai",
        "llm_model": model,
        "query": parsed.get("query") or None,
        "reason": str(parsed.get("reason", "")),
        "answerable_by_chunk": answerable_by_chunk,
        "clinically_useful": clinically_useful,
    }


def _process_one(
    chunk: dict[str, Any],
    backend: str,
    model: str,
    ollama_url: str,
    base_url: str,
    api_key: str,
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
            if backend == "openai":
                return _call_openai(
                    chunk, model, base_url, api_key, timeout, num_predict, temperature, think
                )
            return _call_ollama(
                chunk, model, ollama_url, timeout, num_predict, temperature, think
            )
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
        "llm_filter_schema_version": RESULT_SCHEMA_VERSION,
        "llm_filter_prompt_hash": PROMPT_HASH,
        "llm_backend": backend,
        "llm_model": model,
        "query": None,
        "reason": f"error: {type(last_err).__name__}",
        "answerable_by_chunk": False,
        "clinically_useful": False,
    }


def main() -> int:
    args = _parse_args()
    model = _resolve_model(args.backend, args.model)
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
        done_ids, stale_count = _load_done_ids(results_path, args.backend, model)
        print(f"Resuming: {len(done_ids)} current-schema chunks already judged, skipping.")
        if stale_count:
            print(
                f"Ignoring {stale_count} stale judgments from a different prompt/schema/backend/model."
            )

    todo = [c for c in chunks if c["chunk_id"] not in done_ids]
    if args.limit > 0:
        todo = todo[: args.limit]

    if not todo:
        print("Nothing to process.")
        return 0

    print(
        f"Processing {len(todo)} chunks with backend={args.backend}, model={model}, "
        f"workers={args.workers}, timeout={args.timeout}s, "
        f"max_tokens={args.num_predict}, think={not args.no_think}"
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
            out_rec["llm_filter_schema_version"] = RESULT_SCHEMA_VERSION
            out_rec["llm_filter_prompt_hash"] = PROMPT_HASH
            out_rec["llm_backend"] = result["llm_backend"]
            out_rec["llm_model"] = result["llm_model"]
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
                    args.backend,
                    model,
                    args.ollama_url,
                    args.base_url,
                    args.api_key,
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
            if not _matches_current_output_contract(rec, args.backend, model):
                continue
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
