#!/usr/bin/env python
"""LLM-based relevance judge for Phase 2b.

For each (query, candidate_chunk) pair in the candidate pool, asks an LLM to
assess relevance on three binary dimensions:

  D1 — Topic:      Same clinical problem as the query
  D2 — Meaningful: Contains clinical information (background counts)
  D3 — Actionable: Contains specific clinical guidance (dosing, protocols)

Score computed in post-processing:
  - D1=False                        → score 0  (off topic, or on-topic but
  - D1=True, D2=False, D3=False     → score 0   meaningless)
  - D1=True, D2=True,  D3=False     → score 1  (on topic, background)
  - D1=True, D2=True,  D3=True      → score 2  (on topic, actionable)

Note: D3=True implies D2=True (enforced in post-processing, not by the model).

Usage:
    python scripts/judge_relevance.py                          # full run
    python scripts/judge_relevance.py --limit 5               # test: 5 queries
    python scripts/judge_relevance.py --resume                # continue after interruption
    python scripts/judge_relevance.py --workers 8             # more concurrency
    python scripts/judge_relevance.py --no-think              # faster smoke test
    python scripts/judge_relevance.py --shard 0 5             # cluster shard 0 of 5
    python scripts/judge_relevance.py \\
        --backend openai --model Qwen/Qwen3-32B               # vLLM backend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OLLAMA_URL        = "http://localhost:11434/api/chat"
OPENAI_BASE_URL   = "http://127.0.0.1:8000/v1"
DEFAULT_OLLAMA_MODEL  = "qwen3:32b"
DEFAULT_OPENAI_MODEL  = "Qwen/Qwen3-32B"

DEFAULT_INPUT  = Path("data/candidates.jsonl")
DEFAULT_OUTPUT = Path("data/relevance_labels.jsonl")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical relevance judge for MAMAI, a RAG system serving midwives \
and doctors in Zanzibar. Assess whether a guideline chunk is relevant to a \
clinical query by answering three questions.

D1 — Topic: Does this chunk address the same clinical problem as the query?

  Answer YES if the chunk directly concerns:
  - the same condition, diagnosis, or syndrome
  - the same clinical event or complication
  - the same drug, intervention, or procedure the query asks about
  - the same physiological process that is the direct subject of the query

  Answer NO if the chunk:
  - covers a different condition or procedure entirely
  - covers a broader category that includes the topic but does not address it
  - shares only a body system or patient population but addresses a different
    clinical problem

D2 — Meaningful: Does this chunk contain clinical information?
  YES: background principles, risk factors, pathophysiology, definitions,
       diagnostic criteria, management principles, or any clinical information
       relevant to the topic.
  NO:  administrative text, assessment rubrics, educator notes,
       table-of-contents listings, or content with no clinical information.

D3 — Actionable: Does this chunk contain specific clinical guidance a midwife
or doctor could act on?
  YES: dosing, protocols, diagnostic thresholds, management steps,
       contraindications, clinical criteria, specific recommendations.
  NO:  general background, broad principles, definitions, or narrative
       descriptions without specific guidance.

Respond with JSON only, reasoning first:
{"reasoning": "...", "d1_topic": true/false, "d2_meaningful": true/false, "d3_actionable": true/false}"""

# Enforced by vLLM guided decoding. Field order in "required" drives
# generation order — reasoning is emitted before the boolean verdicts.
JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning":     {"type": "string"},
        "d1_topic":      {"type": "boolean"},
        "d2_meaningful": {"type": "boolean"},
        "d3_actionable": {"type": "boolean"},
    },
    "required": ["reasoning", "d1_topic", "d2_meaningful", "d3_actionable"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class JudgeResult(TypedDict):
    query_id:                str
    chunk_id:                str
    llm_judge_schema_version: str
    llm_judge_prompt_hash:   str
    llm_backend:             str
    llm_model:               str
    reasoning:               str
    d1_topic:                bool
    d2_meaningful:           bool
    d3_actionable:           bool
    score:                   int   # 0 / 1 / 2, or -1 on error


RESULT_SCHEMA_VERSION = "v-" + hashlib.sha256(
    "\x00".join(sorted(JudgeResult.__annotations__)).encode()
).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_user_content(query_text: str, chunk: dict[str, Any]) -> str:
    source = chunk.get("source", "").strip()
    page   = chunk.get("page")
    text   = chunk.get("text", "").strip()

    meta_parts: list[str] = []
    if source:
        meta_parts.append(f"Source: {source}")
    if page is not None:
        meta_parts.append(f"Page: {page}")
    meta = " | ".join(meta_parts)

    chunk_header = f"{meta}\n---" if meta else "---"
    return f"## Query\n{query_text}\n\n## Guideline chunk\n{chunk_header}\n{text}\n---"


_SENTINEL_CHUNK: dict[str, Any] = {
    "source": "__sentinel__",
    "page": 0,
    "text": "Sentinel chunk text.",
}
_SENTINEL_QUERY = "What is the first-line treatment for postpartum hemorrhage?"

PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + "\x00" + _build_user_content(_SENTINEL_QUERY, _SENTINEL_CHUNK)
     ).encode("utf-8")
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_CORPUS_HEADER_RE = re.compile(
    r"<sep>\[SOURCE:(?P<source>[^|]+)\|PAGE:(?P<page>[^|]+)\|CID:(?P<cid>[^\]]+)\]"
)


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    """Parse corpus file → {chunk_id: {source, page, text}}."""
    corpus: dict[str, dict[str, Any]] = {}
    current_cid: str | None = None
    current_meta: dict[str, Any] = {}
    text_lines: list[str] = []

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = _CORPUS_HEADER_RE.match(line)
            if m:
                if current_cid is not None:
                    corpus[current_cid] = {
                        **current_meta,
                        "text": "\n".join(text_lines).strip(),
                    }
                current_cid = m.group("cid")
                raw_page = m.group("page")
                try:
                    page: Any = int(raw_page)
                except ValueError:
                    page = raw_page
                current_meta = {"source": m.group("source"), "page": page}
                text_lines = []
            else:
                text_lines.append(line)

    if current_cid is not None:
        corpus[current_cid] = {
            **current_meta,
            "text": "\n".join(text_lines).strip(),
        }

    return corpus


def _corpus_path_from_config(config_path: Path) -> Path | None:
    """Read corpus.chunks_path from config.yaml, expanding ~ if present."""
    if not config_path.exists():
        return None
    try:
        import yaml  # type: ignore[import]
        with config_path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        raw = cfg.get("corpus", {}).get("chunks_path")
        if raw:
            return Path(raw).expanduser()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",   default=str(DEFAULT_INPUT),
                   help="candidates.jsonl from Phase 2a. Default: %(default)s")
    p.add_argument("--corpus",  default=None,
                   help="Path to chunks_for_rag.txt. Falls back to corpus.chunks_path "
                        "in --config.")
    p.add_argument("--config",  default="config.yaml",
                   help="config.yaml for corpus path fallback. Default: %(default)s")
    p.add_argument("--output",  default=str(DEFAULT_OUTPUT),
                   help="Output JSONL: one record per (query, chunk) pair. "
                        "Default: %(default)s")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                   help="LLM serving backend. Use 'openai' for vLLM. Default: %(default)s")
    p.add_argument("--model",   default=None,
                   help="Model name. Defaults to %(default)s per backend.")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"Ollama chat endpoint. Default: {OLLAMA_URL}")
    p.add_argument("--base-url",   default=OPENAI_BASE_URL,
                   help=f"OpenAI-compatible base URL. Default: {OPENAI_BASE_URL}")
    p.add_argument("--api-key",    default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                   help="API key for OpenAI-compatible backend.")
    p.add_argument("--workers",    type=int, default=4,
                   help="Concurrent LLM requests. Default: %(default)s")
    p.add_argument("--timeout",    type=int, default=120,
                   help="Per-request timeout in seconds. Default: %(default)s")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max generated tokens per call (covers thinking + output). "
                        "Default: %(default)s")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation temperature. Default: %(default)s (deterministic)")
    p.add_argument("--no-think",        action="store_true",
                   help="Disable model thinking (faster smoke tests).")
    p.add_argument("--thinking-budget", type=int, default=0,
                   help="Cap thinking trace at N tokens (0 = unlimited). Use e.g. "
                        "16384 to prevent the thinking trace from exhausting max_tokens "
                        "and leaving no room for the JSON output.")
    p.add_argument("--limit",      type=int, default=0,
                   help="Process at most N queries (0 = all). Default: %(default)s")
    p.add_argument("--shuffle",    action="store_true",
                   help="Shuffle queries before processing. Use with --limit for a "
                        "diverse random sample across sources.")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Random seed for --shuffle. Default: %(default)s")
    p.add_argument("--resume",     action="store_true",
                   help="Skip pairs already present in --output.")
    p.add_argument("--shard",      type=int, nargs=2, metavar=("INDEX", "COUNT"),
                   help="Process shard INDEX (0-based) of COUNT equal parts, "
                        "splitting by query.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_model(backend: str, model: str | None) -> str:
    if model:
        return model
    return DEFAULT_OPENAI_MODEL if backend == "openai" else DEFAULT_OLLAMA_MODEL


def _openai_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    return exc.read().decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _pair_key(query_id: str, chunk_id: str) -> str:
    return f"{query_id}::{chunk_id}"


def _matches_contract(rec: dict[str, Any], backend: str, model: str) -> bool:
    return (
        rec.get("llm_judge_schema_version") == RESULT_SCHEMA_VERSION
        and rec.get("llm_judge_prompt_hash") == PROMPT_HASH
        and rec.get("llm_backend") == backend
        and rec.get("llm_model") == model
    )


def _load_done_keys(
    output_path: Path, backend: str, model: str
) -> tuple[set[str], int]:
    """Return (done_keys, stale_count). Only successful labels (score >= 0) count."""
    if not output_path.exists():
        return set(), 0
    done: set[str] = set()
    stale = 0
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _matches_contract(rec, backend, model) and rec.get("score", -1) >= 0:
                done.add(_pair_key(rec["query_id"], rec["chunk_id"]))
            else:
                stale += 1
    return done, stale


# ---------------------------------------------------------------------------
# Score computation (post-processing)
# ---------------------------------------------------------------------------

def _compute_score(d1: bool, d2: bool, d3: bool) -> tuple[bool, bool, int]:
    """Apply logical constraints and return (d2_final, d3_final, score).

    Rules:
      - D1=False → D2=False, D3=False, score=0
      - D3=True  → D2=True  (actionable implies meaningful)
      - score    = D1 * (D2 + D3)
    """
    if not d1:
        return False, False, 0
    if d3:
        d2 = True
    return d2, d3, int(d2) + int(d3)


def _validate_result(result: dict[str, Any]) -> JudgeResult:
    expected = set(JudgeResult.__annotations__)
    actual   = set(result.keys())
    missing  = expected - actual
    extra    = actual - expected
    if missing or extra:
        raise ValueError(
            f"JudgeResult schema mismatch — missing: {missing}, extra: {extra}"
        )
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# LLM call functions
# ---------------------------------------------------------------------------

def _call_openai(
    query_id:    str,
    query_text:  str,
    chunk_id:    str,
    chunk:       dict[str, Any],
    model:       str,
    base_url:    str,
    api_key:     str,
    timeout:     int,
    max_tokens:  int,
    temperature: float,
    think:            bool,
    thinking_budget:  int = 0,
) -> JudgeResult:
    user_content = _build_user_content(query_text, chunk)

    chat_template_kwargs: dict[str, Any] = {"enable_thinking": think}
    if think and thinking_budget > 0:
        chat_template_kwargs["thinking_budget"] = thinking_budget

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature":   temperature,
        "max_tokens":    max_tokens,
        # vLLM extension: enforce JSON schema via guided decoding
        "guided_json":   JUDGE_JSON_SCHEMA,
        "chat_template_kwargs": chat_template_kwargs,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        _openai_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = _read_http_error_body(exc)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    msg    = response["choices"][0]["message"]
    raw    = (msg.get("content") or "").strip()

    # The reasoning parser may leak think tokens into content in two ways:
    #   (a) full block: <think>...</think>JSON  — strip the whole block
    #   (b) orphaned closing tag: </think>JSON  — strip just the closing tag
    if raw.startswith("<think>"):
        end = raw.find("</think>")
        raw = raw[end + len("</think>"):].strip() if end != -1 else ""
    elif raw.startswith("</think>"):
        raw = raw[len("</think>"):].strip()

    # Strip markdown code fence (```json\n...\n``` or ```\n...\n```)
    if raw.startswith("```"):
        newline = raw.find("\n")
        if newline != -1:
            raw = raw[newline + 1:].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    if not raw:
        # Thinking trace consumed all tokens — capture it for diagnosis.
        thinking = (msg.get("reasoning_content") or "")
        snippet  = thinking[:3000] if thinking else "(no reasoning_content in response)"
        raise json.JSONDecodeError(
            f"Empty content after thinking. reasoning_content[:3000]:\n{snippet}",
            "", 0,
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as _je:
        raise json.JSONDecodeError(
            f"JSON parse failed. raw_repr={repr(raw[:300])}: {_je.msg}",
            raw, _je.pos,
        ) from _je

    d1 = bool(parsed["d1_topic"])
    d2 = bool(parsed["d2_meaningful"])
    d3 = bool(parsed["d3_actionable"])
    d2_final, d3_final, score = _compute_score(d1, d2, d3)

    return _validate_result({
        "query_id":                query_id,
        "chunk_id":                chunk_id,
        "llm_judge_schema_version": RESULT_SCHEMA_VERSION,
        "llm_judge_prompt_hash":   PROMPT_HASH,
        "llm_backend":             "openai",
        "llm_model":               model,
        "reasoning":               str(parsed.get("reasoning", "")),
        "d1_topic":                d1,
        "d2_meaningful":           d2_final,
        "d3_actionable":           d3_final,
        "score":                   score,
    })


def _call_ollama(
    query_id:    str,
    query_text:  str,
    chunk_id:    str,
    chunk:       dict[str, Any],
    model:       str,
    url:         str,
    timeout:     int,
    max_tokens:  int,
    temperature: float,
    think:       bool,
) -> JudgeResult:
    user_content = _build_user_content(query_text, chunk)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "think": think,
        "format": "json",   # valid JSON only; no schema enforcement
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read())

    raw    = response["message"]["content"].strip()
    parsed = json.loads(raw)

    d1 = bool(parsed.get("d1_topic",     False))
    d2 = bool(parsed.get("d2_meaningful", False))
    d3 = bool(parsed.get("d3_actionable", False))
    d2_final, d3_final, score = _compute_score(d1, d2, d3)

    return _validate_result({
        "query_id":                query_id,
        "chunk_id":                chunk_id,
        "llm_judge_schema_version": RESULT_SCHEMA_VERSION,
        "llm_judge_prompt_hash":   PROMPT_HASH,
        "llm_backend":             "ollama",
        "llm_model":               model,
        "reasoning":               str(parsed.get("reasoning", "")),
        "d1_topic":                d1,
        "d2_meaningful":           d2_final,
        "d3_actionable":           d3_final,
        "score":                   score,
    })


def _process_one(
    query_id:    str,
    query_text:  str,
    chunk_id:    str,
    chunk:       dict[str, Any],
    backend:     str,
    model:       str,
    ollama_url:  str,
    base_url:    str,
    api_key:     str,
    timeout:     int,
    max_tokens:  int,
    temperature: float,
    think:            bool,
    thinking_budget:  int = 0,
    retries:          int = 2,
) -> JudgeResult:
    """Judge one pair, retrying transient errors up to `retries` times."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if backend == "openai":
                return _call_openai(
                    query_id, query_text, chunk_id, chunk,
                    model, base_url, api_key, timeout, max_tokens, temperature, think,
                    thinking_budget,
                )
            return _call_ollama(
                query_id, query_text, chunk_id, chunk,
                model, ollama_url, timeout, max_tokens, temperature, think,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            last_err = exc
            break  # malformed output — no point retrying
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(2 ** attempt)

    return _validate_result({
        "query_id":                query_id,
        "chunk_id":                chunk_id,
        "llm_judge_schema_version": RESULT_SCHEMA_VERSION,
        "llm_judge_prompt_hash":   PROMPT_HASH,
        "llm_backend":             backend,
        "llm_model":               model,
        "reasoning":               f"error: {type(last_err).__name__}: {last_err}",
        "d1_topic":                False,
        "d2_meaningful":           False,
        "d3_actionable":           False,
        "score":                   -1,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    model = _resolve_model(args.backend, args.model)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # Resolve corpus path: CLI arg → config.yaml fallback
    if args.corpus:
        corpus_path = Path(args.corpus).expanduser()
    else:
        corpus_path = _corpus_path_from_config(Path(args.config))
        if corpus_path is None:
            print(
                "ERROR: --corpus not provided and could not read corpus.chunks_path "
                f"from {args.config}.",
                file=sys.stderr,
            )
            return 1

    for path, label in [
        (input_path,  "candidates (--input)"),
        (corpus_path, "corpus (--corpus)"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    # Load corpus
    print(f"Loading corpus from {corpus_path} ...")
    corpus = _load_corpus(corpus_path)
    print(f"  {len(corpus):,} chunks loaded.")

    # Load queries with candidate lists
    queries = _read_jsonl(input_path)
    print(f"Loaded {len(queries):,} queries from {input_path}")

    # Shard by query index (all candidates of a query stay in the same shard)
    if args.shard is not None:
        shard_idx, shard_count = args.shard
        if not (0 <= shard_idx < shard_count):
            print(f"ERROR: invalid shard {shard_idx}/{shard_count}", file=sys.stderr)
            return 1
        shard_size = math.ceil(len(queries) / shard_count)
        start  = shard_idx * shard_size
        queries = queries[start : start + shard_size]
        print(
            f"Shard {shard_idx}/{shard_count}: {len(queries):,} queries "
            f"(indices {start}–{start + len(queries) - 1})"
        )

    if args.shuffle:
        import random
        random.Random(args.shuffle_seed).shuffle(queries)
        print(f"Shuffled with seed={args.shuffle_seed}")

    if args.limit > 0:
        queries = queries[: args.limit]
        print(f"Limited to {args.limit} queries.")

    # Load already-labeled pairs for --resume
    done_keys: set[str] = set()
    if args.resume:
        done_keys, stale = _load_done_keys(output_path, args.backend, model)
        print(f"Resuming: {len(done_keys):,} pairs already labeled.")
        if stale:
            print(f"  {stale:,} stale / errored records will be re-processed.")

    # Flatten queries → individual work items, skipping done and missing chunks
    todo: list[tuple[str, str, str]] = []  # (query_id, query_text, chunk_id)
    missing_chunks = 0
    for q in queries:
        qid   = q["query_id"]
        qtext = q["query_text"]
        for cand in q["candidates"]:
            cid = cand["chunk_id"]
            if _pair_key(qid, cid) in done_keys:
                continue
            if cid not in corpus:
                missing_chunks += 1
                continue
            todo.append((qid, qtext, cid))

    if missing_chunks:
        print(
            f"Warning: {missing_chunks:,} candidate chunk IDs not found in corpus "
            "— skipped.",
            file=sys.stderr,
        )
    if not todo:
        print("Nothing to process.")
        return 0

    print(
        f"Processing {len(todo):,} pairs | "
        f"backend={args.backend} model={model} workers={args.workers} "
        f"timeout={args.timeout}s max_tokens={args.max_tokens} "
        f"temperature={args.temperature} think={not args.no_think}"
    )
    print(f"Output → {output_path}")

    # --- Run ---
    out_lock     = Lock()
    score_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    errors       = 0
    start_time   = time.monotonic()

    out_mode = "a" if args.resume else "w"
    out_fh   = output_path.open(out_mode, encoding="utf-8")

    def _on_done(result: JudgeResult) -> None:
        nonlocal errors

        with out_lock:
            out_fh.write(json.dumps(result) + "\n")
            out_fh.flush()

        if result["score"] < 0:
            errors += 1
        else:
            score_counts[result["score"]] += 1

        total   = sum(score_counts.values()) + errors
        elapsed = time.monotonic() - start_time
        rate    = total / elapsed if elapsed > 0 else 0
        eta     = (len(todo) - total) / rate if rate > 0 else float("inf")
        eta_str = f"{eta / 60:.0f}m" if eta < 7200 else f"{eta / 3600:.1f}h"

        status  = "ERR" if result["score"] < 0 else f"s={result['score']}"
        dist    = " ".join(f"{k}:{v}" for k, v in sorted(score_counts.items()))
        print(
            f"[{total:>6}/{len(todo)}] {status} | "
            f"{rate:.1f}/s ETA {eta_str} | "
            f"scores({dist}) err={errors} | "
            f"{result['query_id'][:8]}..{result['chunk_id'][:8]}"
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    qid, qtext, cid, corpus[cid],
                    args.backend, model,
                    args.ollama_url, args.base_url, args.api_key,
                    args.timeout, args.max_tokens, args.temperature,
                    not args.no_think, args.thinking_budget,
                ): (qid, cid)
                for qid, qtext, cid in todo
            }
            for future in as_completed(futures):
                _on_done(future.result())
    finally:
        out_fh.close()

    elapsed = time.monotonic() - start_time
    total   = sum(score_counts.values()) + errors
    print()
    print(f"Done in {elapsed:.0f}s — {total:,} pairs judged, {errors} errors")
    print("Score distribution: " +
          "  ".join(f"score={k}: {v}" for k, v in sorted(score_counts.items())))

    # On resume: deduplicate output, keeping only current-schema successful labels
    if args.resume:
        seen:          set[str]          = set()
        deduped:       list[dict[str, Any]] = []
        error_records: list[dict[str, Any]] = []
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not _matches_contract(rec, args.backend, model):
                    continue
                if rec.get("score", -1) < 0:
                    error_records.append(rec)
                    continue
                key = _pair_key(rec["query_id"], rec["chunk_id"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(rec)
        with output_path.open("w", encoding="utf-8") as fh:
            for rec in deduped:
                fh.write(json.dumps(rec) + "\n")
        if error_records:
            error_path = output_path.parent / (output_path.stem + "_errors.jsonl")
            with error_path.open("w", encoding="utf-8") as fh:
                for rec in error_records:
                    fh.write(json.dumps(rec) + "\n")
            print(f"Error records saved:  {len(error_records):,} → {error_path}")
        print(f"Deduplicated output: {len(deduped):,} labeled pairs → {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
