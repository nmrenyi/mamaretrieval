#!/usr/bin/env python
"""Capture one full LLM chat response, including model thinking.

This is a diagnostic helper for inspecting whether a thinking-enabled Qwen run
actually reaches a final JSON answer for a sampled chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from llm_filter_chunks import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    OLLAMA_URL,
    OPENAI_BASE_URL,
    SYSTEM_PROMPT,
    _build_user_content,
    _openai_chat_url,
    _resolve_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sampled_chunks.jsonl")
    parser.add_argument("--chunk-line", type=int, default=124,
                        help="1-based line number in the JSONL input.")
    parser.add_argument("--chunk-id", default="",
                        help="Optional chunk_id override. If set, --chunk-line is ignored.")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                        help="LLM serving backend. Use openai for vLLM/SGLang.")
    parser.add_argument("--model", default=None,
                        help=f"Model name. Defaults to {DEFAULT_OLLAMA_MODEL} for Ollama and {DEFAULT_OPENAI_MODEL} for OpenAI-compatible backends.")
    parser.add_argument("--url", default=OLLAMA_URL,
                        help=f"Ollama chat endpoint. Default {OLLAMA_URL}.")
    parser.add_argument("--base-url", default=OPENAI_BASE_URL,
                        help=f"OpenAI-compatible base URL. Default {OPENAI_BASE_URL}.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="OpenAI-compatible API key. vLLM commonly accepts EMPTY.")
    parser.add_argument("--output-prefix", default="data/qwen35_9b_thinking_example")
    parser.add_argument("--num-predict", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking; thinking is enabled by default.")
    parser.add_argument("--no-format-json", action="store_true",
                        help="Do not request backend JSON-mode formatting.")
    parser.add_argument("--pretty-only", action="store_true",
                        help="Write only the pretty text output, not raw events or full JSON.")
    return parser.parse_args()


def _read_chunk(path: Path, chunk_line: int, chunk_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk_id and chunk.get("chunk_id") == chunk_id:
                return chunk
            if not chunk_id and idx == chunk_line:
                return chunk
    if chunk_id:
        raise ValueError(f"Could not find chunk_id={chunk_id!r} in {path}")
    raise ValueError(f"Could not find line {chunk_line} in {path}")


def _build_ollama_payload(args: argparse.Namespace, model: str, user_content: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
        "think": not args.no_think,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
        },
    }
    if not args.no_format_json:
        payload["format"] = "json"
    return payload


def _build_openai_payload(args: argparse.Namespace, model: str, user_content: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.num_predict,
        "chat_template_kwargs": {"enable_thinking": not args.no_think},
    }
    if not args.no_format_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _capture_ollama(
    args: argparse.Namespace,
    payload: dict[str, Any],
    raw_path: Path,
) -> tuple[str, str, int, dict[str, Any] | None]:
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    thinking_parts: list[str] = []
    content_parts: list[str] = []
    event_count = 0
    last_event: dict[str, Any] | None = None

    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        raw_fh = None
        try:
            if not args.pretty_only:
                raw_fh = raw_path.open("w", encoding="utf-8")
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                if raw_fh is not None:
                    raw_fh.write(line + "\n")
                    raw_fh.flush()

                event_count += 1
                event = json.loads(line)
                last_event = event
                message = event.get("message") or {}

                thinking = message.get("thinking")
                if thinking:
                    thinking_parts.append(thinking)

                content = message.get("content")
                if content:
                    content_parts.append(content)

                if event.get("done"):
                    break
        finally:
            if raw_fh is not None:
                raw_fh.close()

    return "".join(thinking_parts), "".join(content_parts), event_count, last_event


def _capture_openai(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> tuple[str, str, int, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    req = urllib.request.Request(
        _openai_chat_url(args.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            response = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible request failed with HTTP {exc.code}: {detail}"
        ) from exc

    message = response["choices"][0]["message"]
    thinking_text = message.get("reasoning") or message.get("reasoning_content") or ""
    content_text = message.get("content") or ""
    return thinking_text, content_text, 1, response


def _format_params_section(payload: dict[str, Any]) -> str:
    """Format request parameters (everything except messages) as readable key: value lines."""
    params = {k: v for k, v in payload.items() if k != "messages"}
    return json.dumps(params, ensure_ascii=False, indent=2)


def _write_pretty(
    path: Path,
    *,
    backend: str,
    model: str,
    chunk_line: int,
    chunk: dict[str, Any],
    payload: dict[str, Any],
    user_content: str,
    elapsed: float | None = None,
    event_count: int | None = None,
    thinking_text: str = "",
    content_text: str = "",
    error: Exception | None = None,
) -> None:
    header_lines = [
        f"BACKEND: {backend}",
        f"MODEL: {model}",
        f"SOURCE: {chunk.get('source', '')}",
        f"CHUNK LINE: {chunk_line}",
        f"CHUNK ID: {chunk.get('chunk_id')}",
        f"BREADCRUMB: {chunk.get('breadcrumb', '')}",
    ]
    if elapsed is not None:
        header_lines.append(f"ELAPSED SECONDS: {elapsed:.2f}")
    if event_count is not None:
        header_lines.append(f"STREAM EVENTS: {event_count}")
    header_lines += [
        f"THINKING CHARS: {len(thinking_text)}",
        f"CONTENT CHARS: {len(content_text)}",
    ]
    if error is not None:
        header_lines.append(f"ERROR: {type(error).__name__}: {error}")

    sections = [
        "\n".join(header_lines),
        "===== REQUEST PARAMETERS =====\n" + _format_params_section(payload),
        "===== SYSTEM PROMPT =====\n" + SYSTEM_PROMPT,
        "===== USER MESSAGE =====\n" + user_content,
        "===== THINKING =====\n" + thinking_text,
        "===== FINAL CONTENT =====\n" + content_text,
    ]
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def _write_error_pretty(
    args: argparse.Namespace,
    model: str,
    input_path: Path,
    prefix: Path,
    chunk: dict[str, Any],
    payload: dict[str, Any],
    user_content: str,
    exc: Exception,
) -> None:
    pretty_path = Path(f"{prefix}_pretty.txt")
    _write_pretty(
        pretty_path,
        backend=args.backend,
        model=model,
        chunk_line=args.chunk_line,
        chunk=chunk,
        payload=payload,
        user_content=user_content,
        error=exc,
    )
    Path(f"{prefix}_error.txt").write_text(
        f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
    )
    print(f"Saved error pretty text: {pretty_path}", file=sys.stderr)


def main() -> int:
    args = _parse_args()
    model = _resolve_model(args.backend, args.model)
    input_path = Path(args.input)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    chunk = _read_chunk(input_path, args.chunk_line, args.chunk_id)
    text = chunk.get("text", "").strip()
    user_content = _build_user_content(chunk)

    if args.backend == "openai":
        payload = _build_openai_payload(args, model, user_content)
    else:
        payload = _build_ollama_payload(args, model, user_content)

    raw_path = Path(f"{prefix}_raw_events.jsonl")
    full_path = Path(f"{prefix}_full.json")
    pretty_path = Path(f"{prefix}_pretty.txt")
    started = time.time()

    try:
        if args.backend == "openai":
            thinking_text, content_text, event_count, last_event = _capture_openai(
                args, payload
            )
        else:
            thinking_text, content_text, event_count, last_event = _capture_ollama(
                args, payload, raw_path
            )
    except Exception as exc:
        _write_error_pretty(args, model, input_path, prefix, chunk, payload, user_content, exc)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - started

    full = {
        "backend": args.backend,
        "model": model,
        "input": str(input_path),
        "chunk_line": args.chunk_line,
        "chunk_id": chunk.get("chunk_id"),
        "source": chunk.get("source"),
        "breadcrumb": chunk.get("breadcrumb"),
        "chunk_text": text,
        "system_prompt": SYSTEM_PROMPT,
        "user_content": user_content,
        "payload": payload,
        "elapsed_seconds": elapsed,
        "num_stream_events": event_count,
        "thinking": thinking_text,
        "content": content_text,
        "done_event": last_event,
    }
    if not args.pretty_only:
        full_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_pretty(
        pretty_path,
        backend=args.backend,
        model=model,
        chunk_line=args.chunk_line,
        chunk=chunk,
        payload=payload,
        user_content=user_content,
        elapsed=elapsed,
        event_count=event_count,
        thinking_text=thinking_text,
        content_text=content_text,
    )

    if not args.pretty_only:
        print(f"Saved raw events: {raw_path}")
        print(f"Saved full JSON: {full_path}")
    print(f"Saved pretty text: {pretty_path}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Events: {event_count}")
    print(f"Thinking chars: {len(thinking_text)}")
    print(f"Content chars: {len(content_text)}")
    print("Final content:")
    print(content_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
