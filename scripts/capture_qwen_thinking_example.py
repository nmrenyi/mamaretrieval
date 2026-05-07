#!/usr/bin/env python
"""Capture one full Ollama chat response, including model thinking.

This is a diagnostic helper for inspecting whether a thinking-enabled Qwen run
actually reaches a final JSON answer for a sampled chunk.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from llm_filter_chunks import DEFAULT_MODEL, SYSTEM_PROMPT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sampled_chunks.jsonl")
    parser.add_argument("--chunk-line", type=int, default=124,
                        help="1-based line number in the JSONL input.")
    parser.add_argument("--chunk-id", default="",
                        help="Optional chunk_id override. If set, --chunk-line is ignored.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--output-prefix", default="data/qwen35_9b_thinking_example")
    parser.add_argument("--num-predict", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking; thinking is enabled by default.")
    parser.add_argument("--no-format-json", action="store_true",
                        help="Do not request Ollama JSON-mode formatting.")
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


def _build_user_content(chunk: dict[str, Any]) -> str:
    text = chunk.get("text", "").strip()
    breadcrumb = chunk.get("breadcrumb", "").strip()
    if breadcrumb:
        return f"Breadcrumb: {breadcrumb}\n\nChunk:\n{text}"
    return f"Chunk:\n{text}"


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    chunk = _read_chunk(input_path, args.chunk_line, args.chunk_id)
    text = chunk.get("text", "").strip()

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_content(chunk)},
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

    raw_path = Path(f"{prefix}_raw_events.jsonl")
    full_path = Path(f"{prefix}_full.json")
    pretty_path = Path(f"{prefix}_pretty.txt")

    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    thinking_parts: list[str] = []
    content_parts: list[str] = []
    event_count = 0
    last_event: dict[str, Any] | None = None
    started = time.time()

    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        with raw_path.open("w", encoding="utf-8") as raw_fh:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

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

    elapsed = time.time() - started
    thinking_text = "".join(thinking_parts)
    content_text = "".join(content_parts)

    full = {
        "model": args.model,
        "input": str(input_path),
        "chunk_line": args.chunk_line,
        "chunk_id": chunk.get("chunk_id"),
        "breadcrumb": chunk.get("breadcrumb"),
        "chunk_text": text,
        "payload": payload,
        "elapsed_seconds": elapsed,
        "num_stream_events": event_count,
        "thinking": thinking_text,
        "content": content_text,
        "done_event": last_event,
    }
    full_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    pretty_path.write_text(
        "MODEL: {model}\n"
        "CHUNK LINE: {chunk_line}\n"
        "CHUNK ID: {chunk_id}\n"
        "BREADCRUMB: {breadcrumb}\n"
        "ELAPSED SECONDS: {elapsed:.2f}\n"
        "STREAM EVENTS: {events}\n"
        "THINKING CHARS: {thinking_chars}\n"
        "CONTENT CHARS: {content_chars}\n\n"
        "===== CHUNK TEXT =====\n"
        "{chunk_text}\n\n"
        "===== THINKING =====\n"
        "{thinking}\n\n"
        "===== FINAL CONTENT =====\n"
        "{content}\n".format(
            model=args.model,
            chunk_line=args.chunk_line,
            chunk_id=chunk.get("chunk_id"),
            breadcrumb=chunk.get("breadcrumb"),
            elapsed=elapsed,
            events=event_count,
            thinking_chars=len(thinking_text),
            content_chars=len(content_text),
            chunk_text=text,
            thinking=thinking_text,
            content=content_text,
        ),
        encoding="utf-8",
    )

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
