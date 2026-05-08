#!/usr/bin/env python3
"""Phase 1c — Assemble final benchmark queries.

Reads data/llm_filtered_chunks.jsonl and produces data/queries.jsonl with
three query types:
  per_chunk   — seed_query from the LLM filter, one per kept chunk
  synthesis   — cross-chunk question for clinical topic groups with 3+ chunks
  adversarial — robustness reformulation of 15% of per_chunk queries

Usage:
  python scripts/generate_queries.py [options]

Options:
  --input PATH           default: data/llm_filtered_chunks.jsonl
  --output PATH          default: data/queries.jsonl
  --config PATH          default: config.yaml
  --skip-synthesis       skip synthesis query generation
  --skip-adversarial     skip adversarial query generation
  --backend {openai,ollama}
  --base-url URL         for openai backend (default: https://api.openai.com/v1)
  --api-key KEY          overrides OPENAI_API_KEY env var
  --model MODEL          overrides config.yaml models.query_generation
  --adversarial-seed N   random seed for adversarial sampling (default: 42)
  --resume               skip queries already written to output
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


# ── Query schema ──────────────────────────────────────────────────────────────

QUERY_SCHEMA_VERSION = "v1"


def _make_query(
    *,
    idx: int,
    query_text: str,
    seed_chunk_id: str,
    source: str,
    tier: str,
    query_type: str,
    adversarial_type: str | None = None,
    seed_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "query_id": f"q_{idx:05d}",
        "query_text": query_text,
        "seed_chunk_id": seed_chunk_id,
        "source": source,
        "tier": tier,
        "query_type": query_type,
        "adversarial_type": adversarial_type,
    }
    if seed_chunk_ids is not None:
        record["seed_chunk_ids"] = seed_chunk_ids
    return record


# ── LLM client ────────────────────────────────────────────────────────────────

def _chat(
    messages: list[dict],
    *,
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    elif backend == "ollama":
        import urllib.request
        payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
        req = urllib.request.Request(f"{base_url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"]
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


# ── Synthesis ─────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = (
    "You are a clinical benchmark designer creating retrieval evaluation questions "
    "for a midwifery/nursing assistant deployed in Zanzibar, Tanzania."
)

_SYNTHESIS_USER_TMPL = """\
Given these passages on the clinical topic "{topic}", generate exactly 1 question \
that requires information from MULTIPLE passages to answer fully.

Requirements:
- The question must NOT be fully answerable by any single passage.
- Write as a nurse or midwife in Zanzibar would ask it.
- The answer should require combining information across passages.
- Maximum 25 words.

Passages:
{passages}

Return ONLY a JSON object with no extra text:
{{"query_text": "...", "seed_chunk_ids": ["{id_hint}", ...]}}
seed_chunk_ids should list the chunk_ids whose combination is needed (2-4 IDs)."""


def _generate_synthesis(
    topic: str,
    chunks: list[dict[str, Any]],
    *,
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any] | None:
    passages = "\n\n".join(
        f'[chunk_id: {c["chunk_id"]}]\n{c["text"][:600]}' for c in chunks[:6]
    )
    id_hint = chunks[0]["chunk_id"]
    user_msg = _SYNTHESIS_USER_TMPL.format(
        topic=topic, passages=passages, id_hint=id_hint
    )
    raw = _chat(
        [{"role": "system", "content": _SYNTHESIS_SYSTEM},
         {"role": "user", "content": user_msg}],
        backend=backend, base_url=base_url, api_key=api_key, model=model,
        temperature=0.7, max_tokens=200,
    )
    # extract JSON
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    qt = obj.get("query_text", "").strip()
    ids = obj.get("seed_chunk_ids", [])
    if not qt or not ids:
        return None
    return {"query_text": qt, "seed_chunk_ids": ids}


# ── Adversarial ───────────────────────────────────────────────────────────────

_ADVERSARIAL_SYSTEM = (
    "You are a clinical NLP researcher generating retrieval stress-test variants "
    "of medical queries for a midwifery benchmark in Zanzibar, Tanzania."
)

_ADVERSARIAL_PROMPTS: dict[str, str] = {
    "abbreviation": (
        "Rephrase the following clinical question to use common medical abbreviations "
        "and shorthand as a nurse would (e.g. PPH for postpartum haemorrhage, MgSO4 "
        "for magnesium sulphate, IM/IV, BP, ANC, CS, FHR, LMP, PMTCT). Only use "
        "abbreviations that are natural here — don't force them.\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
    "typo": (
        "Introduce 1-2 realistic spelling or keyboard typos into the following clinical "
        "question while preserving the clinical intent. The typos should be plausible "
        "human mistakes, not random noise.\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
    "lay_synonym": (
        "Rephrase the following clinical question using colloquial, patient-facing "
        "language instead of professional medical terminology (e.g. 'bleeding too much "
        "after birth' for postpartum haemorrhage, 'water breaking' for rupture of "
        "membranes).\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
    "redundant_context": (
        "Add a brief bedside narrative sentence before the following clinical question "
        "that provides extra context but does not change the core information need.\n\n"
        "Original: {question}\n\nReturn ONLY the expanded question/query string."
    ),
    "ambiguous": (
        "Rephrase the following clinical question to be slightly underspecified or "
        "ambiguous while still pointing to the same likely clinical topic.\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
    "multi_condition": (
        "Rephrase the following clinical question to include an additional constraint "
        "(e.g. a comorbidity, risk factor, contraindication, or patient state) that "
        "makes it a multi-condition query.\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
    "negation": (
        "Rephrase the following clinical question to use negation, avoidance, or "
        "contraindication framing (e.g. 'what should I avoid', 'what is contraindicated', "
        "'when should I not').\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
    "rare_exact": (
        "Rephrase the following clinical question to emphasise or introduce a specific "
        "drug name, dose, measurement, procedure, or rare salient term where exact "
        "matching would matter for retrieval.\n\n"
        "Original: {question}\n\nReturn ONLY the rephrased question string."
    ),
}


def _generate_adversarial(
    question: str,
    adv_type: str,
    *,
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
) -> str | None:
    prompt = _ADVERSARIAL_PROMPTS[adv_type].format(question=question)
    raw = _chat(
        [{"role": "system", "content": _ADVERSARIAL_SYSTEM},
         {"role": "user", "content": prompt}],
        backend=backend, base_url=base_url, api_key=api_key, model=model,
        temperature=0.8, max_tokens=100,
    ).strip().strip('"').strip()
    if not raw or raw.lower() == question.lower():
        return None
    return raw


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/llm_filtered_chunks.jsonl")
    p.add_argument("--output", default="data/queries.jsonl")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-synthesis", action="store_true")
    p.add_argument("--skip-adversarial", action="store_true")
    p.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--adversarial-seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = args.model or cfg.get("models", {}).get("query_generation", "gpt-4o-mini")
    base_url = args.base_url or (
        "http://localhost:11434" if args.backend == "ollama"
        else "https://api.openai.com/v1"
    )
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    adversarial_fraction = cfg.get("queries", {}).get("adversarial_fraction", 0.15)

    # ── Load filtered chunks ─────────────────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    with open(args.input) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} filtered chunks from {args.input}")

    # ── Resume: load already-written query IDs ───────────────────────────────
    done_ids: set[str] = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                done_ids.add(json.loads(line)["query_id"])
        print(f"Resuming: {len(done_ids)} queries already written")

    out = open(args.output, "a" if args.resume else "w")
    idx = len(done_ids)

    def _write(record: dict[str, Any]) -> None:
        nonlocal idx
        if record["query_id"] in done_ids:
            return
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()

    # ── Phase 1: per-chunk queries ───────────────────────────────────────────
    print("\n── Per-chunk queries ──")
    per_chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        idx += 1
        r = _make_query(
            idx=idx,
            query_text=chunk["seed_query"],
            seed_chunk_id=chunk["chunk_id"],
            source=chunk.get("source", ""),
            tier=chunk.get("tier", ""),
            query_type="per_chunk",
        )
        per_chunk_records.append(r)
        _write(r)
    print(f"  Written {len(per_chunk_records)} per_chunk queries")

    # ── Phase 2: synthesis queries ───────────────────────────────────────────
    synthesis_count = 0
    if not args.skip_synthesis:
        print("\n── Synthesis queries ──")
        # group by top-level breadcrumb (first segment before ' > ')
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            bc = chunk.get("breadcrumb", "") or ""
            topic = bc.split(" > ")[0].strip() or chunk.get("section", "__root__")
            by_topic[topic].append(chunk)

        eligible = {t: cs for t, cs in by_topic.items() if len(cs) >= 3}
        print(f"  {len(eligible)} topic groups with 3+ chunks")

        for topic, topic_chunks in sorted(eligible.items()):
            idx += 1
            qid = f"q_{idx:05d}"
            if qid in done_ids:
                synthesis_count += 1
                continue
            result = _generate_synthesis(
                topic, topic_chunks,
                backend=args.backend, base_url=base_url,
                api_key=api_key, model=model,
            )
            if result is None:
                idx -= 1
                print(f"  SKIP (parse error): {topic[:60]}")
                continue
            seed_ids = result["seed_chunk_ids"]
            primary_id = seed_ids[0] if seed_ids else topic_chunks[0]["chunk_id"]
            primary_chunk = next(
                (c for c in topic_chunks if c["chunk_id"] == primary_id),
                topic_chunks[0],
            )
            r = _make_query(
                idx=idx,
                query_text=result["query_text"],
                seed_chunk_id=primary_id,
                source=primary_chunk.get("source", ""),
                tier=primary_chunk.get("tier", ""),
                query_type="synthesis",
                seed_chunk_ids=seed_ids,
            )
            _write(r)
            synthesis_count += 1
            print(f"  [{synthesis_count}] {result['query_text'][:80]}")
        print(f"  Written {synthesis_count} synthesis queries")

    # ── Phase 3: adversarial queries ─────────────────────────────────────────
    adversarial_count = 0
    if not args.skip_adversarial:
        print("\n── Adversarial queries ──")
        n_adversarial = math.ceil(len(per_chunk_records) * adversarial_fraction)
        rng = random.Random(args.adversarial_seed)
        sample = rng.sample(per_chunk_records, min(n_adversarial, len(per_chunk_records)))
        adv_types = list(_ADVERSARIAL_PROMPTS.keys())

        for base_record in sample:
            adv_type = rng.choice(adv_types)
            idx += 1
            qid = f"q_{idx:05d}"
            if qid in done_ids:
                adversarial_count += 1
                continue
            rephrased = _generate_adversarial(
                base_record["query_text"], adv_type,
                backend=args.backend, base_url=base_url,
                api_key=api_key, model=model,
            )
            if rephrased is None:
                idx -= 1
                continue
            r = _make_query(
                idx=idx,
                query_text=rephrased,
                seed_chunk_id=base_record["seed_chunk_id"],
                source=base_record["source"],
                tier=base_record["tier"],
                query_type="adversarial",
                adversarial_type=adv_type,
            )
            _write(r)
            adversarial_count += 1
            if adversarial_count % 50 == 0:
                print(f"  {adversarial_count}/{len(sample)} adversarial queries written")
        print(f"  Written {adversarial_count} adversarial queries")

    out.close()

    total = len(per_chunk_records) + synthesis_count + adversarial_count
    print(f"\nDone. {total} queries total → {args.output}")
    print(f"  per_chunk:   {len(per_chunk_records)}")
    print(f"  synthesis:   {synthesis_count}")
    print(f"  adversarial: {adversarial_count}")


if __name__ == "__main__":
    main()
