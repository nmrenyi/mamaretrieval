#!/usr/bin/env python3
"""Build a human-reviewable Markdown of the audit-discovered relevant chunks.

For each audit query, lists every (query, chunk) pair that:
  - The audit judged at score = 2 (strict) or score >= 1 (lenient), AND
  - Was NOT in Phase 2a's candidate pool (i.e. would have been silently
    missed by the released benchmark).

Each entry shows: query text, chunk source/page/text, which retrievers
surfaced it, and the judge's reasoning. Goes into `data/audit/` (gitignored
under the *.md rule); the writeup is reproducible from the JSONLs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.config import corpus_chunks_path, load_config
from mamaretrieval.corpus import iter_chunks

PER_RETRIEVER_INPUTS = {
    "bm25":   "data/audit/bm25_top20.jsonl",
    "medcpt": "data/audit/medcpt_top20.jsonl",
    "octen":  "data/audit/octen_top20.jsonl",
    "voyage": "data/audit/voyage_top20.jsonl",
    "lateon": "data/audit/lateon_top20.jsonl",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", default="data/queries_audit.jsonl")
    p.add_argument("--phase2b-labels", default="data/relevance_labels.jsonl")
    p.add_argument("--audit-labels", default="data/audit/relevance_labels_audit.jsonl")
    p.add_argument(
        "--threshold",
        choices=["strict", "lenient"],
        default="strict",
        help="strict = score=2 only; lenient = score>=1",
    )
    p.add_argument("--max-chunk-chars", type=int, default=1200,
                   help="Truncate long chunks to keep the doc scannable.")
    p.add_argument("--output", default=None,
                   help="Default: data/audit/review_missed_{threshold}.md")
    args = p.parse_args()

    out_path = Path(args.output) if args.output else Path(
        f"data/audit/review_missed_{args.threshold}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load audit queries
    audit_queries = {}
    for line in Path(args.queries).open():
        r = json.loads(line)
        audit_queries[r["query_id"]] = r
    audit_qids = set(audit_queries)
    print(f"Audit queries: {len(audit_qids)}", flush=True)

    # Phase 2a pool: any chunk_id that Phase 2b labelled, regardless of score
    pool: dict[str, set[str]] = defaultdict(set)
    for line in Path(args.phase2b_labels).open():
        r = json.loads(line)
        if r["query_id"] in audit_qids:
            pool[r["query_id"]].add(r["chunk_id"])

    # Audit labels with reasoning
    audit_records: dict[tuple[str, str], dict] = {}
    for line in Path(args.audit_labels).open():
        r = json.loads(line)
        audit_records[(r["query_id"], r["chunk_id"])] = r

    # Per-retriever sources
    surfaced_by: dict[tuple[str, str], list[str]] = defaultdict(list)
    for retriever, path in PER_RETRIEVER_INPUTS.items():
        for rec in (json.loads(line) for line in Path(path).open()):
            qid = rec["query_id"]
            for hit in rec["results"]:
                surfaced_by[(qid, hit["chunk_id"])].append(retriever)

    # Load corpus chunks once
    config = load_config("config.yaml")
    corpus_path = corpus_chunks_path(config)
    print(f"Loading corpus from {corpus_path}...", flush=True)
    chunk_text: dict[str, dict] = {}
    for c in iter_chunks(corpus_path):
        chunk_text[c.chunk_id] = {
            "source": c.source, "page": c.page,
            "breadcrumb": c.breadcrumb, "text": c.text,
        }
    print(f"  {len(chunk_text):,} chunks indexed", flush=True)

    # Filter: missed-by-pool and at/above threshold
    threshold_fn = (
        (lambda s: s == 2) if args.threshold == "strict" else (lambda s: s >= 1)
    )
    by_query: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for (qid, cid), rec in audit_records.items():
        if qid not in audit_qids:
            continue
        if not threshold_fn(rec["score"]):
            continue
        if cid in pool.get(qid, set()):
            continue  # already in Phase 2a pool — not "missed"
        by_query[qid].append(rec)
        total += 1

    print(
        f"Missed at threshold={args.threshold}: {total} (q,c) pairs across "
        f"{len(by_query)} queries (avg {total / max(len(by_query), 1):.1f}/query)",
        flush=True,
    )

    # Write Markdown
    md = [
        f"# Audit-discovered relevant chunks Phase 2a missed ({args.threshold})",
        "",
        f"Each entry = a (query, chunk) pair that the audit's LLM judge labelled "
        f"as {'actionable (score = 2)' if args.threshold == 'strict' else 'on-topic (score ≥ 1)'} "
        "but that Phase 2a's candidate pool did not include — so the released "
        "benchmark currently treats it as not-relevant by silence.",
        "",
        f"Total: **{total}** missed pairs across **{len(by_query)}** queries "
        f"(avg {total / max(len(by_query), 1):.1f} per query).",
        "",
        "---",
        "",
    ]

    for qid in sorted(by_query):
        q = audit_queries[qid]
        md.append(f"## {qid} — {q['source']} ({q['tier']})")
        md.append("")
        md.append(f"> {q['query_text']}")
        md.append("")
        md.append(f"{len(by_query[qid])} missed chunk{'s' if len(by_query[qid]) != 1 else ''}:")
        md.append("")
        for rec in sorted(by_query[qid], key=lambda r: r["chunk_id"]):
            cid = rec["chunk_id"]
            chunk = chunk_text.get(cid)
            if chunk is None:
                md.append(f"### chunk `{cid}` — *(NOT IN CORPUS)*")
                md.append("")
                continue
            retrievers = ", ".join(sorted(set(surfaced_by.get((qid, cid), []))))
            text = chunk["text"]
            if len(text) > args.max_chunk_chars:
                text = text[: args.max_chunk_chars].rstrip() + "  …[truncated]"
            md.append(
                f"### chunk `{cid}` — {chunk['source']}, page {chunk['page']}"
                + (f" — {chunk['breadcrumb']}" if chunk["breadcrumb"] else "")
            )
            md.append("")
            md.append(f"**Surfaced by:** {retrievers}")
            md.append(
                f"**Judge:** D1={rec['d1_topic']} D2={rec['d2_meaningful']} "
                f"D3={rec['d3_actionable']} → score={rec['score']}"
            )
            md.append("")
            md.append(f"**Reasoning:** {rec['reasoning']}")
            md.append("")
            md.append("**Chunk text:**")
            md.append("")
            md.append("```")
            md.append(text)
            md.append("```")
            md.append("")
        md.append("---")
        md.append("")

    out_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
