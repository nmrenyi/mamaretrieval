#!/usr/bin/env python3
"""Build the unified audit candidate pool and list new (q, c) pairs to judge.

Reads the 5 per-retriever top-K JSONLs for the audit subset, unions them
per query, marks each candidate with the per-retriever rank it appeared
at (or null), and flags whether the (query_id, chunk_id) pair was already
labeled in data/relevance_labels.jsonl from Phase 2b.

Outputs:
  - data/audit/candidates_audit.jsonl
      One record per audit query, with all unique candidates and their
      per-retriever ranks + a `judged_in_phase2b` flag.
  - data/audit/new_pairs_to_judge.jsonl
      Flat (query_id, chunk_id) records the LLM judge has not yet seen.
      This is the input for the audit's judging step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_INPUTS = {
    "bm25": "data/audit/bm25_top20.jsonl",
    "medcpt": "data/audit/medcpt_top20.jsonl",
    "octen": "data/audit/octen_top20.jsonl",
    "voyage": "data/audit/voyage_top20.jsonl",
    "lateon": "data/audit/lateon_top20.jsonl",
}


def load_retriever(path: Path) -> dict[str, list[tuple[str, int]]]:
    """query_id -> list of (chunk_id, rank), sorted by rank."""
    out: dict[str, list[tuple[str, int]]] = {}
    for line in path.open():
        r = json.loads(line)
        out[r["query_id"]] = [
            (item["chunk_id"], item["rank"]) for item in r["results"]
        ]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument(
        "--relevance-labels", default="data/relevance_labels.jsonl"
    )
    p.add_argument(
        "--pool-output", default="data/audit/candidates_audit.jsonl"
    )
    p.add_argument(
        "--new-pairs-output", default="data/audit/new_pairs_to_judge.jsonl"
    )
    args = p.parse_args()

    audit_ids = [
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    ]
    print(f"Audit queries: {len(audit_ids)}", flush=True)

    per_retriever: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for retriever, path in DEFAULT_INPUTS.items():
        per_retriever[retriever] = load_retriever(Path(path))
        n = sum(len(v) for v in per_retriever[retriever].values())
        print(f"  {retriever:<7} loaded {n:5,} (q,c) pairs from {path}", flush=True)

    print(f"\nLoading Phase 2b labels from {args.relevance_labels}...", flush=True)
    labeled: set[tuple[str, str]] = set()
    for line in Path(args.relevance_labels).open():
        r = json.loads(line)
        labeled.add((r["query_id"], r["chunk_id"]))
    print(f"  {len(labeled):,} labeled pairs", flush=True)

    pool_out = Path(args.pool_output)
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    new_pairs_out = Path(args.new_pairs_output)

    n_pool_total = n_pool_unique = n_new = 0
    new_pairs: list[tuple[str, str]] = []
    with pool_out.open("w") as f_pool:
        for qid in audit_ids:
            ranks: dict[str, dict[str, int | None]] = {}
            for retriever, data in per_retriever.items():
                for chunk_id, rank in data.get(qid, []):
                    if chunk_id not in ranks:
                        ranks[chunk_id] = {r: None for r in per_retriever}
                    ranks[chunk_id][retriever] = rank

            candidates = []
            for chunk_id, retriever_ranks in ranks.items():
                judged = (qid, chunk_id) in labeled
                if not judged:
                    new_pairs.append((qid, chunk_id))
                    n_new += 1
                n_pool_total += 1
                candidates.append(
                    {
                        "chunk_id": chunk_id,
                        "ranks": retriever_ranks,
                        "judged_in_phase2b": judged,
                    }
                )
            n_pool_unique += len(candidates)

            f_pool.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "retrievers": list(per_retriever.keys()),
                        "n_candidates": len(candidates),
                        "candidates": candidates,
                    }
                )
                + "\n"
            )

    with new_pairs_out.open("w") as f_new:
        for qid, cid in new_pairs:
            f_new.write(json.dumps({"query_id": qid, "chunk_id": cid}) + "\n")

    print(f"\nAudit pool written to {pool_out}")
    print(f"  unique (q,c) candidates: {n_pool_unique:,}")
    print(f"  average per query:       {n_pool_unique / len(audit_ids):.1f}")
    print(f"\nNew pairs to judge written to {new_pairs_out}")
    print(f"  pairs not yet labeled by Phase 2b: {n_new:,} "
          f"({n_new / max(n_pool_unique, 1) * 100:.1f}% of audit pool)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
