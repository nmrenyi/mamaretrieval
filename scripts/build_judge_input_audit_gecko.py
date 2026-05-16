#!/usr/bin/env python3
"""Build the judge_relevance.py input file for the Gecko pool expansion.

Reads:
  - data/audit/gecko_top20.jsonl    (top-K Gecko picks per audit query)
  - data/relevance_labels.jsonl     (Phase 2b labels)
  - data/audit/relevance_labels_audit.jsonl  (Phase 3 audit labels)
  - data/queries_audit.jsonl        (query text for the 100 audit queries)

Emits (per-query record, omitting queries with no unlabeled candidates):
  {"query_id": ..., "query_text": ..., "candidates": [{"chunk_id": ...}, ...]}

Default depth: top-3 (matches the Step 4 "Option A" plan — judge the 79
unlabeled (q,c) pairs from Gecko's top-3). Use --top-k to widen.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_labeled_keys(paths: list[Path], audit_qids: set[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for p in paths:
        with p.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec["query_id"] in audit_qids:
                    keys.add((rec["query_id"], rec["chunk_id"]))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gecko-top20", default="data/audit/gecko_top20.jsonl")
    ap.add_argument("--phase2b", default="data/relevance_labels.jsonl")
    ap.add_argument("--audit-labels", default="data/audit/relevance_labels_audit.jsonl")
    ap.add_argument("--queries", default="data/queries_audit.jsonl")
    ap.add_argument("--query-ids", default="data/audit/query_ids.txt")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Depth in Gecko results to consider (default 3)")
    ap.add_argument("--output", default="data/audit/candidates_audit_gecko.jsonl")
    args = ap.parse_args()

    audit_qids = {
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    }
    if len(audit_qids) != 100:
        print(f"WARN: expected 100 audit qids, got {len(audit_qids)}", file=sys.stderr)

    queries_by_id = {
        json.loads(line)["query_id"]: json.loads(line)
        for line in Path(args.queries).open()
    }

    labeled = load_labeled_keys(
        [Path(args.phase2b), Path(args.audit_labels)], audit_qids
    )
    print(f"existing labeled (q,c) pairs (across audit qids): {len(labeled)}")

    # Group unlabeled Gecko top-K chunks per query
    by_qid: dict[str, list[str]] = defaultdict(list)
    total_seen = 0
    total_unlabeled = 0
    for line in Path(args.gecko_top20).open():
        rec = json.loads(line)
        qid = rec["query_id"]
        results = sorted(rec["results"], key=lambda r: r["rank"])
        for r in results[: args.top_k]:
            cid = r["chunk_id"]
            total_seen += 1
            if (qid, cid) not in labeled:
                by_qid[qid].append(cid)
                total_unlabeled += 1

    print(
        f"Gecko top-{args.top_k}: {total_seen} pairs total, "
        f"{total_unlabeled} unlabeled "
        f"({100*total_unlabeled/max(total_seen,1):.1f}%)"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_queries = 0
    with out.open("w") as f:
        for qid in sorted(by_qid):
            if qid not in queries_by_id:
                print(f"ERROR: query_id {qid} not in {args.queries}", file=sys.stderr)
                return 1
            payload = {
                "query_id": qid,
                "query_text": queries_by_id[qid]["query_text"],
                "candidates": [{"chunk_id": c} for c in by_qid[qid]],
            }
            f.write(json.dumps(payload) + "\n")
            n_queries += 1

    print(f"Wrote {n_queries} queries / {total_unlabeled} (q,c) pairs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
