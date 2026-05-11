#!/usr/bin/env python3
"""Filter data/queries.jsonl to the 100 audit query IDs.

Reads data/audit/query_ids.txt and writes the matching subset of
data/queries.jsonl to data/queries_audit.jsonl, preserving the original
schema and order from query_ids.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--query-ids", default="data/audit/query_ids.txt")
    ap.add_argument("--output", default="data/queries_audit.jsonl")
    args = ap.parse_args()

    ids = [
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    ]
    by_id = {
        json.loads(line)["query_id"]: line.rstrip("\n")
        for line in Path(args.queries).open()
    }
    missing = [qid for qid in ids if qid not in by_id]
    if missing:
        print(f"ERROR: {len(missing)} ids missing: {missing[:5]}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for qid in ids:
            f.write(by_id[qid] + "\n")
    print(f"Wrote {len(ids)} records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
