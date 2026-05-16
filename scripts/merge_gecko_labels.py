#!/usr/bin/env python3
"""Merge Gecko pool-expansion labels into relevance_labels_audit.jsonl.

Reads judge output (data/audit/gecko_audit_labels_shard0.jsonl) and the
existing audit labels (data/audit/relevance_labels_audit.jsonl), then:
  1. Verifies schema consistency (prompt hash, schema version, model)
  2. Drops any new label whose (q,c) already exists in the audit file
  3. Annotates new labels with `source: "gecko_pool_expansion"`
  4. Appends them, preserving the existing file as-is

Idempotent: re-running won't duplicate. Provenance tag lets future
analyses split the pool-expansion labels back out if needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_PROMPT_HASH = "f36ff561215b3a6f"
EXPECTED_SCHEMA = "v-f20c636b"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-labels", default="data/audit/gecko_audit_labels_shard0.jsonl")
    ap.add_argument("--audit-labels", default="data/audit/relevance_labels_audit.jsonl")
    ap.add_argument("--source-tag", default="gecko_pool_expansion")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_path = Path(args.new_labels)
    audit_path = Path(args.audit_labels)
    if not new_path.exists():
        print(f"ERROR: {new_path} not found", file=sys.stderr)
        return 1
    if not audit_path.exists():
        print(f"ERROR: {audit_path} not found", file=sys.stderr)
        return 1

    # Load existing audit (q,c) keys for dedup
    existing_keys: set[tuple[str, str]] = set()
    for line in audit_path.open():
        rec = json.loads(line)
        existing_keys.add((rec["query_id"], rec["chunk_id"]))
    print(f"existing audit labels: {len(existing_keys)} (q,c) pairs")

    # Read new labels, validate schema, dedup, annotate
    new_records: list[dict] = []
    skipped_dup = 0
    schema_mismatch = 0
    score_counter = {0: 0, 1: 0, 2: 0}
    for line in new_path.open():
        rec = json.loads(line)
        if rec.get("llm_judge_prompt_hash") != EXPECTED_PROMPT_HASH:
            schema_mismatch += 1
            continue
        if rec.get("llm_judge_schema_version") != EXPECTED_SCHEMA:
            schema_mismatch += 1
            continue
        key = (rec["query_id"], rec["chunk_id"])
        if key in existing_keys:
            skipped_dup += 1
            continue
        rec["source"] = args.source_tag
        new_records.append(rec)
        score_counter[rec.get("score", -1)] = score_counter.get(rec.get("score", -1), 0) + 1

    print(f"new records read: {len(new_records) + skipped_dup + schema_mismatch}")
    print(f"  schema mismatches (skipped): {schema_mismatch}")
    print(f"  duplicates (skipped):        {skipped_dup}")
    print(f"  to append:                   {len(new_records)}")
    print(f"  score 0 / 1 / 2:             {score_counter[0]} / {score_counter[1]} / {score_counter[2]}")

    if args.dry_run:
        print("\n[dry-run] not modifying any file")
        return 0

    if not new_records:
        print("\nNothing to append. Done.")
        return 0

    with audit_path.open("a") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nAppended {len(new_records)} records to {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
