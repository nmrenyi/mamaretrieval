#!/usr/bin/env python3
"""Coverage check for Gecko top-3 against the existing audit label set.

Computes three coverage decompositions per the runbook:
  1. coverage_phase2b     — Gecko top-3 ∩ Phase 2b labels (data/relevance_labels.jsonl)
  2. coverage_audit_pool  — Gecko top-3 ∩ Phase 3 audit labels (data/audit/relevance_labels_audit.jsonl)
  3. coverage_union       — Gecko top-3 ∩ (Phase 2b ∪ Phase 3 audit)  ← gate

Each is computed as the mean over 100 queries of |gecko_top3 ∩ labels| / 3.

Plus, for every Gecko top-3 chunk that has any label, report the
distribution of relevance scores (0/1/2). This is the H1-vs-H2 signal:
labeled hits that score 0 mean Gecko is finding junk (H2);
labeled hits that score 1 or 2 mean Gecko is finding real stuff (H1).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_labels(path: Path, audit_qids: set[str]) -> dict[tuple[str, str], int]:
    """Return {(query_id, chunk_id): score} restricted to audit_qids."""
    labels: dict[tuple[str, str], int] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["query_id"]
            if qid not in audit_qids:
                continue
            key = (qid, rec["chunk_id"])
            labels[key] = rec["score"]
    return labels


def per_query_coverage(
    gecko_top3: dict[str, list[str]],
    label_set: set[tuple[str, str]],
) -> dict[str, float]:
    """Return {query_id: coverage_fraction} for top-3."""
    out: dict[str, float] = {}
    for qid, top3 in gecko_top3.items():
        hits = sum(1 for cid in top3 if (qid, cid) in label_set)
        out[qid] = hits / 3.0
    return out


def coverage_histogram(per_query: dict[str, float]) -> Counter:
    """Bucket each query by integer hits in {0, 1, 2, 3}."""
    return Counter(round(v * 3) for v in per_query.values())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gecko-top20", default="data/audit/gecko_top20.jsonl")
    p.add_argument("--phase2b", default="data/relevance_labels.jsonl")
    p.add_argument("--audit-labels", default="data/audit/relevance_labels_audit.jsonl")
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--show-zero-coverage-queries", action="store_true",
                   help="Print query_ids with zero union coverage")
    args = p.parse_args()

    audit_qids = {
        line.strip() for line in Path(args.query_ids).read_text().splitlines() if line.strip()
    }
    print(f"audit queries: {len(audit_qids)}")

    print(f"\nLoading labels...")
    phase2b = load_labels(Path(args.phase2b), audit_qids)
    audit_labels = load_labels(Path(args.audit_labels), audit_qids)
    print(f"  phase 2b (restricted to audit qids): {len(phase2b)} (q,c) labels")
    print(f"  phase 3 audit:                       {len(audit_labels)} (q,c) labels")
    print(f"  overlap (same key, both sources):    {len(set(phase2b) & set(audit_labels))}")

    phase2b_keys = set(phase2b.keys())
    audit_keys = set(audit_labels.keys())
    union_keys = phase2b_keys | audit_keys
    print(f"  union (Phase 2b ∪ audit):            {len(union_keys)} (q,c) labels")

    # Load Gecko top-K
    gecko_topk: dict[str, list[str]] = {}
    for line in Path(args.gecko_top20).open():
        rec = json.loads(line)
        results = sorted(rec["results"], key=lambda r: r["rank"])
        gecko_topk[rec["query_id"]] = [r["chunk_id"] for r in results[: args.top_k]]
    print(f"\nGecko top-{args.top_k} loaded for {len(gecko_topk)} queries")

    # Per-query coverage against each source
    cov_phase2b = per_query_coverage(gecko_topk, phase2b_keys)
    cov_audit = per_query_coverage(gecko_topk, audit_keys)
    cov_union = per_query_coverage(gecko_topk, union_keys)

    def agg(d: dict[str, float]) -> float:
        return sum(d.values()) / len(d)

    print(f"\n=== Coverage @ top-{args.top_k} (mean over {len(audit_qids)} queries) ===")
    print(f"  coverage_phase2b:    {agg(cov_phase2b):.3f}")
    print(f"  coverage_audit_pool: {agg(cov_audit):.3f}")
    print(f"  coverage_union:      {agg(cov_union):.3f}   <-- gate")

    print(f"\n=== Per-query coverage histogram (n queries with K hits in top-{args.top_k}) ===")
    print(f"{'hits':>6}  {'phase2b':>8}  {'audit':>8}  {'union':>8}")
    h2 = coverage_histogram(cov_phase2b)
    ha = coverage_histogram(cov_audit)
    hu = coverage_histogram(cov_union)
    for k in range(args.top_k + 1):
        print(f"{k:>6}  {h2.get(k, 0):>8}  {ha.get(k, 0):>8}  {hu.get(k, 0):>8}")

    # Relevance-score distribution of LABELED gecko top-3
    print(f"\n=== Score distribution of LABELED Gecko top-{args.top_k} chunks ===")
    score_phase2b = Counter()
    score_audit = Counter()
    score_union = Counter()
    n_labeled_union = 0
    for qid, top in gecko_topk.items():
        for cid in top:
            key = (qid, cid)
            in_p = key in phase2b
            in_a = key in audit_labels
            if in_p:
                score_phase2b[phase2b[key]] += 1
            if in_a:
                score_audit[audit_labels[key]] += 1
            if in_p or in_a:
                # Prefer audit label if both present (newer)
                score = audit_labels[key] if in_a else phase2b[key]
                score_union[score] += 1
                n_labeled_union += 1

    print(f"  labeled chunks (union): {n_labeled_union} / {len(gecko_topk) * args.top_k}")
    print(f"  score=0 (non-relevant):  union={score_union.get(0, 0)}  phase2b={score_phase2b.get(0, 0)}  audit={score_audit.get(0, 0)}")
    print(f"  score=1 (partially):     union={score_union.get(1, 0)}  phase2b={score_phase2b.get(1, 0)}  audit={score_audit.get(1, 0)}")
    print(f"  score=2 (fully relevant):union={score_union.get(2, 0)}  phase2b={score_phase2b.get(2, 0)}  audit={score_audit.get(2, 0)}")

    # Implied precision among labeled
    if n_labeled_union > 0:
        prec_lenient = (score_union.get(1, 0) + score_union.get(2, 0)) / n_labeled_union
        prec_strict = score_union.get(2, 0) / n_labeled_union
        print(f"\n  implied precision among LABELED top-{args.top_k}:")
        print(f"    lenient (score >= 1): {prec_lenient:.3f}")
        print(f"    strict  (score == 2): {prec_strict:.3f}")
        print(f"    (Note: only valid on labeled subset; metric on full set requires Step 4 if union coverage is low)")

    # Step 4 decision
    cu = agg(cov_union)
    print(f"\n=== Gate decision ===")
    if cu >= 0.80:
        print(f"  coverage_union = {cu:.3f} >= 0.80  ->  SKIP Step 4. Proceed to metrics with footnote.")
    elif cu >= 0.50:
        print(f"  coverage_union = {cu:.3f} in [0.50, 0.80)  ->  RUN Step 4 to clean up metrics.")
    else:
        print(f"  coverage_union = {cu:.3f} < 0.50  ->  MUST RUN Step 4. Metrics uninterpretable otherwise.")

    if args.show_zero_coverage_queries:
        zeros = sorted(qid for qid, v in cov_union.items() if v == 0)
        if zeros:
            print(f"\n=== Queries with zero union coverage ({len(zeros)}) ===")
            for qid in zeros[:20]:
                print(f"  {qid}")
            if len(zeros) > 20:
                print(f"  ... and {len(zeros) - 20} more")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
