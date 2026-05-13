#!/usr/bin/env python3
"""Phase 3 audit — recall-gap metrics on the 100-query subset.

Treats the audit-augmented label set (Phase 2b ∪ audit) as ground truth and
computes three recall numbers:

  Variant A (headline) — Benchmark-pool recall.
    Fraction of truly-relevant chunks that Phase 2a's candidate pool actually
    surfaced as candidates (i.e., that have any Phase 2b label). Answers:
    "did the candidate-generation step cover the truly-relevant set?"

  Variant B (diagnostic) — Per-retriever recall@k.
    For each retriever (BM25, MedCPT, Octen, voyage-4-large, LateOn), fraction
    of truly-relevant chunks present in that retriever's top-k. Answers:
    "which retriever recovers more of the truth and at what depth?"

  Variant C (decomposition) — Union-pool recall by retriever-subset × k.
    Recall of the union of N retrievers' top-k against the truly-relevant set,
    sweeping retriever-subset and k together. Answers: "is the gap driven
    by retriever choice (model-bounded) or candidate depth (top-k-bounded)?"

All variants are reported at two relevance thresholds:
  - lenient: score ≥ 1 (on-topic, with or without actionable guidance)
  - strict:  score = 2 (actionable)

Outputs:
  data/audit/results.md   — Markdown report (headline + per-retriever table)
  data/audit/results.json — raw per-query stats, for any follow-up drill-down
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PER_RETRIEVER_INPUTS = {
    "bm25":   "data/audit/bm25_top20.jsonl",
    "medcpt": "data/audit/medcpt_top20.jsonl",
    "octen":  "data/audit/octen_top20.jsonl",
    "voyage": "data/audit/voyage_top20.jsonl",
    "lateon": "data/audit/lateon_top20.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {
        "n_queries_used": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument("--phase2b-labels", default="data/relevance_labels.jsonl")
    p.add_argument("--audit-labels", default="data/audit/relevance_labels_audit.jsonl")
    p.add_argument("--report", default="data/audit/results.md")
    p.add_argument("--raw", default="data/audit/results.json")
    p.add_argument("--cutoffs", nargs="+", type=int, default=[5, 10, 20])
    args = p.parse_args()

    audit_qids = {
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    }
    print(f"Audit queries: {len(audit_qids)}", flush=True)

    # Phase 2b labels filtered to the audit query subset
    phase2b_score: dict[tuple[str, str], int] = {}
    for line in Path(args.phase2b_labels).open():
        r = json.loads(line)
        if r["query_id"] in audit_qids:
            phase2b_score[(r["query_id"], r["chunk_id"])] = r["score"]
    print(f"Phase 2b labels for audit queries: {len(phase2b_score)}", flush=True)

    # Audit-only labels (the 3,939 new pairs)
    audit_score: dict[tuple[str, str], int] = {}
    for line in Path(args.audit_labels).open():
        r = json.loads(line)
        audit_score[(r["query_id"], r["chunk_id"])] = r["score"]
    print(f"Audit-only labels: {len(audit_score)}", flush=True)

    overlap = set(phase2b_score) & set(audit_score)
    if overlap:
        print(
            f"WARN: {len(overlap)} (q,c) pairs are labeled by BOTH Phase 2b "
            "and audit — using audit value.",
            file=sys.stderr,
        )
    augmented = {**phase2b_score, **audit_score}
    print(f"Audit-augmented label set: {len(augmented)} pairs", flush=True)

    # Per-query relevant sets and Phase 2a pool
    qid_relevant_lenient: dict[str, set[str]] = defaultdict(set)
    qid_relevant_strict:  dict[str, set[str]] = defaultdict(set)
    qid_pool:             dict[str, set[str]] = defaultdict(set)
    for (qid, cid), score in augmented.items():
        if score >= 1:
            qid_relevant_lenient[qid].add(cid)
        if score == 2:
            qid_relevant_strict[qid].add(cid)
    for (qid, _cid) in phase2b_score:
        qid_pool[qid].add(_cid)

    # ──────── Variant A: benchmark-pool recall ────────
    def pool_recall(qid_relevant: dict[str, set[str]]) -> list[float]:
        out = []
        for qid in audit_qids:
            rel = qid_relevant.get(qid, set())
            if not rel:
                continue
            pool = qid_pool.get(qid, set())
            out.append(len(pool & rel) / len(rel))
        return out

    variant_a = {
        "lenient_score>=1": stats(pool_recall(qid_relevant_lenient)),
        "strict_score==2":  stats(pool_recall(qid_relevant_strict)),
    }

    # ──────── Variant B: per-retriever recall@k ────────
    per_retriever_ranks: dict[str, dict[str, list[str]]] = {}
    for retriever, path in PER_RETRIEVER_INPUTS.items():
        m: dict[str, list[str]] = {}
        for rec in load_jsonl(Path(path)):
            m[rec["query_id"]] = [r["chunk_id"] for r in rec["results"]]
        per_retriever_ranks[retriever] = m

    variant_b: dict[str, dict[str, dict | None]] = {}
    for retriever, ranks in per_retriever_ranks.items():
        retriever_results: dict[str, dict | None] = {}
        for label_name, qid_relevant in [
            ("lenient_score>=1", qid_relevant_lenient),
            ("strict_score==2",  qid_relevant_strict),
        ]:
            for k in args.cutoffs:
                recalls = []
                for qid in audit_qids:
                    rel = qid_relevant.get(qid, set())
                    if not rel:
                        continue
                    top_k = set(ranks.get(qid, [])[:k])
                    recalls.append(len(top_k & rel) / len(rel))
                retriever_results[f"{label_name}@{k}"] = stats(recalls)
        variant_b[retriever] = retriever_results

    # ──────── Variant C: union-pool recall (model-vs-k decomposition) ────────
    # Same recall computation as Variant A, but the "pool" is rebuilt from the
    # audit per-retriever rankings — so we can sweep retriever-subset and k.
    # Lets us answer: is the benchmark's recall gap driven by retriever choice
    # (model-bounded) or by candidate depth (top-k-bounded)?
    ORIGINAL_3 = ["bm25", "medcpt", "octen"]
    ALL_5 = list(PER_RETRIEVER_INPUTS)

    def union_pool_recall(
        retrievers: list[str], k: int, qid_relevant: dict[str, set[str]]
    ) -> list[float]:
        out = []
        for qid in audit_qids:
            rel = qid_relevant.get(qid, set())
            if not rel:
                continue
            union: set[str] = set()
            for retriever in retrievers:
                union.update(per_retriever_ranks[retriever].get(qid, [])[:k])
            out.append(len(union & rel) / len(rel))
        return out

    variant_c: dict[str, dict[str, dict | None]] = {}
    for label, subset in [("3_retrievers_bm25+medcpt+octen", ORIGINAL_3),
                          ("5_retrievers_all", ALL_5)]:
        rows: dict[str, dict | None] = {}
        for k in args.cutoffs:
            for thresh_label, qid_rel in [
                ("lenient_score>=1", qid_relevant_lenient),
                ("strict_score==2",  qid_relevant_strict),
            ]:
                rows[f"{thresh_label}@{k}"] = stats(
                    union_pool_recall(subset, k, qid_rel)
                )
        variant_c[label] = rows

    # ──────── Raw JSON ────────
    raw = {
        "audit_queries": len(audit_qids),
        "label_counts": {
            "phase2b_audit_queries": len(phase2b_score),
            "audit_only": len(audit_score),
            "augmented_total": len(augmented),
            "overlap_phase2b_and_audit": len(overlap),
        },
        "per_query_sizes": {
            "pool":             {
                "mean": sum(len(v) for v in qid_pool.values()) / len(audit_qids),
                "min":  min(len(v) for v in qid_pool.values()),
                "max":  max(len(v) for v in qid_pool.values()),
            },
            "relevant_lenient": {
                "mean": sum(len(v) for v in qid_relevant_lenient.values()) / len(audit_qids),
                "min":  min((len(v) for v in qid_relevant_lenient.values()), default=0),
                "max":  max((len(v) for v in qid_relevant_lenient.values()), default=0),
            },
            "relevant_strict":  {
                "mean": sum(len(v) for v in qid_relevant_strict.values()) / len(audit_qids),
                "min":  min((len(v) for v in qid_relevant_strict.values()), default=0),
                "max":  max((len(v) for v in qid_relevant_strict.values()), default=0),
            },
        },
        "cutoffs": args.cutoffs,
        "variant_a_benchmark_pool": variant_a,
        "variant_b_per_retriever":  variant_b,
        "variant_c_union_pool_by_subset_and_k": variant_c,
    }
    Path(args.raw).write_text(json.dumps(raw, indent=2))
    print(f"Raw results -> {args.raw}", flush=True)

    # ──────── Markdown report ────────
    def fmt(v: dict | None) -> str:
        return f"{v['mean']:.3f}" if v else "n/a"

    md = [
        "# Phase 3 audit — recall-gap results",
        "",
        f"Computed over **{len(audit_qids)} audit queries** with the audit-augmented label set treated as ground truth.",
        "",
        f"- Audit-augmented labels: **{len(augmented):,} (q,c) pairs** "
        f"(Phase 2b: {len(phase2b_score):,} + audit-only: {len(audit_score):,})",
        f"- Avg Phase 2a pool size per query: **{raw['per_query_sizes']['pool']['mean']:.1f}**",
        f"- Avg relevant chunks per query (lenient, score ≥ 1): **{raw['per_query_sizes']['relevant_lenient']['mean']:.1f}**",
        f"- Avg relevant chunks per query (strict, score = 2): **{raw['per_query_sizes']['relevant_strict']['mean']:.1f}**",
        "",
        "---",
        "",
        "## Variant A — Benchmark-pool recall (headline)",
        "",
        "Fraction of truly-relevant chunks (audit-augmented) that Phase 2a's candidate pool surfaced.",
        "",
        "| Threshold | Queries used | Mean recall | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, s in variant_a.items():
        if s is None:
            md.append(f"| {label} | 0 | n/a | n/a | n/a |")
        else:
            md.append(
                f"| {label} | {s['n_queries_used']} | "
                f"**{s['mean']:.3f}** | {s['min']:.3f} | {s['max']:.3f} |"
            )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Variant B — Per-retriever recall@k (diagnostic)")
    md.append("")
    md.append("Each retriever's top-k against the audit-augmented relevant set.")
    md.append("")
    for label_name in ["lenient_score>=1", "strict_score==2"]:
        md.append(f"### Threshold: `{label_name}`")
        md.append("")
        head = ["Retriever"] + [f"Recall@{k}" for k in args.cutoffs]
        md.append("| " + " | ".join(head) + " |")
        md.append("|---" + "|---:" * len(args.cutoffs) + "|")
        for retriever in PER_RETRIEVER_INPUTS:
            row = [retriever]
            for k in args.cutoffs:
                row.append(fmt(variant_b[retriever].get(f"{label_name}@{k}")))
            md.append("| " + " | ".join(row) + " |")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Variant C — Union-pool recall: model vs top-k decomposition")
    md.append("")
    md.append(
        "Recall of the *union* of N retrievers' top-k against the audit-augmented "
        "relevant set, sweeping retriever-subset × k. The 5-retriever @ top-20 "
        "row is 1.000 by construction (it defines the reference). The other "
        "rows isolate the lever: more retrievers (down a column) vs deeper k "
        "(across a row)."
    )
    md.append("")
    for thresh_label in ["lenient_score>=1", "strict_score==2"]:
        md.append(f"### Threshold: `{thresh_label}`")
        md.append("")
        head = ["Retriever subset"] + [f"Union@{k}" for k in args.cutoffs]
        md.append("| " + " | ".join(head) + " |")
        md.append("|---" + "|---:" * len(args.cutoffs) + "|")
        for subset_label in ["3_retrievers_bm25+medcpt+octen", "5_retrievers_all"]:
            row = [subset_label]
            for k in args.cutoffs:
                row.append(fmt(variant_c[subset_label].get(f"{thresh_label}@{k}")))
            md.append("| " + " | ".join(row) + " |")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Provenance")
    md.append("")
    md.append(f"- Audit query IDs: `{args.query_ids}`")
    md.append(f"- Phase 2b labels: `{args.phase2b_labels}` (filtered to audit queries)")
    md.append(f"- Audit labels:    `{args.audit_labels}`")
    md.append(f"- Per-retriever rankings: " + ", ".join(f"`{p}`" for p in PER_RETRIEVER_INPUTS.values()))
    md.append("")
    Path(args.report).write_text("\n".join(md) + "\n")
    print(f"Report     -> {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
