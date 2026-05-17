#!/usr/bin/env python3
"""Variant D over v2 graded labels — top-k deployment metrics per retriever.

Reads the v2 pilot labels (score = d1 × (d2 + d3 + d4), range 0–6) and the six
top-20 retriever files; computes Hit-Rate and Precision per retriever in two
flavors:

  Binary  — at a lenient (T≥3) and a strict (T≥5) threshold on the score.
  Weighted — each chunk contributes score/6 ∈ [0, 1]; no threshold.

For RAG at k=3 with a long-context LLM consuming all three chunks (no position
bias), HR ("is the information present in the bundle?") and Precision ("how
much of the bundle is useful vs noise?") are the operationally meaningful
metrics. MRR and NDCG add no information beyond these in that setting and are
omitted from this report.

Inputs:
  --labels    data/audit/v2_pilot_h100_shard0.jsonl  (v2 graded judgments)
  per-retriever top-20 files (hardcoded in PER_RETRIEVER_INPUTS)

Outputs:
  --report    data/audit/results_v2.md
  --raw       data/audit/results_v2.json
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
    "gecko":  "data/audit/gecko_top20.jsonl",
}


def stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {
        "n_queries_used": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
    }


def score_v2(rec: dict) -> int:
    """Compute v2 score = d1 × (d2 + d3 + d4) from a label record."""
    d1 = bool(rec.get("d1_topic"))
    if not d1:
        return 0
    return (rec.get("d2_meaningful", 0) or 0) \
         + (rec.get("d3_actionable", 0) or 0) \
         + (rec.get("d4_density", 0) or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/audit/v2_pilot_h100_shard0.jsonl")
    ap.add_argument("--report", default="data/audit/results_v2.md")
    ap.add_argument("--raw", default="data/audit/results_v2.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lenient", type=int, default=3,
                    help="lenient relevance threshold on the 0–6 score (default 3)")
    ap.add_argument("--strict", type=int, default=5,
                    help="strict relevance threshold (default 5)")
    args = ap.parse_args()

    # Load v2 labels
    score: dict[tuple[str, str], int] = {}
    qids: set[str] = set()
    skipped = 0
    for line in Path(args.labels).open():
        r = json.loads(line)
        sv = r.get("llm_judge_schema_version", "")
        if not sv.startswith("v-"):
            skipped += 1
            continue
        if r.get("_error"):
            skipped += 1
            continue
        s = score_v2(r)
        score[(r["query_id"], r["chunk_id"])] = s
        qids.add(r["query_id"])
    print(f"v2 labels: {len(score)} pairs across {len(qids)} queries "
          f"(skipped {skipped} non-v2 / errored)", flush=True)

    # Per-qid relevant sets at the two thresholds
    rel_lenient: dict[str, set[str]] = defaultdict(set)
    rel_strict: dict[str, set[str]] = defaultdict(set)
    qid_has_any_judged: set[str] = set()
    for (q, c), s in score.items():
        if s >= args.lenient:
            rel_lenient[q].add(c)
        if s >= args.strict:
            rel_strict[q].add(c)
        qid_has_any_judged.add(q)

    # Load per-retriever rankings
    ranks: dict[str, dict[str, list[str]]] = {}
    for retriever, path in PER_RETRIEVER_INPUTS.items():
        m: dict[str, list[str]] = {}
        for line in Path(path).open():
            rec = json.loads(line)
            m[rec["query_id"]] = [r["chunk_id"] for r in rec["results"]]
        ranks[retriever] = m

    def binary_metrics(retriever: str, rel_map: dict[str, set[str]]) -> dict:
        """HR / P over all judged queries. Queries with no chunk meeting the
        threshold contribute HR=0 and P=0 (deployment-honest convention)."""
        k = args.k
        hr, prec = [], []
        for q in qids:
            if q not in qid_has_any_judged:
                continue
            rel = rel_map.get(q, set())
            top = ranks[retriever].get(q, [])[:k]
            hits = sum(1 for c in top if c in rel)
            hr.append(1.0 if hits > 0 else 0.0)
            prec.append(hits / k)
        return {f"HR@{k}": stats(hr), f"P@{k}": stats(prec)}

    def weighted_metrics(retriever: str) -> dict:
        """Threshold-free: each chunk contributes score/6 ∈ [0, 1]."""
        k = args.k
        whr, wprec = [], []
        for q in qids:
            if q not in qid_has_any_judged:
                continue
            top = ranks[retriever].get(q, [])[:k]
            top_scores = [score.get((q, c), 0) / 6.0 for c in top]
            while len(top_scores) < k:
                top_scores.append(0.0)
            whr.append(max(top_scores))
            wprec.append(sum(top_scores) / k)
        return {f"wHR@{k}": stats(whr), f"wP@{k}": stats(wprec)}

    per_retriever: dict[str, dict] = {}
    for retriever in PER_RETRIEVER_INPUTS:
        per_retriever[retriever] = {
            f"binary_lenient(>={args.lenient})":
                binary_metrics(retriever, rel_lenient),
            f"binary_strict(>={args.strict})":
                binary_metrics(retriever, rel_strict),
            "weighted_by_score_over_6":
                weighted_metrics(retriever),
        }

    score_dist = {s: sum(1 for v in score.values() if v == s) for s in range(7)}
    n_with_lenient_rel = sum(1 for q in qids if rel_lenient.get(q))
    n_with_strict_rel = sum(1 for q in qids if rel_strict.get(q))

    raw = {
        "labels_path": args.labels,
        "n_queries": len(qids),
        "n_pairs": len(score),
        "k": args.k,
        "thresholds": {"lenient": args.lenient, "strict": args.strict},
        "score_distribution": score_dist,
        "n_queries_with_relevant": {
            f"lenient(>={args.lenient})": n_with_lenient_rel,
            f"strict(>={args.strict})": n_with_strict_rel,
        },
        "per_retriever": per_retriever,
    }
    Path(args.raw).write_text(json.dumps(raw, indent=2))
    print(f"Raw    -> {args.raw}", flush=True)

    # Markdown
    def fmt(v: dict | None) -> str:
        return f"{v['mean']:.3f}" if v else "n/a"

    md = [
        "# v2 graded — Variant D (HR / Precision at k=3)",
        "",
        f"> Auto-generated by `scripts/audit_metrics_v2.py` from `{args.labels}`. "
        "Re-run the script to regenerate; do not hand-edit.",
        "",
        f"- Pilot pairs: **{len(score):,}** across **{len(qids)}** queries",
        f"- Rubric: v2 graded (score = d1 × (d2 + d3 + d4), range 0–6)",
        f"- Lenient threshold: **score ≥ {args.lenient}** "
        f"({n_with_lenient_rel} queries have ≥1 relevant chunk)",
        f"- Strict threshold:  **score ≥ {args.strict}** "
        f"({n_with_strict_rel} queries have ≥1 relevant chunk)",
        f"- Cutoff: k = **{args.k}**",
        "",
        "For RAG at k=3 with a long-context LLM (no position bias), only HR "
        "(\"is the info in the bundle?\") and Precision (\"how much of the bundle "
        "is useful?\") are operationally meaningful. MRR / NDCG would matter if "
        "position within top-k drove downstream behavior; here it doesn't.",
        "",
        "## Pilot score distribution",
        "",
        "| Score | Count | Cumulative ≥ |",
        "|---:|---:|---:|",
    ]
    total = len(score)
    cum = total
    rows = []
    for s in range(7):
        n = score_dist[s]
        rows.append((s, n, cum))
        cum -= n
    for s, n, c in rows:
        md.append(f"| {s} | {n} | {c} |")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## Definitions (k=3)")
    md.append("")
    md.append(
        "**Binary HR / P** — chunk is \"relevant\" if its score is ≥ threshold, "
        "0 otherwise. "
        "HR = 1 if any chunk in top-k is relevant, else 0. "
        "P = (count of relevant in top-k) / k. "
        "Averaged over **all** judged queries; queries with no relevant chunk "
        "in the pool contribute HR=0 and P=0 (deployment-honest convention)."
    )
    md.append("")
    md.append(
        "**Weighted HR / P (wHR / wP)** — threshold-free. Each chunk contributes "
        "its normalized score, score/6 ∈ [0, 1] (6 → 1.0, 3 → 0.5, 0 → 0). "
        "wHR = max contribution in top-k (best chunk seen). "
        "wP = mean contribution in top-k (average chunk quality). "
        "Denominator is all queries."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Per-retriever metrics @ k=3")
    md.append("")

    # Pull n used per metric family
    lenient_n = next(
        (per_retriever[r][f"binary_lenient(>={args.lenient})"][f"HR@{args.k}"]["n_queries_used"]
         for r in PER_RETRIEVER_INPUTS
         if per_retriever[r][f"binary_lenient(>={args.lenient})"][f"HR@{args.k}"]),
        0,
    )
    strict_n = next(
        (per_retriever[r][f"binary_strict(>={args.strict})"][f"HR@{args.k}"]["n_queries_used"]
         for r in PER_RETRIEVER_INPUTS
         if per_retriever[r][f"binary_strict(>={args.strict})"][f"HR@{args.k}"]),
        0,
    )
    weighted_n = next(
        (per_retriever[r]["weighted_by_score_over_6"][f"wHR@{args.k}"]["n_queries_used"]
         for r in PER_RETRIEVER_INPUTS
         if per_retriever[r]["weighted_by_score_over_6"][f"wHR@{args.k}"]),
        0,
    )

    md.append(
        f"| Retriever "
        f"| HR (≥{args.lenient}) | P (≥{args.lenient}) "
        f"| HR (≥{args.strict}) | P (≥{args.strict}) "
        f"| wHR | wP |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for retriever in PER_RETRIEVER_INPUTS:
        bl = per_retriever[retriever][f"binary_lenient(>={args.lenient})"]
        bs = per_retriever[retriever][f"binary_strict(>={args.strict})"]
        w  = per_retriever[retriever]["weighted_by_score_over_6"]
        md.append(
            f"| {retriever} "
            f"| {fmt(bl[f'HR@{args.k}'])} | {fmt(bl[f'P@{args.k}'])} "
            f"| {fmt(bs[f'HR@{args.k}'])} | {fmt(bs[f'P@{args.k}'])} "
            f"| {fmt(w[f'wHR@{args.k}'])} | {fmt(w[f'wP@{args.k}'])} |"
        )
    md.append("")
    md.append(
        f"All metrics averaged over n={lenient_n} queries (the full audit set)."
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append(
        "- Judge agreement with Opus-4.7 reference labels: 95% at score ≥ 3, "
        "85% at score ≥ 5. See `notes/rubric_design_worked_examples.md` "
        "Tier 1 pilot validation section."
    )
    md.append("")
    Path(args.report).write_text("\n".join(md) + "\n")
    print(f"Report -> {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
