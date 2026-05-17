#!/usr/bin/env python3
"""Variant D over v2 graded labels — top-k deployment metrics per retriever.

Reads the v2 pilot labels (score = d1 × (d2 + d3 + d4), range 0–6) and the six
top-20 retriever files; computes HR@k, Precision@k, MRR@k, and graded NDCG@k
per retriever at a lenient (T≥3) and a strict (T≥5) threshold.

NDCG uses graded scores (0–6) directly via gain = 2^score − 1. The "ideal" top-k
for NDCG is computed from the judged pool only (the union of retrievers' top-k
that was labeled), so it reflects the best ranking achievable given what we
chose to judge — same convention as audit_metrics.py.

Inputs:
  --labels    data/audit/v2_pilot_h100_shard0.jsonl  (v2 graded judgments)
  per-retriever top-20 files (hardcoded paths per PER_RETRIEVER_INPUTS)

Outputs:
  --report    data/audit/results_v2.md
  --raw       data/audit/results_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
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


def gain(s: int) -> float:
    return (2 ** s) - 1 if s > 0 else 0.0


def dcg(scores: list[int]) -> float:
    return sum(gain(s) / math.log2(i + 2) for i, s in enumerate(scores))


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

    # Per-qid relevant sets at the two thresholds + score lists for NDCG ideal
    rel_lenient: dict[str, set[str]] = defaultdict(set)
    rel_strict: dict[str, set[str]] = defaultdict(set)
    qid_scores: dict[str, list[int]] = defaultdict(list)
    for (q, c), s in score.items():
        if s >= args.lenient:
            rel_lenient[q].add(c)
        if s >= args.strict:
            rel_strict[q].add(c)
        if s > 0:
            qid_scores[q].append(s)

    # Load per-retriever rankings
    ranks: dict[str, dict[str, list[str]]] = {}
    for retriever, path in PER_RETRIEVER_INPUTS.items():
        m: dict[str, list[str]] = {}
        for line in Path(path).open():
            rec = json.loads(line)
            m[rec["query_id"]] = [r["chunk_id"] for r in rec["results"]]
        ranks[retriever] = m

    def metrics_for(retriever: str, rel_map: dict[str, set[str]]) -> dict:
        k = args.k
        hr, prec, mrr = [], [], []
        for q in qids:
            rel = rel_map.get(q, set())
            if not rel:
                continue
            top = ranks[retriever].get(q, [])[:k]
            hits = [1 if c in rel else 0 for c in top]
            hr.append(1.0 if sum(hits) > 0 else 0.0)
            prec.append(sum(hits) / k)
            first = next((i + 1 for i, h in enumerate(hits) if h == 1), None)
            mrr.append(1.0 / first if first else 0.0)
        return {
            f"HR@{k}": stats(hr),
            f"P@{k}": stats(prec),
            f"MRR@{k}": stats(mrr),
        }

    def weighted_metrics(retriever: str) -> dict:
        """Threshold-free weighted variants — each chunk contributes score/6."""
        k = args.k
        whr, wprec, wmrr = [], [], []
        rank_weights = [1.0 / (i + 1) for i in range(k)]  # 1/1, 1/2, 1/3, ...
        rank_sum = sum(rank_weights)
        for q in qids:
            # Use any query that has at least one judged chunk in the pool
            if not qid_scores.get(q):
                continue
            top = ranks[retriever].get(q, [])[:k]
            top_scores = [score.get((q, c), 0) / 6.0 for c in top]
            # Pad with 0 if retriever returned fewer than k
            while len(top_scores) < k:
                top_scores.append(0.0)
            whr.append(max(top_scores))
            wprec.append(sum(top_scores) / k)
            wmrr.append(
                sum(s * w for s, w in zip(top_scores, rank_weights)) / rank_sum
            )
        return {
            f"wHR@{k}": stats(whr),
            f"wP@{k}": stats(wprec),
            f"wMRR@{k}": stats(wmrr),
        }

    def ndcg_for(retriever: str) -> dict | None:
        k = args.k
        out = []
        for q in qids:
            ideal = sorted(qid_scores.get(q, []), reverse=True)[:k]
            if not ideal:
                continue
            idcg = dcg(ideal)
            if idcg == 0:
                continue
            top = ranks[retriever].get(q, [])[:k]
            top_scores = [score.get((q, c), 0) for c in top]
            out.append(dcg(top_scores) / idcg)
        return stats(out)

    variant_d: dict[str, dict] = {}
    for retriever in PER_RETRIEVER_INPUTS:
        variant_d[retriever] = {
            f"lenient_score>={args.lenient}": metrics_for(retriever, rel_lenient),
            f"strict_score>={args.strict}": metrics_for(retriever, rel_strict),
            "weighted_by_score_over_6": weighted_metrics(retriever),
            f"NDCG@{args.k}_graded": ndcg_for(retriever),
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
        "variant_d_per_retriever": variant_d,
    }
    Path(args.raw).write_text(json.dumps(raw, indent=2))
    print(f"Raw    -> {args.raw}", flush=True)

    # Markdown
    def fmt(v: dict | None) -> str:
        return f"{v['mean']:.3f}" if v else "n/a"

    md = [
        "# v2 graded — Variant D (top-k deployment metrics)",
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
    for thresh_label in [
        f"lenient_score>={args.lenient}",
        f"strict_score>={args.strict}",
    ]:
        md.append(f"## Threshold: `{thresh_label}`")
        md.append("")
        md.append(
            f"| Retriever | HR@{args.k} | P@{args.k} | MRR@{args.k} | "
            f"NDCG@{args.k} (graded) | n queries |"
        )
        md.append("|---|---:|---:|---:|---:|---:|")
        for retriever in PER_RETRIEVER_INPUTS:
            d = variant_d[retriever][thresh_label]
            ndcg = variant_d[retriever][f"NDCG@{args.k}_graded"]
            n_used = d[f"HR@{args.k}"]["n_queries_used"] if d[f"HR@{args.k}"] else 0
            md.append(
                f"| {retriever} | {fmt(d[f'HR@{args.k}'])} | "
                f"{fmt(d[f'P@{args.k}'])} | {fmt(d[f'MRR@{args.k}'])} | "
                f"{fmt(ndcg)} | {n_used} |"
            )
        md.append("")
    # Weighted (threshold-free) section
    md.append(f"## Weighted by score/6 (threshold-free, range [0, 1])")
    md.append("")
    md.append(
        "Each chunk contributes its score normalized to [0, 1] (perfect 6 = 1.0, "
        "score 3 = 0.5, score 0 = 0). No binary threshold; the full graded signal "
        "shapes every metric."
    )
    md.append("")
    md.append(
        "- **wHR@k** = max(score_i/6) over top-k — best chunk seen\n"
        "- **wP@k** = mean(score_i/6) over top-k — average chunk quality\n"
        "- **wMRR@k** = Σ(score_i/6 × 1/rank_i) / Σ(1/rank_i) — rank-weighted "
        "average quality"
    )
    md.append("")
    md.append(
        f"| Retriever | wHR@{args.k} | wP@{args.k} | wMRR@{args.k} | "
        f"NDCG@{args.k} (graded) | n queries |"
    )
    md.append("|---|---:|---:|---:|---:|---:|")
    for retriever in PER_RETRIEVER_INPUTS:
        w = variant_d[retriever]["weighted_by_score_over_6"]
        ndcg = variant_d[retriever][f"NDCG@{args.k}_graded"]
        n_used = w[f"wHR@{args.k}"]["n_queries_used"] if w[f"wHR@{args.k}"] else 0
        md.append(
            f"| {retriever} | {fmt(w[f'wHR@{args.k}'])} | "
            f"{fmt(w[f'wP@{args.k}'])} | {fmt(w[f'wMRR@{args.k}'])} | "
            f"{fmt(ndcg)} | {n_used} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append(
        "- NDCG ideal is computed from the **judged pool** for each query "
        "(the union of retrievers' top-3 that was labeled in the pilot). "
        "This is the standard convention when judging is incomplete; "
        "NDCG values are comparable across retrievers but not directly "
        "comparable to NDCG against the full corpus."
    )
    md.append(
        "- HR / P / MRR are computed only over queries that have at least one "
        "chunk meeting the threshold (\"n queries\" column)."
    )
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
