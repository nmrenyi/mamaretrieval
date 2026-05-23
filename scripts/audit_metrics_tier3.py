#!/usr/bin/env python3
"""Tier 3 metrics — HR / Precision at varying k, plus Pool Recall.

Tier 3 judged the union of all 6 retrievers' top-20, so we can now compute
HR / Precision at deeper k (k=3, 5, 10, 20) and Pool Recall — metrics that
the Tier 2 top-3 labels couldn't support.

- **HR@k, P@k** at multiple k for both lenient (≥3) and strict (≥5)
  thresholds. Same deployment-honest convention as audit_metrics_v2.py:
  averaged over all queries; queries with no relevant chunk in the pool
  contribute HR=0 and P=0.

- **Pool Recall@k** per retriever: of the chunks the judge marked relevant
  in the union-of-top-20 pool, what fraction did THIS retriever's top-k
  capture? Diagnostic for "are weaker retrievers MISSING relevant chunks
  or just RANKING them lower?".

Both binary metrics use the lenient threshold by default for pool recall
(score >= 3); the rationale is that's the "useful clinical content"
threshold that matters for end-user value.

Inputs:
  --labels       data/audit/v2_top20_all.jsonl  (full top-20 union labels)
  --rankings-dir data/full                       (per-retriever top-20 files)
Outputs:
  --report data/audit/results_v2_tier3.md
  --raw    data/audit/results_v2_tier3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RETRIEVERS = ("bm25", "medcpt", "octen", "voyage", "lateon", "gecko")
K_VALUES = [3, 5, 10, 20]


def stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
    }


def score_v2(rec: dict) -> int:
    d1 = bool(rec.get("d1_topic"))
    if not d1:
        return 0
    return (rec.get("d2_meaningful", 0) or 0) \
         + (rec.get("d3_actionable", 0) or 0) \
         + (rec.get("d4_density", 0) or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/audit/v2_top20_all.jsonl")
    ap.add_argument("--rankings-dir", default="data/full")
    ap.add_argument("--report", default="data/audit/results_v2_tier3.md")
    ap.add_argument("--raw", default="data/audit/results_v2_tier3.json")
    ap.add_argument("--lenient", type=int, default=3)
    ap.add_argument("--strict", type=int, default=5)
    args = ap.parse_args()

    # Load labels
    score: dict[tuple[str, str], int] = {}
    qids: set[str] = set()
    for line in Path(args.labels).open():
        r = json.loads(line)
        if not r.get("llm_judge_schema_version", "").startswith("v-"):
            continue
        if r.get("_error"):
            continue
        score[(r["query_id"], r["chunk_id"])] = score_v2(r)
        qids.add(r["query_id"])
    print(f"Loaded {len(score)} labels across {len(qids)} queries", flush=True)

    # Build per-query relevance sets at each threshold
    rel_lenient: dict[str, set[str]] = defaultdict(set)
    rel_strict:  dict[str, set[str]] = defaultdict(set)
    for (q, c), s in score.items():
        if s >= args.lenient:
            rel_lenient[q].add(c)
        if s >= args.strict:
            rel_strict[q].add(c)

    # Load per-retriever rankings (full top-20)
    ranks: dict[str, dict[str, list[str]]] = {}
    for retr in RETRIEVERS:
        m: dict[str, list[str]] = {}
        for line in Path(f"{args.rankings_dir}/{retr}_top20.jsonl").open():
            rec = json.loads(line)
            m[rec["query_id"]] = [r["chunk_id"] for r in rec["results"]]
        ranks[retr] = m

    # --- Binary HR / P at each k, each threshold ---
    def binary_at_k(retr: str, rel_map: dict[str, set[str]], k: int) -> dict:
        hr, prec = [], []
        for q in qids:
            rel = rel_map.get(q, set())
            top = ranks[retr].get(q, [])[:k]
            hits = sum(1 for c in top if c in rel)
            hr.append(1.0 if hits > 0 else 0.0)
            prec.append(hits / k)
        return {"HR": stats(hr), "P": stats(prec)}

    # --- Pool Recall@k: fraction of relevant-in-pool that top-k captured ---
    def pool_recall_at_k(retr: str, rel_map: dict[str, set[str]], k: int) -> dict:
        """For each query with ≥1 relevant chunk in the judged pool, compute
        |retriever_top_k ∩ relevant| / |relevant|. Average over those queries.

        Note: 'relevant' here = relevant chunks in the judged 6-retriever
        top-20 union pool. Pool recall is bounded above by the union pool's
        coverage of the corpus, not absolute corpus recall.
        """
        recalls = []
        for q in qids:
            rel = rel_map.get(q, set())
            if not rel:
                continue
            top = set(ranks[retr].get(q, [])[:k])
            recalls.append(len(top & rel) / len(rel))
        return stats(recalls)

    per_retriever: dict[str, dict] = {}
    for retr in RETRIEVERS:
        rs: dict = {}
        for k in K_VALUES:
            rs[f"binary_lenient_@{k}"] = binary_at_k(retr, rel_lenient, k)
            rs[f"binary_strict_@{k}"] = binary_at_k(retr, rel_strict, k)
            rs[f"pool_recall_lenient_@{k}"] = pool_recall_at_k(retr, rel_lenient, k)
        per_retriever[retr] = rs

    # Raw
    raw = {
        "labels_path": args.labels,
        "rankings_dir": args.rankings_dir,
        "n_queries": len(qids),
        "n_pairs": len(score),
        "k_values": K_VALUES,
        "thresholds": {"lenient": args.lenient, "strict": args.strict},
        "n_queries_with_lenient_relevant": sum(1 for q in qids if rel_lenient.get(q)),
        "n_queries_with_strict_relevant": sum(1 for q in qids if rel_strict.get(q)),
        "per_retriever": per_retriever,
    }
    Path(args.raw).write_text(json.dumps(raw, indent=2))
    print(f"Raw    -> {args.raw}", flush=True)

    # Markdown
    def fmt(v: dict | None) -> str:
        return f"{v['mean']:.3f}" if v else "n/a"

    md = [
        "# Tier 3 metrics — HR / Precision / Pool Recall at varying k",
        "",
        f"> Auto-generated by `scripts/audit_metrics_tier3.py` from `{args.labels}`. "
        "Re-run the script to regenerate; do not hand-edit.",
        "",
        f"- Pool: **{len(score):,}** (q, c) pairs across **{len(qids)}** queries",
        f"- Rubric: v2 graded (score = d1 × (d2 + d3 + d4), range 0–6)",
        f"- Lenient threshold: score ≥ {args.lenient} "
        f"({sum(1 for q in qids if rel_lenient.get(q))} queries have ≥1 relevant)",
        f"- Strict threshold:  score ≥ {args.strict} "
        f"({sum(1 for q in qids if rel_strict.get(q))} queries have ≥1 relevant)",
        f"- k values: {', '.join(str(k) for k in K_VALUES)}",
        "",
        "This report builds on the Tier 3 full top-20 union judging — every "
        "retriever's top-20 chunks (~231k pairs total) was labeled by the v2 "
        "graded rubric. That lets us compute HR / Precision at deeper k and "
        "Pool Recall, which the Tier 2 top-3 labels couldn't support.",
        "",
        "---",
        "",
        "## HR / Precision — Binary lenient (score ≥ 3)",
        "",
    ]
    md.append("| Retriever | " + " | ".join(f"HR@{k}" for k in K_VALUES)
              + " | " + " | ".join(f"P@{k}" for k in K_VALUES) + " |")
    md.append("|---" + "|---:" * (2 * len(K_VALUES)) + "|")
    for retr in RETRIEVERS:
        row = [retr]
        for k in K_VALUES:
            row.append(fmt(per_retriever[retr][f"binary_lenient_@{k}"]["HR"]))
        for k in K_VALUES:
            row.append(fmt(per_retriever[retr][f"binary_lenient_@{k}"]["P"]))
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## HR / Precision — Binary strict (score ≥ 5)")
    md.append("")
    md.append("| Retriever | " + " | ".join(f"HR@{k}" for k in K_VALUES)
              + " | " + " | ".join(f"P@{k}" for k in K_VALUES) + " |")
    md.append("|---" + "|---:" * (2 * len(K_VALUES)) + "|")
    for retr in RETRIEVERS:
        row = [retr]
        for k in K_VALUES:
            row.append(fmt(per_retriever[retr][f"binary_strict_@{k}"]["HR"]))
        for k in K_VALUES:
            row.append(fmt(per_retriever[retr][f"binary_strict_@{k}"]["P"]))
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Pool Recall — lenient (score ≥ 3)")
    md.append("")
    md.append(
        "Of the chunks the judge marked relevant in the 6-retriever top-20 "
        "union pool, what fraction did THIS retriever's top-k capture? "
        "Diagnostic for *MISS vs RANK* — i.e. is a weaker retriever failing "
        "to *find* relevant chunks, or just *ranking* them lower?"
    )
    md.append("")
    md.append("Restricted to queries with ≥1 relevant chunk in the pool.")
    md.append("")
    md.append("| Retriever | " + " | ".join(f"Pool R@{k}" for k in K_VALUES) + " |")
    md.append("|---" + "|---:" * len(K_VALUES) + "|")
    for retr in RETRIEVERS:
        row = [retr]
        for k in K_VALUES:
            row.append(fmt(per_retriever[retr][f"pool_recall_lenient_@{k}"]))
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append(
        "- **Pool Recall is bounded by the 6-retriever union top-20**. A "
        "retriever can never have Pool Recall@20 > 1.0 even if it would "
        "find more relevant chunks at deeper k, because we only judged the "
        "first 20 chunks per retriever."
    )
    md.append(
        "- HR/P at k=3 should match Tier 2's results (top-3 union was "
        "labeled in both tiers). HR/P at k=5,10,20 only became measurable "
        "in Tier 3."
    )
    md.append(
        "- Judge agreement with Opus-4.7 reference labels: 95% at score ≥ 3, "
        "85% at score ≥ 5 (Tier 1 validation in "
        "`notes/rubric_design_worked_examples.md`)."
    )
    md.append("")
    Path(args.report).write_text("\n".join(md) + "\n")
    print(f"Report -> {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
