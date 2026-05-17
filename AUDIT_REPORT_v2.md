# Audit Report — v2 graded rubric (Phase 4)

> **Headline retrieval evaluation** using the v2 graded rubric (Qwen3.5-397B-A17B-FP8 judge, score = d1 × (d2 + d3 + d4) ∈ {0..6}). For the v1 audit narrative, see [`AUDIT_REPORT.md`](AUDIT_REPORT.md). For rubric design and judge calibration, see [`notes/rubric_design_worked_examples.md`](notes/rubric_design_worked_examples.md).

## Overview

Two-tier evaluation of 6 retrievers (bm25, medcpt, octen, voyage-4-large, lateon, gecko) at deployment depth k=3:

- **Tier 1 pilot** (2026-05-17): 100 audit queries × 6 retrievers, **1,150 (q, c) pairs**. Used to validate the v2 graded rubric and the production Qwen judge against Claude Opus 4.7 reference labels before scaling. See [`notes/rubric_design_worked_examples.md` § Tier 1 pilot validation](notes/rubric_design_worked_examples.md#tier-1-pilot-validation--qwen-judge-vs-opus-47-reference-labels-2026-05-17) for the calibration data (95% threshold agreement at score ≥ 3, 85% at ≥ 5).
- **Tier 2 full audit** (2026-05-18): 3,185 queries × 6 retrievers, **36,418 (q, c) pairs**. Same rubric, same judge, same thinking budgets as Tier 1.

For RAG at k=3 with a long-context LLM consuming all three chunks (no position bias), **HR and Precision are the operationally meaningful metrics** — "is the information in the bundle?" and "how much of the bundle is useful vs noise?". MRR / NDCG would only matter if position within top-k drove downstream behavior; here it doesn't.

Three lenses on the same six retrievers:
- **Binary lenient (≥3)** — at least some clinically useful content
- **Binary strict (≥5)** — top-tier complete/specific content
- **Weighted (wHR / wP)** — threshold-free; each chunk contributes score/6 ∈ [0, 1]

All metrics use the deployment-honest convention: queries with no chunk meeting the threshold contribute HR=0 and P=0 (don't exclude them — the user will hit those queries in production).

---

## Tier 2 scoreboard (n=3,185 queries, headline result)

Computed by `scripts/audit_metrics_v2.py --labels data/audit/v2_full_h100.jsonl --rankings-dir data/full`.

| Retriever | HR (≥3) | P (≥3) | HR (≥5) | P (≥5) | wHR | wP |
|---|---:|---:|---:|---:|---:|---:|
| **voyage** | **0.996** | **0.867** | **0.753** | **0.452** | **0.860** | **0.682** |
| octen      | 0.991 | 0.804 | 0.716 | 0.403 | 0.847 | 0.637 |
| lateon     | 0.971 | 0.738 | 0.664 | 0.350 | 0.815 | 0.581 |
| gecko      | 0.814 | 0.477 | 0.439 | 0.193 | 0.662 | 0.393 |
| bm25       | 0.754 | 0.417 | 0.371 | 0.163 | 0.602 | 0.338 |
| medcpt     | 0.644 | 0.334 | 0.272 | 0.112 | 0.517 | 0.277 |

### Three-tier reading

1. **Top tier — voyage > octen > lateon**: all three deliver any-relevant content in 97-99% of queries. Voyage takes a clear lead at scale (was tied with octen at Tier 1): higher P(≥3) by ~6 pp, higher wP by ~5 pp.
2. **Middle — gecko** (on-device deployed retriever): HR drops to 0.81 lenient / 0.44 strict; precision roughly half the top tier. Substantial structural gap (see [issue #17](https://github.com/nmrenyi/mamaretrieval/issues/17) for remediation options).
3. **Bottom — bm25, medcpt**: bm25's lexical overlap matches gecko on raw HR(≥3) but with lower precision; medcpt is the weakest across every metric.

### Honest read of the strict numbers

Even the best retriever (voyage) surfaces a strict-relevant (score ≥ 5) chunk in only **75% of queries** with **45% of the top-3 being strict-relevant**. The ~25% of queries with no strict-relevant chunk in any retriever's top-3 is an inherent ceiling at depth-3 — pushing past it would require either deeper k or a broader candidate pool, not just a better re-ranker.

---

## Tier 1 (n=100) vs Tier 2 (n=3,185) — confirmation at scale

### Tier 1 scoreboard

| Retriever | HR (≥3) | P (≥3) | HR (≥5) | P (≥5) | wHR | wP |
|---|---:|---:|---:|---:|---:|---:|
| **voyage** | **0.990** | **0.820** | **0.730** | **0.430** | **0.847** | **0.657** |
| octen      | 0.990 | 0.760 | 0.720 | 0.413 | 0.845 | 0.624 |
| lateon     | 0.990 | 0.727 | 0.710 | 0.380 | 0.833 | 0.586 |
| gecko      | 0.840 | 0.493 | 0.490 | 0.210 | 0.693 | 0.404 |
| bm25       | 0.740 | 0.413 | 0.390 | 0.163 | 0.613 | 0.336 |
| medcpt     | 0.610 | 0.287 | 0.310 | 0.117 | 0.523 | 0.259 |

### T1 → T2 deltas

| Retriever | HR(≥3) T1→T2 | P(≥3) T1→T2 | HR(≥5) T1→T2 | P(≥5) T1→T2 | wHR T1→T2 | wP T1→T2 |
|---|---:|---:|---:|---:|---:|---:|
| voyage | 0.990 → 0.996 | 0.820 → 0.867 | 0.730 → 0.753 | 0.430 → 0.452 | 0.847 → 0.860 | 0.657 → 0.682 |
| octen  | 0.990 → 0.991 | 0.760 → 0.804 | 0.720 → 0.716 | 0.413 → 0.403 | 0.845 → 0.847 | 0.624 → 0.637 |
| lateon | 0.990 → 0.971 | 0.727 → 0.738 | 0.710 → 0.664 | 0.380 → 0.350 | 0.833 → 0.815 | 0.586 → 0.581 |
| gecko  | 0.840 → 0.814 | 0.493 → 0.477 | 0.490 → 0.439 | 0.210 → 0.193 | 0.693 → 0.662 | 0.404 → 0.393 |
| bm25   | 0.740 → 0.754 | 0.413 → 0.417 | 0.390 → 0.371 | 0.163 → 0.163 | 0.613 → 0.602 | 0.336 → 0.338 |
| medcpt | 0.610 → 0.644 | 0.287 → 0.334 | 0.310 → 0.272 | 0.117 → 0.112 | 0.523 → 0.517 | 0.259 → 0.277 |

### What Tier 2 confirmed vs added

- **Same three-tier ranking** at full scale: voyage > octen > lateon ≫ gecko > bm25 > medcpt. Tier 1's conclusions hold.
- **Voyage clearly best now**. At Tier 1, voyage / octen / lateon all tied at HR(≥3) = 0.990. At n=3,185 the difference is real and visible across every metric.
- **Octen vs lateon now separable**: HR(≥3) 0.991 vs 0.971 — small but consistent.
- **Strict numbers drift down for top retrievers** at scale. More queries reveal more "no strict-relevant chunk anywhere in the pool" cases — the depth-3 ceiling.
- **Weighted metrics are very stable** between T1 and T2 (within 0.02). The graded signal smooths per-query noise that the binary thresholds expose.

---

## Method notes

**Judge**: `Qwen/Qwen3.5-397B-A17B-FP8` via vLLM, temperature 0, soft thinking budget 10k tokens, hard cap 25k. Prompt hash `9d2abdfb76b030ea`. Validated against Claude Opus 4.7 on 62 reference (q, c) pairs: 95% threshold agreement at score ≥ 3, 85% at score ≥ 5 (see [`notes/rubric_design_worked_examples.md` § Tier 1 pilot validation](notes/rubric_design_worked_examples.md#tier-1-pilot-validation--qwen-judge-vs-opus-47-reference-labels-2026-05-17)).

**Rubric**: v2 graded — `score = d1 × (d2 + d3 + d4)` where
- **D1 (topic, boolean)**: does the chunk address the query's clinical question/procedure?
- **D2 (meaningful, 0/1/2)**: depth of clinical content
- **D3 (actionable, 0/1/2)**: specificity of guidance (doses, thresholds, steps)
- **D4 (density, 0/1/2)**: fraction of chunk content directly relevant to the query

**Pool**: per query, union of all 6 retrievers' top-3 chunks, deduped (~11.4 unique chunks per query at Tier 2 scale).

**Run**: 2 H100 shards × 32 workers, 1 preemption cycle on shard 0, ~13h wall-clock total. 0 errored rows out of 36,418.

---

## Source files

- Tier 2 results: `data/audit/results_v2_full.{md,json}` (auto-generated, gitignored)
- Tier 1 results: `data/audit/results_v2.{md,json}`
- Labels: `data/audit/v2_full_h100.jsonl` (merged shards, gitignored)
- Candidates: `data/audit/candidates_v2_full.jsonl`
- Per-retriever rankings: `data/full/{bm25,medcpt,octen,voyage,lateon,gecko}_top20.jsonl`
- Metrics regen: `python3 scripts/audit_metrics_v2.py --labels data/audit/v2_full_h100.jsonl --rankings-dir data/full --report data/audit/results_v2_full.md --raw data/audit/results_v2_full.json`

## Follow-ups

- [#17](https://github.com/nmrenyi/mamaretrieval/issues/17) — Gecko on-device retrieval gap: prioritized options to close the gecko vs voyage gap (threshold calibration, embedding bake-off, fine-tune, distillation).
