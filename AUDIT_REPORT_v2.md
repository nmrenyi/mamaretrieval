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

---

## Tier 3 — Full top-20 union judging (n=3,185 queries, 230,964 pairs)

Tier 3 expanded the judged pool from each retriever's top-3 (Tier 2) to each retriever's **top-20** chunks. Result: **230,964 (q, c) pairs judged** (Tier 2's 36,418 + 194,546 new pairs at ranks 4-20 of the union). Same Qwen3.5-397B-A17B-FP8 judge, same v2 graded rubric, same thinking budgets.

The Tier 3 labels enable two analyses that Tier 2 couldn't support:
1. **HR / Precision at deeper k** (k=5, 10, 20) — does going deeper help the weaker retrievers?
2. **Pool Recall** — of all relevant chunks discovered by the union of 6 retrievers, what fraction does each retriever's top-k capture? Diagnostic for "MISS vs RANK".

### Run notes

- Both shards used H100, 32 workers each.
- Total wall-clock: **~5 days** (vs ~13h for Tier 2's 36k pairs at the same throughput). The bulk of the time was lost to cluster preemption — our 2-shard request (16 GPUs) was 1 GPU over the project's 15-GPU deserved quota, making the over-quota shard preemptible. Once shard 0 finished, the single remaining shard (8 GPUs) was within quota and ran preempt-free for ~10 hours straight until completion.
- **Lesson**: future multi-shard judge runs should size tensor-parallel so total GPU usage ≤ 15 (the project's deserved quota), e.g. 1 large shard with 8 GPUs instead of 2 over-quota shards. Single-shard at 8 GPUs averages ~0.6 records/sec — for 200k pairs that's ~4 days of pure judging, still painful but at least preempt-free.
- Judge: same Qwen3.5-397B-A17B-FP8 model, v2_graded prompt hash 9d2abdfb76b030ea, soft thinking budget 10k, hard cap 25k.
- 0 errored rows out of 230,964.

### HR / Precision at varying k — Binary lenient (score ≥ 3)

| Retriever | HR@3 | HR@5 | HR@10 | HR@20 | P@3 | P@5 | P@10 | P@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **voyage** | **0.996** | **0.999** | **1.000** | **1.000** | **0.867** | **0.816** | **0.725** | **0.614** |
| octen      | 0.991 | 0.996 | 0.999 | 0.999 | 0.804 | 0.747 | 0.651 | 0.542 |
| lateon     | 0.971 | 0.985 | 0.995 | 0.998 | 0.738 | 0.662 | 0.555 | 0.451 |
| gecko      | 0.814 | 0.895 | 0.950 | 0.977 | 0.477 | 0.435 | 0.370 | 0.305 |
| bm25       | 0.754 | 0.833 | 0.907 | 0.952 | 0.417 | 0.367 | 0.299 | 0.239 |
| medcpt     | 0.644 | 0.745 | 0.859 | 0.926 | 0.334 | 0.302 | 0.258 | 0.213 |

### HR / Precision at varying k — Binary strict (score ≥ 5)

| Retriever | HR@3 | HR@5 | HR@10 | HR@20 | P@3 | P@5 | P@10 | P@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **voyage** | **0.753** | **0.819** | **0.883** | **0.918** | **0.452** | **0.402** | **0.329** | **0.252** |
| octen      | 0.716 | 0.790 | 0.855 | 0.901 | 0.403 | 0.353 | 0.282 | 0.215 |
| lateon     | 0.664 | 0.734 | 0.814 | 0.872 | 0.350 | 0.294 | 0.230 | 0.170 |
| gecko      | 0.439 | 0.547 | 0.663 | 0.755 | 0.193 | 0.169 | 0.136 | 0.105 |
| bm25       | 0.371 | 0.457 | 0.570 | 0.666 | 0.163 | 0.139 | 0.108 | 0.080 |
| medcpt     | 0.272 | 0.347 | 0.464 | 0.586 | 0.112 | 0.099 | 0.082 | 0.065 |

### Pool Recall — lenient (score ≥ 3)

Of the chunks judged relevant in the 6-retriever top-20 union pool, what fraction does THIS retriever's top-k capture?

| Retriever | Pool R@3 | Pool R@5 | Pool R@10 | Pool R@20 |
|---|---:|---:|---:|---:|
| **voyage** | **0.176** | **0.258** | **0.421** | **0.658** |
| octen      | 0.164 | 0.236 | 0.378 | 0.583 |
| lateon     | 0.151 | 0.210 | 0.324 | 0.489 |
| gecko      | 0.101 | 0.143 | 0.221 | 0.337 |
| bm25       | 0.088 | 0.122 | 0.183 | 0.271 |
| medcpt     | 0.071 | 0.101 | 0.159 | 0.244 |

### Pool Recall — strict (score ≥ 5)

Same computation restricted to strict-relevant chunks (complete / specific actionable content: doses, thresholds, management steps), over the 2,989 queries with ≥1 strict-relevant chunk in the pool (avg 7.5 strict-relevant chunks per query, vs 20.4 lenient).

| Retriever | Pool R@3 | Pool R@5 | Pool R@10 | Pool R@20 |
|---|---:|---:|---:|---:|
| **voyage** | **0.267** | **0.368** | **0.544** | **0.768** |
| octen      | 0.240 | 0.327 | 0.471 | 0.661 |
| lateon     | 0.212 | 0.277 | 0.395 | 0.540 |
| gecko      | 0.120 | 0.169 | 0.242 | 0.345 |
| bm25       | 0.110 | 0.145 | 0.207 | 0.283 |
| medcpt     | 0.078 | 0.106 | 0.162 | 0.234 |

The strict view *widens* the deployment gap: at k=20 gecko captures 34.5% of strict-relevant chunks vs voyage's 76.8% (2.2×, vs 2.0× lenient). Notably, gecko's strict pool recall (0.345) is essentially equal to its lenient one (0.337) — moving the bar from "any useful content" to "complete, actionable content" doesn't change what gecko finds, while every top-tier retriever improves (voyage 0.658 → 0.768). At strict, gecko sits closer to bm25 (0.283) than to the dense top tier. The chunks the generator most needs are exactly the ones the deployed retriever most fails to surface — at any rank in its top-20.

### Key findings from Tier 3

1. **HR@3 numbers are identical to Tier 2** (same top-3 chunks judged in both tiers). The new value is at k=5, 10, 20 and in Pool Recall.

2. **Gecko's gap is mostly a ranking problem (at lenient), but also a real retrieval gap (at strict)**:
   - At lenient (≥3): gecko HR@3 = 0.81 → HR@20 = 0.98 — the chunks ARE in gecko's pool, just not in its top-3. Depth-based interventions (issue #17 option 1: threshold calibration, or retrieving deeper) could substantially close this gap.
   - At strict (≥5): gecko HR@3 = 0.44 → HR@20 = 0.76 — even at k=20, gecko misses ~24% of queries where voyage finds strict content at k=3. This is a real retrieval gap, not just ranking.

3. **Pool Recall sharply discriminates the two tiers**: voyage captures 66% of pool relevance at k=20; gecko only 34%. Gecko misses about half of the discoverable relevance entirely — never surfaces it in its top-20. The strict (≥5) view is worse: gecko stays at 0.345 while voyage rises to 0.768 — gecko's deficit is concentrated precisely on the complete/actionable chunks.

4. **The voyage/octen/lateon cluster** stays clearly above the gecko/bm25/medcpt cluster at every depth. The depth-3-only conclusion holds; deeper-k just shrinks gaps on the easier (lenient) metric.

### Implications for issue #17 (gecko remediation)

- **Threshold calibration (option 1)** can address the ~15% lenient HR gap (gecko@3 → gecko@10 worth of relevance) by raising the bar on what gecko returns and falling back to "I don't know" when low confidence. Doesn't fix the structural ~33% pool-recall gap.
- **Embedding bake-off (option 2)** or **distillation from voyage (option 5)** remain the most promising paths to actually close the pool-recall gap. The Tier 3 numbers strengthen the case: this isn't only ranking — gecko's embeddings genuinely fail to find relevant content for many queries.

### Source files

- Tier 3 results: `data/audit/results_v2_tier3.{md,json}`
- Full top-20 labels (Tier 2 + Tier 3): `data/audit/v2_top20_all.jsonl` (230,964 pairs)
- New-only Tier 3 labels: `data/audit/v2_top20_new.jsonl` (194,546 pairs)
- Raw thinking traces (Tier 3 only): `data/audit/v2_top20_new_h100_shard{0,1}.raw.jsonl`
- Metrics regen: `python3 scripts/audit_metrics_tier3.py`
