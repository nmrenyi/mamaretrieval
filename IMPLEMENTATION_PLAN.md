# mamaretrieval Implementation Plan

This plan breaks the implementation into checkpoints so each stage can be
validated before moving on to API calls, dense indexing, or release packaging.

---

## Methodology update (2026-05-16) — Stages 5-8 superseded by per-retriever evaluation

After Stages 1-4 produced the 3,185-query set and Stages 5-7 produced labels
on a candidate pool, we shifted the evaluation methodology. The original plan
treated mamaretrieval as a **static labeled benchmark**: a one-time, heavy
upfront investment to produce `queries.jsonl` + `labels.jsonl` that could be
reused to evaluate any future retriever cheaply. In practice we discovered
two things:

1. **At deployment depth k=3, recall is not the question** — precision is.
   Building a complete `relevant_chunk_ids` set per query (the labels' design
   purpose) only matters if recall metrics are load-bearing. They aren't:
   the deployed app only shows the LLM 3 chunks per query.
2. **Each retriever has its own "out-of-pool" chunks** that the pre-built
   labels don't cover. Phase 4's Gecko measurement showed that 26% of the
   deployed retriever's top-3 picks were unlabeled by the existing pool,
   forcing a per-retriever pool-expansion step anyway.

The new approach: skip building a static labeled benchmark. For each retriever
under evaluation, judge what *that retriever* actually returns at top-3 (or
top-15 for future flexibility) using a fine-grained rubric. Compute
precision-focused metrics (HR@3, Precision@3, MRR@3, NDCG@3 graded) directly.
This is documented as **Stage 9** below.

**What stays from the original plan**:
- Stages 1-4 — corpus parsing, chunk sampling, LLM filter, query assembly.
  The 3,185 queries in `data/queries.jsonl` are unchanged and reusable.
- Phase 2b labels (78,571 (q,c) pairs in `data/relevance_labels.jsonl`) are
  preserved as a **warm-start cache** — when a retriever surfaces a chunk
  Phase 2b already judged, we look up the label instead of re-judging.
- Phase 3 completeness audit (100 queries × 5 retrievers) is preserved as
  the precedent for the new approach and as the source of the 6-retriever
  Variant D comparison in `AUDIT_REPORT.md`.

**What is superseded**:
- Stage 5 (`pool_candidates.py`) — pool-based candidate generation is no longer
  the primary input to judging. Retrievers are run directly.
- Stage 6 (`judge_relevance.py` at pool scale) — the script is reused at the
  per-retriever scale, not at the union-pool scale.
- Stage 7 (`audit.py`) — the audit was the Phase 3 deliverable; the new approach
  IS the audit, scaled to all 3,185 queries.
- Stage 8 (release packaging of static `labels.jsonl`) — replaced by
  per-evaluation reports. `releases/mamaretrieval-v1/` may still package the
  queries + a snapshot of the most recent per-retriever evaluation results,
  but `labels.jsonl` is no longer the primary artifact.

The pre-shift stages 1-8 below are preserved as historical context; Stage 9
describes the new methodology. The IMPLEMENTATION_GUIDE.md has the matching
shift on the implementation side.

---

## Stage 1: Repository Skeleton

Create the basic project structure:

- `README.md`
- `requirements.txt`
- `config.yaml`
- `scripts/`
- shared package code, e.g. `mamaretrieval/`
- `data/.gitkeep`
- `releases/.gitkeep`
- `.gitignore` for generated JSONL files, caches, virtualenvs, FAISS artifacts,
  and other local build outputs

Checkpoint:

- The repo structure is clean.
- `config.yaml` loads successfully.
- No corpus processing happens yet.

## Stage 2: Corpus Parser

Implement shared parsing code for:

```text
~/Downloads/mamai-medical-guidelines/processed/chunks_for_rag.txt
```

The parser must handle chunks in this format:

```text
<sep>[SOURCE:...|PAGE:...|CID:...]
> breadcrumb
chunk text...
```

Internal chunk records should expose:

- `chunk_id`
- `source`
- `page`
- `breadcrumb`
- `text`

Checkpoint:

- Parser reads `63,650` chunks.
- Parser finds `87` sources.
- All 19 configured benchmark sources are present.
- Header parsing reports zero malformed `<sep>` records.

## Stage 3: Sampling

Implement:

```bash
python scripts/sample_chunks.py
```

This script should:

- Read `config.yaml`.
- Parse the full corpus.
- Filter to configured included sources only.
- Remove obvious boilerplate chunks before sampling.
- Compute each source's chunk target from its tier query budget:
  `ceil(queries_per_source / questions_per_chunk)`.
- Cap sources with fewer usable chunks than their target and log the shortfall.
- Group usable chunks within each source by top-level breadcrumb section.
- Use `__root__` as the section for chunks without a breadcrumb.
- Sample proportionally across those section groups without replacement.
- Use the configured fixed random seed for reproducibility.
- Write `data/sampled_chunks.jsonl`.

Boilerplate filtering should be conservative and deterministic. Drop chunks that
are clearly unsuitable for clinical query generation, including:

- body text shorter than 100 characters
- citation/front-matter chunks such as "Suggested citation", "Endorsed by",
  "Acknowledgements", "Foreword", "References", and "Table of Contents"
- copyright or license-only chunks
- web-resource/contact-link chunks
- mostly empty form/template tables
- heading-only chunks with no meaningful body text

Do not use an LLM for this filter. The first implementation should keep the
rules cheap, inspectable, and easy to tune after reviewing sample outputs.

Within-source stratification should use the first breadcrumb segment:

```text
Postpartum Haemorrhage > Active Management of Third Stage
```

maps to:

```text
Postpartum Haemorrhage
```

This grouping affects only which chunks are selected. Query generation still
uses individual sampled chunks directly.

Checkpoint:

- Counts per source are sane.
- `icm-essential-competencies-for-midwifery-practice` is capped because it has
  only 15 chunks.
- Missing configured sources are logged as warnings.
- The sampling report includes total, filtered, usable, target, sampled, and
  shortfall counts per source.
- Re-running with the same config and corpus produces identical output.
- Output JSONL validates against the expected schema.

## Stage 4: LLM Chunk Filtering and Query Assembly

First implement and run the LLM chunk filter:

```bash
python scripts/llm_filter_chunks.py
```

Start with small, resumable runs:

```bash
python scripts/llm_filter_chunks.py --limit 5
python scripts/llm_filter_chunks.py --resume
```

This script applies the current system prompt to each sampled chunk. For each
chunk, the model should:

- carefully understand the chunk's clinical topic, guidance, purpose, and
  completeness
- judge whether the chunk is clinically relevant before writing a query
- return `query=null` for non-clinical scaffolding, administration,
  bibliography, professional conduct / organization advice unrelated to patient
  counseling or care, and very sparse fragments
- generate exactly one grounded clinical seed query for clinically relevant
  chunks, with the prompt-level limit of `≤20` words
- write a reason explaining the clinical relevance decision, with the
  prompt-level limit of `≤30` words

Keep a chunk only when the returned `query` is non-null. Write all judgments to
`data/llm_filter_results.jsonl`, and write kept chunks to
`data/llm_filtered_chunks.jsonl` with `seed_query`, `llm_filter_reason`, the
result schema version, prompt hash, backend, and model. Resume logic must only
reuse judgments and kept output records that match the current schema, prompt
hash, backend, and model.

Then implement query assembly:

```bash
python scripts/generate_queries.py
```

`generate_queries.py` should consume `data/llm_filtered_chunks.jsonl`, not raw
`data/sampled_chunks.jsonl`. It should treat each `seed_query` as the
`per_chunk` query for that chunk, then add:

- `per_chunk` questions
- `synthesis` questions
- `adversarial` reformulations

Adversarial questions are additive. Keep the original `per_chunk` question and
write each robustness-oriented reformulation as a separate `adversarial` query
record with the same seed chunk. `queries.target_total` is a planning target for
source sampling, not a hard cap on the final query count.

Research on RAG and medical retrieval shows several query situations need
explicit retrieval evaluation coverage:

- `abbreviation`: common clinical shorthand such as PPH, MgSO4, BP, IV, IM,
  PMTCT, ANC, CS, FHR, or LMP. Only generate these when the original question
  contains natural abbreviation candidates; skip unchanged or unnatural
  reformulations.
- `typo`: realistic spelling or keyboard mistakes that preserve the clinical
  intent.
- `lay_synonym`: colloquial patient-facing wording instead of professional
  medical terminology.
- `redundant_context`: extra bedside narrative around the actual information
  need.
- `ambiguous`: underspecified wording that still points to the same likely
  clinical topic.
- `multi_condition`: questions with multiple constraints, such as condition
  plus risk factor, contraindication, or patient state.
- `negation`: "avoid", "do not", contraindication, or absence-of-symptom
  wording.
- `rare_exact`: drug names, doses, measurements, procedures, or rare salient
  terms where exact matching matters.

Do not include corpus poisoning or prompt injection in Stage 4 query generation;
those belong in a later security/robustness audit because this benchmark uses a
fixed curated guideline corpus.

Checkpoint:

- `data/llm_filter_results.jsonl` is valid JSONL using the current
  `query` / `reason` schema.
- `data/llm_filtered_chunks.jsonl` contains only chunks with non-null
  `seed_query` values after the LLM clinical relevance gate.
- `data/queries.jsonl` is valid JSONL.
- `query_id` values are stable and sequential.
- `seed_chunk_ids` are preserved for synthesis queries.
- A small dry run looks clinically plausible before spending on a full API run.

## Stage 5: Candidate Pooling

Implement:

```bash
python scripts/pool_candidates.py
```

Build this in two passes:

1. BM25 only.
2. Dense FAISS retrieval after BM25 works.

Notes:

- Build indices once and reuse across all queries.
- Always include seed chunk IDs in the candidate set.
- Deduplicate candidates by `chunk_id`.
- Dense model downloads may require network approval.

Checkpoint:

- `data/candidates.jsonl` contains one record per query.
- Candidate records include retriever names and raw scores.
- Seed chunks are always present in their query candidate pools.

## Stage 6: Relevance Judging

Implement:

```bash
python scripts/judge_relevance.py
```

This script should:

- Resume safely by skipping already judged query IDs.
- Auto-label seed chunks as `fully`.
- Call the LLM judge for non-seed candidates.
- Accept only `fully`, `partially`, or `not`.
- Default invalid judge output to `not` and log it.
- Write `data/labels.jsonl`.

Before the full run:

- Manually label 50-100 `(query, chunk)` pairs.
- Run the LLM judge on the same pairs.
- Proceed only if agreement is greater than 85%.

Checkpoint:

- Small subset judging completes.
- `relevant_chunk_ids` includes seeds plus `fully` and `partially` candidates.
- `partial_chunk_ids` is a subset of `relevant_chunk_ids`.

## Stage 7: Completeness Audit

Implement:

```bash
python scripts/audit.py
```

This script should:

- Select 30 stratified audit queries.
- Build larger exhaustive pools using at least 6 retrievers.
- Judge all exhaustive candidates.
- Produce files under `data/audit/`.
- Generate a report template for human review and metric comparison.

Checkpoint:

- `data/audit/query_ids.txt` exists.
- `data/audit/labels_exhaustive.jsonl` uses the same schema as labels.
- `data/audit/results.md` reports metric gaps between exhaustive labels and
  pipeline labels.

## Stage 8: Release Packaging

Implement release packaging after the pipeline output is trusted.

Expected release artifact:

```text
releases/mamaretrieval-v1/
  queries.jsonl
  labels.jsonl
  manifest.json
```

The manifest should record:

- benchmark version
- corpus version from `rag-bundle-v0.2.0`
- date
- query count
- label count
- sampled source count
- notes

Checkpoint:

- Release query and label counts match the source JSONL files.
- Manifest corpus version matches the guideline bundle manifest.
- Release directory is self-contained for downstream evaluation.

## Recommended Starting Scope

Start with Stages 1-3 only:

1. Build the repo skeleton.
2. Implement and test the corpus parser.
3. Implement and validate sampling.

This gets the local, deterministic foundation correct before spending API money
or building dense retrieval indices.

---

## Stage 9: Per-retriever evaluation (new methodology)

Replaces Stages 5-8 as the primary evaluation path. The retrievers being
evaluated drive the labelling — no pre-built static pool. Conceptually:

```
queries.jsonl (3,185) ──▶ run each retriever ──▶ judge top-K results
                              (top-3 default,        with fine-grained
                               up to top-15)          rubric (0-5)
                                                        │
                                                        ▼
                                                  per-retriever
                                                  Variant D table
                                                  (HR@3, P@3, MRR@3,
                                                   NDCG@3 graded)
```

### Step 9.1 — Design the fine-grained rubric

The original 3-level rubric (`D1 × (D2 + D3)` ∈ {0, 1, 2}) collapses too
much variation for the precision question. Design a 0-5 graded rubric
preserving the prompt's structural sensibilities:

- 5: directly and fully answers the query (specific dose / protocol for the
  exact scenario asked)
- 4: mostly answers; minor scope gap (e.g. dose for a related scenario)
- 3: partially answers; useful but incomplete
- 2: tangentially relevant; mentions the topic without useful detail
- 1: barely related; touches an adjacent topic
- 0: irrelevant

Include 5 worked examples in the prompt that anchor the scale (especially
the 2/3 and 3/4 boundaries). Register a new `llm_judge_prompt_hash` and
`llm_judge_schema_version` so the new labels are distinguishable from the
old Phase 2b/3 ones.

Checkpoint: 30-pair pilot run, inspect score distribution. If >90% are
clumped in 1-2 buckets, rubric needs tuning.

### Step 9.2 — Run retrievers on the full query set

For each retriever to evaluate, produce `data/per_retriever/<name>_top<k>.jsonl`:

- 3,185 queries × top-K (default 3, optional 15)
- Schema matches the existing per-retriever JSONLs (`{query_id, model,
  top_k, results: [{chunk_id, rank, score}]}`)
- For embedding retrievers, build the index once per retriever and reuse

Initial retriever set (deployment-relevant): Gecko (production), voyage-4-large
(strongest API), LateOn (best on-prem dense), Octen, BM25, MedCPT.

### Step 9.3 — Judge what each retriever returned

For each retriever's top-K JSONL, build a candidates input file containing
only (q, c) pairs **not already judged** in:
- `data/relevance_labels.jsonl` (Phase 2b warm-start cache), OR
- `data/audit/relevance_labels_audit.jsonl` (Phase 3 + Phase 4 expansions), OR
- prior Stage 9 judge runs with the same schema version

Submit one cluster judge job per retriever (or batch multiple retrievers in
one job — many will share chunks). Merge results into a per-retriever label
file `data/per_retriever_labels/<name>_<schema>_labels.jsonl`.

Cost estimate (rough): 3,185 q × 6 retrievers × top-3 ≈ 57,000 pairs;
~50% dedup against the warm-start cache and across retrievers leaves
~25,000-30,000 fresh judgments. ~1-2 hours of cluster judging.

### Step 9.4 — Compute metrics

Extend `scripts/audit_metrics.py` (which already handles Variant D) to:

- Operate on the 3,185-query set, not just 100 audit queries
- Use the Step-9.1 0-5 graded scores for NDCG
- Report Variant D per retriever, with bootstrap 95% CIs (since the sample
  is now 3,185 not 100, CIs will be tight)
- Output a top-level `data/per_retriever/results.md`

### Step 9.5 — Validation

Two cheap quality controls before declaring the per-retriever ranking final:

- **Judge calibration**: manually review 50 random fine-grained calls per
  rubric grade (5 per grade × 10 = 50). Computes a human-LLM agreement
  estimate. Replaces the deferred Stage 6 "85% agreement" check.
- **Bootstrap CIs**: included in Step 9.4 metrics output. Tells us which
  inter-retriever gaps are statistically significant.

### Step 9.6 — Release packaging (replaces Stage 8)

`releases/mamaretrieval-vN/`:

- `queries.jsonl` (unchanged from Stage 4)
- `per_retriever/` directory with one labelled top-K JSONL per evaluated retriever
- `results.md` snapshot
- `manifest.json` recording: bundle/corpus version, judge model + prompt hash
  + schema version, list of retrievers evaluated, date

Static `labels.jsonl` may still be included as a denormalised dump of all
Phase 2b + Phase 3 + Phase 4 + Stage 9 labels for warm-start by future
evaluations — but it is no longer claimed to be a complete `relevant_chunk_ids`
reference per query.

### Checkpoint for Stage 9

- One retriever (Gecko) evaluated end-to-end on the full 3,185-query set
- Variant D table reports HR@3 / P@3 / MRR@3 / NDCG@3 with CIs
- Judge calibration agreement ≥ 80% on the 50-pair sample
- Results.md is self-explanatory; reviewer who didn't run the pipeline can
  read it and understand what each number means
