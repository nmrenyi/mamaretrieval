# mamaretrieval Implementation Plan

This plan breaks the implementation into checkpoints so each stage can be
validated before moving on to API calls, dense indexing, or release packaging.

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
