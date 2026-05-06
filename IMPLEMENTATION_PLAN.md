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
- Remove boilerplate chunks.
- Sample by tier quota with a fixed random seed.
- Write `data/sampled_chunks.jsonl`.

Checkpoint:

- Counts per source are sane.
- `icm-essential-competencies-for-midwifery-practice` is capped because it has
  only 15 chunks.
- Missing configured sources are logged as warnings.
- Output JSONL validates against the expected schema.

## Stage 4: Query Generation

Implement:

```bash
python scripts/generate_queries.py
```

Start with small, resumable runs:

```bash
python scripts/generate_queries.py --limit 5
python scripts/generate_queries.py --resume
```

This script should generate:

- `per_chunk` questions
- `synthesis` questions
- `adversarial` reformulations

Implementation decision to lock before full runs:

- Decide whether adversarial questions replace selected originals or are added
  as extra records. Prefer replacement or additive-with-cap so query count stays
  close to `queries.target_total`.

Checkpoint:

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
