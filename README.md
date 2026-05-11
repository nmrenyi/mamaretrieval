# mamaretrieval

`mamaretrieval` builds a retrieval benchmark for the MAMAI medical RAG system.
The final artifact is a set of clinical queries paired with relevant guideline
chunk IDs, used to evaluate retrievers and to provide oracle context for
generator faithfulness checks.

The benchmark is built from the production guideline corpus at:

```text
~/Downloads/mamai-medical-guidelines/processed/chunks_for_rag.txt
```

See `IMPLEMENTATION_GUIDE.md` for the full benchmark specification and
`IMPLEMENTATION_PLAN.md` for staged implementation checkpoints.

---

## Pipeline

```bash
python scripts/sample_chunks.py          # Phase 1a: sample chunks by source tier
python scripts/llm_filter_chunks.py      # Phase 1b: LLM filter + seed query generation
python scripts/generate_queries.py       # Phase 1c: assemble final query records
python scripts/pool_candidates.py        # Phase 2a: run retrievers, union top-k
python scripts/judge_relevance.py        # Phase 2b: LLM judge → relevance labels
python scripts/audit.py                  # Phase 3: completeness audit
```

All pipeline outputs are written under `data/` and are gitignored.

### Data artefacts

One authoritative file per phase. Full schemas in `IMPLEMENTATION_GUIDE.md` §5.

| File | Phase | Produced by | What it is |
|------|-------|-------------|------------|
| `data/sampled_chunks.jsonl` | 1a | `sample_chunks.py` | Tier-weighted sample of corpus chunks |
| `data/llm_filter_results.jsonl` | 1b | `llm_filter_chunks.py` | Raw filter judgments (kept and rejected) |
| `data/llm_filtered_chunks.jsonl` | 1b | `llm_filter_chunks.py` | Chunks that passed the clinical-relevance gate + seed query |
| `data/queries.jsonl` | 1c | `generate_queries.py` | Final query records (one per `query_id`) |
| `data/candidates.jsonl` | 2a | `pool_candidates.py` | One record per query — RRF-fused pool of retriever candidates |
| `data/relevance_labels.jsonl` | 2b | `judge_relevance.py` | One record per `(query_id, chunk_id)` pair with D1/D2/D3 + score. **The benchmark's primary artefact.** Schema also in §Phase 2b below. |

Other files under `data/`:

- `queries_review.txt` — human-readable per-source query listing for spot-checking Phase 1 outputs (not consumed by downstream phases).
- `qwen36_27b_fp8_shuffle50_seed42_thinking_<timestamp>/` — captured reasoning traces from the Phase 1b filter run, for audit / debugging only.

---

## Current Status

**Phase 2b complete** — 78,571 (query, chunk) relevance labels generated and verified. Next: Phase 3 completeness audit.

| Phase | Status |
|-------|--------|
| 1a — Corpus sampling | Done |
| 1b — LLM filtering + query generation | Done |
| 1c — Assemble final query records | Done |
| 2a — Retrieval candidate pooling | Done |
| 2b — LLM relevance judging | Done |
| 3 — Audit | Pending |

---

## Phase 1a — Corpus Sampling

`scripts/sample_chunks.py` samples chunks from the 63,650-chunk corpus,
weighted by source quality tier defined in `config.yaml`.

**Tier system:** sources are rated `very_high` / `high` / `moderate_high` /
`moderate` / `low_moderate` based on clinical depth and relevance to the
Zanzibar OBGYN/midwifery context (ratings from
`~/Downloads/mamai-medical-guidelines/source-research/rag-source-evaluation.md`).
Higher-tier sources receive proportionally larger chunk budgets.

**Outcome:** 4,540 chunks sampled across 19 sources (target was 5,000; some
sources were exhausted before their budget was reached).

---

## Phase 1b — LLM Filtering and Seed Query Generation

`scripts/llm_filter_chunks.py` sends each sampled chunk to an LLM and asks it
to (a) judge whether the chunk contains clinically answerable content, and (b)
if so, generate a specific clinical question answerable from the chunk alone.

This is a single-pass gate: chunks that fail the clinical relevance judgment
are discarded; the generated question becomes the `seed_query` for kept chunks.

### Model

**Qwen3.6-27B-FP8** via vLLM with `--reasoning-parser qwen3`. The thinking
trace is used for judgment quality but only the final structured output
(verdict + query + reason) is retained.

Smaller models (e.g. Qwen3 9B) were tested first and produced acceptable
results, but Qwen3.6 27B showed stronger clinical discrimination — correctly
rejecting administrative, narrative, and rubric-style chunks while generating
more specific, answerable questions from dense clinical content.

### Key design decisions

**Prompt hash covers both system prompt and user message template.**
`PROMPT_HASH` is computed as SHA256(system_prompt + "\x00" +
rendered_user_message_for_a_sentinel_chunk). This means any change to either
the prompt text or the input formatting (e.g. adding Source/Page fields)
automatically invalidates `--resume` caches. A manual version string would
silently go stale.

**Output schema is enforced at runtime.** `FilterResult` is a TypedDict that
defines all expected output fields. `RESULT_SCHEMA_VERSION` is derived
automatically from `FilterResult.__annotations__` — if a field is added or
removed, the version hash changes. `_validate_result()` enforces exact field
match at every return site.

**User message includes Source, Page, and Breadcrumb.** Giving the model
document provenance (not just the raw chunk text) improves judgment quality on
ambiguous chunks — it can assess whether the content is from a clinical
reference vs. an assessment rubric without relying solely on the text.

**Shuffle before sharding.** `--shuffle` randomizes chunk order before
processing. Without it, `--limit N` or a partial run only samples from the
first source alphabetically, producing misleading test results.

### Multi-GPU execution

Each GPU runs an independent vLLM instance + one shard of the input:

```bash
bash scripts/submit_llm_filter.sh   # submits 5 Run:ai jobs on EPFL light cluster
```

Each job: 1× H100, 8 CPU cores, 96 GB RAM, `--shard INDEX 5`. Outputs to
separate `data/llm_filtered_chunks_shard{N}.jsonl` files, merged afterward:

```bash
cat data/llm_filtered_chunks_shard{0..4}.jsonl > data/llm_filtered_chunks.jsonl
cat data/llm_filter_results_shard{0..4}.jsonl  > data/llm_filter_results.jsonl
```

Wall time: ~38 minutes for 4,540 chunks across 5 GPUs (8 async workers/GPU,
~0.5 chunks/s/GPU).

### Results

| Metric | Value |
|--------|-------|
| Chunks judged | 4,540 |
| Queries kept | 3,185 (70.2%) |
| Discarded | 1,355 (29.8%) |
| Errors | 0 |
| Prompt hash variants | 1 (all consistent) |
| Exact duplicate queries | 13 (legitimate topic overlaps) |

**Per-source keep rates** revealed a clear signal: high-tier clinical references
(MSF, Oxford Handbook, Hesperian) kept 90–97% of chunks, while assessment and
marking rubric materials (NMC marking criteria, NMC mock OSCE, WHO childbirth
course) kept <10%. The model correctly identifies that rubric text does not
yield answerable clinical questions. These three low-rate sources may be
dropped from future sampling runs.

---

## Phase 2a — Retriever Selection Rationale

The goal of Phase 2a is to build a **high-recall candidate pool** for each query. Every relevant chunk that the retrievers miss at this stage is silently lost — the LLM judge in Phase 2b never sees it. Maximising recall here is therefore more important than precision.

### Evidence base

We evaluated retriever choices against two benchmarks:

- **BEIR** (Thakur et al. 2021, NeurIPS) — 18 heterogeneous zero-shot retrieval datasets. Key finding from the paper: BM25 is a surprisingly strong zero-shot baseline; most dense models from 2021 (DPR −47.7%, ANCE −7.4%) underperform it. Re-ranking (BM25+CE, +11%) and late interaction (ColBERT, +2.5%) are the only approaches that consistently beat BM25.
- **RTEB English (beta)** — a newer retrieval benchmark covering legal, finance, code, and healthcare domains. Top open-source performers: Octen-Embedding-8B (81.17), Qwen3-Embedding-8B (73.88).

### Why each retriever was chosen

**BM25** — the zero-shot lexical baseline. Indispensable for exact-match recall: drug names, specific doses, procedure names, rare clinical terms. BEIR shows it outperforms most neural models in zero-shot settings. Fast and requires no GPU.

**MedCPT** — a bi-encoder trained on 255M biomedical query–article pairs from PubMed (Jin et al., NIH/NLM). Not on general leaderboards (domain-specific), but the best domain fit for a midwifery/obstetric clinical guidelines corpus. Captures biomedical vocabulary, abbreviations, and clinical framing that general-purpose models may miss.

**Octen-Embedding-8B** — the strongest open-source dense retriever on RTEB English (81.17). General-purpose, deployable on the cluster. Covers semantic matching and paraphrases that BM25 and MedCPT may miss.

**RRF (Reciprocal Rank Fusion)** — a parameter-free mathematical combination of the ranked lists from BM25, MedCPT, and Octen. Adds no new candidates but surfaces chunks that appeared in multiple retrievers' top-k near the top of the fused list, improving overall recall at no indexing cost.

### Why certain retrievers were deferred to Phase 3

**voyage-4-large** (78.14 RTEB) — top-ranked overall but API-only (Voyage AI). Not deployable locally. Reserved for the Phase 3 exhaustive audit where 30 queries are labelled with maximum-recall pools.

**BGE-reranker / BM25+CE** — re-rankers improve precision, not recall. They are the right tool for Phase 3 where we want the most accurate labels on the 30-query gold subset, not for Phase 2a where recall is the priority.

**LateOn** (lightonai, ColBERT-style, 57.22 BEIR) — late-interaction model with architectural diversity (MaxSim vs. cosine). Scored below Octen-Embedding-8B on the shared benchmark dimension we could compare. Requires a separate library (`pylate`/PLAID index) adding implementation complexity. Retained as a Phase 3 option for architectural diversity in the exhaustive pool.

### Phase 2a retriever set

| Retriever | Type | Purpose |
|-----------|------|---------|
| BM25 | Sparse lexical | Exact term recall |
| MedCPT | Medical dense bi-encoder | Biomedical domain recall |
| Octen-Embedding-8B | General dense bi-encoder | Semantic recall |

Candidate pool = union of the three top-10 lists (max 30 per query; actual avg ~25 due to inter-retriever overlap).
RRF is applied after pooling to rank candidates for Phase 2b judging — it reorders the pool, it does not add new candidates.

### Phase 3 audit additions

voyage-4-large (API, best overall), BGE-reranker (cross-encoder re-ranking), LateOn (late-interaction diversity), top-20 per retriever instead of top-10.

Full candidate evaluation with scores and reasoning for all considered retrievers: https://github.com/nmrenyi/mamaretrieval/issues/6

### Phase 2a — Execution

`scripts/pool_candidates.py` parses the 63,650-chunk corpus, builds BM25 and
dense indexes (embeddings cached to `.cache/`), retrieves top-10 per retriever
for each query, applies RRF, and force-includes the seed chunk in each pool.

```bash
bash scripts/submit_pool_candidates.sh   # submits 5 Run:ai jobs on EPFL light cluster
```

Each job: 1× H100, 8 CPU cores, 96 GB RAM, `--shard INDEX 5`. Corpus embeddings
(MedCPT and Octen) are shared across shards via a common `.cache/` directory.
Outputs to `data/candidates_shard{N}.jsonl`, merged afterward:

```bash
cat data/candidates_shard{0..4}.jsonl > data/candidates.jsonl
```

**Seed chunk policy:** the seed chunk for each query is force-included in the
candidate pool if no retriever returned it. It is **not** pre-labeled as
relevant — the LLM judge in Phase 2b evaluates it along with all other
candidates. A seed chunk may not fully answer a general query.

### Results

| Metric | Value |
|--------|-------|
| Queries processed | 3,185 |
| Total candidates | 78,571 |
| Avg candidates/query | 24.7 (max 30, actual max 31) |
| Seed found by ≥1 retriever | 3,015 (94.7%) |
| Seed force-included (missed by all) | 170 (5.3%) |

Wall time: ~20 minutes across 5 H100 shards (corpus encoding dominated by
Octen-Embedding-8B; MedCPT embeddings were cached from a prior run).

---

## Phase 2b — LLM Relevance Judging

`scripts/judge_relevance.py` calls an LLM to label each of the 78,571
(query, chunk) candidate pairs on three binary dimensions.

### Relevance scoring design

Each pair is assessed independently on:

| Dimension | Question | YES means… |
|-----------|----------|------------|
| **D1 — Topic** | Same clinical problem? | Same condition / event / drug / procedure as the query |
| **D2 — Meaningful** | Contains clinical info? | Background, risk factors, definitions, diagnostic criteria, management principles |
| **D3 — Actionable** | Contains specific guidance? | Dosing, protocols, thresholds, management steps, specific recommendations |

**Score formula** (computed in Python post-processing, not by the model):

```
score = D1 * (D2 + D3)   →   range 0–2
```

| D1 | D2 | D3 | score | Interpretation |
|----|----|-----|-------|---------------|
| No | — | — | 0 | Off-topic |
| Yes | No | No | 0 | On-topic but no clinical content |
| Yes | Yes | No | 1 | On-topic, background only |
| Yes | Yes | Yes | 2 | On-topic, actionable guidance |

D3=True implies D2=True (actionable ⟹ meaningful); this constraint is
enforced in post-processing. The model answers all three dimensions
independently — no conditioning in the prompt.

#### Research backing

- **TREC-CDS 3-level scale** (Not Relevant / Possibly Relevant / Definitely
  Relevant) motivates a three-level ordinal scale for clinical IR rather than
  binary.
- **DeCE** (decomposed binary evaluation) shows that asking dimensions
  separately and combining in post-processing achieves inter-annotator
  correlation r = 0.78 vs. r = 0.35 for holistic grading.
- **Saracevic relevance hierarchy** (topical → cognitive → situational)
  maps to D1 (topical match), D2 (clinical depth), D3 (situational
  actionability for a midwife/doctor).
- **UMBRELA** (TREC 2024): binary relevance with a clear topical gate is
  more reproducible than nuanced multi-level scales.
- **HealthBench** (Arora et al. 2025): the step-by-step CoT structure
  (reasoning before verdict) and binary per-criterion assessment were
  adopted for our prompt design.
- **G-Eval principle**: placing `reasoning` first in the JSON schema
  forces the model to reason before committing to labels; vLLM's
  `guided_json` enforces the field generation order.

### Implementation highlights

- **guided_json**: vLLM extension that enforces the output schema at
  inference time. Field order in `"required"` drives token generation —
  `reasoning` is emitted before the boolean verdicts.
- **PROMPT_HASH**: SHA256(system_prompt + sentinel_user_content)[:16].
  Any change to either the prompt text or the user message template
  automatically invalidates `--resume` caches.
- **RESULT_SCHEMA_VERSION**: derived from `JudgeResult.__annotations__`.
  Changes automatically when a field is added or removed.
- **Resume**: done pairs identified by `query_id::chunk_id`. Error records
  (score = −1) are re-processed on the next run. A post-run dedup step
  rewrites the output to contain only successful labels, deduplicated.
- **Sharding by query**: all candidates of a query go to the same shard,
  so per-query statistics are never split across shards.
- **No negative safety gate**: the corpus is curated clinical guidelines
  (MSF, WHO, Oxford Handbook). Source quality is the safety layer; the
  judge focuses on relevance, not content safety.

### Running Phase 2b

```bash
bash scripts/submit_judge_relevance.sh   # submit N parallel jobs (default 5, H100)
```

Each job: 8× GPUs (tensor-parallel), 16 CPU, 256 GB RAM, `--shard INDEX N`.
Model: **`Qwen/Qwen3.5-397B-A17B-FP8`** via vLLM (`--reasoning-parser qwen3`, `guided_json`).

The model occasionally emits two output artefacts that need stripping in
post-processing before JSON parsing:

- `<think>...</think>JSON` or orphaned `</think>JSON` — leaked thinking tokens
- ```` ```json\n{...}\n``` ```` — markdown code-fence wrapping

Both are handled in `judge_relevance.py` (`_parse_response`). If you change the
judge model and start seeing JSON parse errors, check the `raw_repr` in the
error record's `reasoning` field — a new leakage pattern may need handling.

Monitor:

```bash
ssh light 'runai list jobs --project light-yiren'
ssh light 'runai logs mamaretrieval-judge-shard0 -f --project light-yiren'
```

Merge after all shards complete:

```bash
rsync -av light:/mnt/light/scratch/users/yiren/mamaretrieval/data/relevance_labels_shard*.jsonl data/
cat data/relevance_labels_shard{0..4}.jsonl > data/relevance_labels.jsonl
```

### Output schema

`data/relevance_labels.jsonl` — one record per `(query_id, chunk_id)` pair:

```json
{
  "query_id": "q_01682",
  "chunk_id": "61cdabfce6cd8a6f",
  "reasoning": "The chunk directly addresses ...",
  "d1_topic": true,
  "d2_meaningful": true,
  "d3_actionable": false,
  "score": 1,
  "llm_model": "Qwen/Qwen3.5-397B-A17B-FP8",
  "llm_judge_schema_version": "v-f20c636b",
  "llm_judge_prompt_hash": "f36ff561215b3a6f",
  "llm_backend": "openai"
}
```

### Results

| Metric | Value |
|--------|-------|
| (query, chunk) pairs labeled | 78,571 / 78,571 (100%) |
| Unique queries | 3,185 |
| Errors | 0 |
| Duplicates | 0 |

Score distribution:

| Score | Count | % |
|-------|------:|------:|
| 0 (off-topic or no useful information) | 38,762 | 49.3% |
| 1 (on-topic, background only) | 15,023 | 19.1% |
| 2 (on-topic, actionable guidance) | 24,786 | 31.6% |

The run spanned multiple sharded submissions across A100/H100/H200 nodes on
the EPFL light cluster, with several mop-up rounds to label pairs that hit
output-format errors before the post-processing fixes were in place.

### Verification

`scripts/verify_relevance_labels.py` checks every record against three
invariants:

- `score == d1_topic × (d2_meaningful + d3_actionable)`
- `d1_topic == False → d2_meaningful == False, d3_actionable == False`
- `d3_actionable == True → d2_meaningful == True`

```bash
python scripts/verify_relevance_labels.py
```

Exits non-zero on any violation. Last run: **78,571 records checked, 0 violations.**

In addition, a 10-sample stratified manual spot check (3 each at scores 0/1/2,
plus one seed-chunk reference case) found all judgments defensible — including
nuanced cases such as drug-name collisions across different clinical contexts
and mislabeled section headers that the model correctly read past. The
upfront >85% agreement calibration against 50–100 human labels (originally
planned) was deferred; the Phase 3 completeness audit is the planned
mitigation.

---

## Phase 3 — Completeness Audit (next)

Pipeline labels are produced by TREC-style pooling over a finite retriever
set, so any relevant chunk the Phase 2a pool missed is silently lost. Phase 3
measures the size of that gap on a 30-query gold subset.

Steps (full spec in `IMPLEMENTATION_GUIDE.md` §8 Phase 3):

1. Stratified random sample 30 queries — 2 per tier (very_high: 6, high: 4,
   moderate_high: 2, moderate: 4, low_moderate: 2) plus random fill to 30.
2. For those 30 queries, run ≥6 retrievers at top-20 each (add voyage-4-large
   for API best-overall, BGE-reranker for cross-encoder precision, LateOn for
   architectural diversity beyond the Phase 2a BM25/MedCPT/Octen set). Union
   pools.
3. LLM-judge every candidate with the same Phase 2b prompt.
4. Hand-review all LLM-relevant labels plus a 20% random sample of
   LLM-not-relevant labels. Record final verdicts in
   `data/audit/labels_exhaustive.jsonl`.
5. For each of the 30 audit queries, compute Hit Rate@k, MRR, nDCG@10,
   Recall@5, Precision@5 (k ∈ {1, 3, 5, 10}) using (a) exhaustive labels and
   (b) pipeline labels from `data/relevance_labels.jsonl`. Report the gap.

**Acceptance:** gap < 2–3 pp on primary metrics (Hit Rate, MRR). Larger gap →
increase `pool_candidates.py` top-k or improve the judge prompt before
proceeding.

### Metric choice under incomplete labels

Lead with completeness-robust metrics; report the rest as secondary:

- **Robust:** Hit Rate@k, MRR
- **Moderately sensitive:** nDCG@k
- **Most sensitive:** Recall@k, Precision@k

Since pipeline labels are necessarily incomplete (pooled from a finite
retriever set), absolute Recall/Precision numbers carry an error bar. Relative
ordering across retrievers is more reliable than absolute values. Report the
audit gap alongside the main retrieval scores — it is the error bar on all
retrieval numbers.

### Versioning

The benchmark is tied to a specific corpus version. Corpus version and
chunking scheme are coupled in the corpus repository, so
`(corpus_version, queries, labels)` is the versioned artefact. If the corpus
is updated, chunk IDs change and the entire labeling pipeline must be re-run.

---

## Corpus Contract

The expected corpus is the `rag-bundle-v0.2.0` guideline bundle:

- `63,650` chunks
- `87` sources
- Headers formatted as `<sep>[SOURCE:<source>|PAGE:<page>|CID:<chunk_id>]`

The guideline repository (`~/Downloads/mamai-medical-guidelines/`) is
read-only for this project.
