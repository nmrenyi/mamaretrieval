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

---

## Current Status

**Phase 1c complete** — 3,185 per-chunk queries assembled. Synthesis and adversarial queries tracked in issue #5.

| Phase | Status |
|-------|--------|
| 1a — Corpus sampling | Done |
| 1b — LLM filtering + query generation | Done |
| 1c — Assemble final query records | Done |
| 2a — Retrieval candidate pooling | Next |
| 2b — LLM relevance judging | Pending |
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
| RRF (BM25 + MedCPT + Octen) | Hybrid fusion | Combined recall |

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

---

## Corpus Contract

The expected corpus is the `rag-bundle-v0.2.0` guideline bundle:

- `63,650` chunks
- `87` sources
- Headers formatted as `<sep>[SOURCE:<source>|PAGE:<page>|CID:<chunk_id>]`

The guideline repository (`~/Downloads/mamai-medical-guidelines/`) is
read-only for this project.
