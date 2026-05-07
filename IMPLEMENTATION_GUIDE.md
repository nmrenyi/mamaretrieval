# mamaretrieval — Implementation Guide

*This guide is written for an agent implementing the mamaretrieval benchmark from scratch. It contains all necessary context, design decisions, data schemas, and per-script implementation specs.*

---

## 1. Project context

**MAMAI** is a Gemma 4 E4B + RAG medical-advice chatbot deployed for nurses and midwives in Zanzibar, covering OBGYN, neonatal/infant care, and reproductive health. The RAG corpus is a collection of chunked clinical guidelines (WHO, Tanzania MOH, Zanzibar MOH, and supplementary midwifery references).

**mamaretrieval** is a retrieval benchmark — a set of `(query, relevant_chunk_ids)` pairs — used to:
1. Evaluate and compare retrievers (BM25, dense embeddings, hybrid, re-rankers)
2. Provide oracle contexts for generator faithfulness evaluation (feeding ground-truth chunks to Gemma to measure hallucination)

The evaluation methodology is bottom-up: retriever is validated first on mamaretrieval, then the generator is evaluated with oracle contexts from mamaretrieval, then the end-to-end system is evaluated on mamabench.

### Reference repositories

Read both of these before implementing:

- **`~/Downloads/mamai-mamabench-docs/`** — evaluation planning docs. Read `mamaretrieval.md` (benchmark spec), `mamai-quality-evaluation.md` (full evaluation protocol), and `mamai-quality-evaluation-minimal.md` (minimal version). These are the authoritative source of truth for methodology.
- **`~/Downloads/mamai-medical-guidelines/`** — the production RAG corpus. Do NOT modify any files in this repo (another agent owns it). Read `README.md` for pipeline architecture, `processed/chunks_for_rag.txt` for the chunk data, and `source-research/rag-source-evaluation.md` for source relevance ratings used to drive sampling.

---

## 2. Repository structure

```
mamaretrieval/
├── IMPLEMENTATION_GUIDE.md         # this file
├── README.md                       # short overview for humans
├── requirements.txt
├── config.yaml                     # source tiers, query counts, model/path settings
├── scripts/
│   ├── sample_chunks.py            # Phase 1a: sample chunks from corpus by tier
│   ├── llm_filter_chunks.py        # Phase 1b: LLM chunk filter and seed queries
│   ├── generate_queries.py         # Phase 1c: assemble final query records
│   ├── pool_candidates.py          # Phase 2a: run retrievers, union top-k per query
│   ├── judge_relevance.py          # Phase 2b: LLM judge candidates → final labels
│   └── audit.py                    # Phase 3: completeness audit
├── data/
│   ├── sampled_chunks.jsonl        # output of sample_chunks.py
│   ├── llm_filter_results.jsonl    # all LLM chunk filter judgments
│   ├── llm_filtered_chunks.jsonl   # chunks kept after LLM filtering
│   ├── queries.jsonl               # output of generate_queries.py
│   ├── candidates.jsonl            # output of pool_candidates.py
│   ├── labels.jsonl                # output of judge_relevance.py
│   └── audit/
│       ├── query_ids.txt           # 30 selected audit query IDs (one per line)
│       ├── labels_exhaustive.jsonl # exhaustive labels for the 30-query subset
│       └── results.md              # comparison report: pipeline vs exhaustive labels
└── releases/
    └── mamaretrieval-v1/
        ├── queries.jsonl           # copy of data/queries.jsonl
        ├── labels.jsonl            # copy of data/labels.jsonl
        └── manifest.json           # versioning metadata
```

---

## 3. Dependencies

```
# requirements.txt
openai>=1.0.0          # query generation + LLM judge (GPT-4o-mini / GPT-4o)
rank-bm25              # BM25 retriever
sentence-transformers  # dense embedding retriever
faiss-cpu              # vector index for dense retrieval
tqdm
pyyaml
```

The corpus embeddings in `~/Downloads/mamai-medical-guidelines/processed/embeddings.sqlite` use a Gecko TFLite model (Android-specific, VF32 format) — these are NOT usable for Python-side retrieval. Build your own FAISS index from the chunk texts using `sentence-transformers` for the retriever evaluation pipeline.

---

## 4. config.yaml specification

```yaml
corpus:
  chunks_path: ~/Downloads/mamai-medical-guidelines/processed/chunks_for_rag.txt

queries:
  target_total: 3000
  questions_per_chunk: 1        # one per-chunk question per sampled chunk
  synthesis_questions: true     # generate cross-chunk synthesis questions
  adversarial_fraction: 0.15    # fraction of questions with robustness reformulations

# Source tier assignments and query targets.
# Only sources listed in ~/Downloads/mamai-medical-guidelines/source-research/rag-source-evaluation.md
# with RAG recommendation "Include" or "Include (selective)" are sampled.
# All other sources in the corpus are excluded from sampling.
source_tiers:
  very_high:
    queries_per_source: 300
    sources:
      - hesperian-a-book-for-midwives
      - msf-essential-obstetric-and-newborn-care
      - midwifery-essentials-6
  high:
    queries_per_source: 165
    sources:
      - oxford-handbook-of-midwifery
      - skills-for-midwifery-practice
      - who-midwifery-education-modules-2
      - who-midwifery-education-modules-3
      - who-midwifery-education-modules-4
      - who-midwifery-education-modules-5
      - who-midwifery-education-modules-6
      - who-platform-malta-guide-for-midwifery-skills
  moderate_high:
    queries_per_source: 130
    sources:
      - midwifery-essentials-3
  moderate:
    queries_per_source: 100
    sources:
      - clinical-practice-guidelines-midwifery-womens-health
      - midwifery-essentials-2
      - midwifery-essentials-5
      - who-essential-childbirth-care-course
      - nmc-midwifery-marking-criteria
      - icm-essential-competencies-for-midwifery-practice  # only 15 chunks; capped at ~40 queries
  low_moderate:
    queries_per_source: 65
    sources:
      - nmc-midwifery-mock-osce

models:
  query_generation: gpt-4o-mini   # cheap, fast; sufficient for query generation
  llm_judge: gpt-4o               # stronger model for relevance judgments

retrieval:
  top_k: 10                       # per retriever, before union
  pool_retrievers:                 # retrievers used in Phase 2a pooling
    - bm25
    - dense                        # sentence-transformers model defined below
  dense_model: sentence-transformers/all-MiniLM-L6-v2  # placeholder; replace with medical-domain model if available

audit:
  n_queries: 30
  top_k_exhaustive: 20            # larger pool for exhaustive labeling
  n_retrievers_exhaustive: 6      # minimum retrievers for audit pool
```

The `queries_per_source` values are targets. For sources with fewer chunks than required (e.g. `icm-essential-competencies-for-midwifery-practice` has only 15 chunks), cap at `n_chunks × 3` and note the shortfall.

---

## 5. Data schemas

### `data/sampled_chunks.jsonl`
One JSON object per line. Output of `sample_chunks.py`.

```json
{
  "chunk_id": "dcaeb591065c7c22",
  "source": "msf-essential-obstetric-and-newborn-care",
  "tier": "very_high",
  "section": "Postpartum Haemorrhage",
  "page": 14,
  "breadcrumb": "Postpartum Haemorrhage > Active Management of Third Stage",
  "text": "...(full chunk text, without the [SOURCE:|PAGE:|CID:] prefix)..."
}
```

### `data/llm_filter_results.jsonl`
One JSON object per line. Output of `llm_filter_chunks.py`, including both kept
and rejected sampled chunks.

```json
{
  "chunk_id": "dcaeb591065c7c22",
  "llm_filter_schema_version": "answerable-clinically-useful-v1",
  "llm_filter_prompt_hash": "<prompt hash>",
  "query": "What dose of oxytocin is used for active management of third stage?",
  "reason": "Chunk gives the oxytocin dose and the question guides bedside care.",
  "answerable_by_chunk": true,
  "clinically_useful": true
}
```

### `data/llm_filtered_chunks.jsonl`
One JSON object per line. Output of `llm_filter_chunks.py`. This contains only
sampled chunks whose generated seed query is both answerable by the chunk and
clinically useful.

```json
{
  "chunk_id": "dcaeb591065c7c22",
  "source": "msf-essential-obstetric-and-newborn-care",
  "tier": "very_high",
  "section": "Postpartum Haemorrhage",
  "page": 14,
  "breadcrumb": "Postpartum Haemorrhage > Active Management of Third Stage",
  "text": "...",
  "llm_filter_schema_version": "answerable-clinically-useful-v1",
  "llm_filter_prompt_hash": "<prompt hash>",
  "seed_query": "What dose of oxytocin is used for active management of third stage?",
  "llm_answerable_by_chunk": true,
  "llm_clinically_useful": true,
  "llm_filter_reason": "Chunk gives the oxytocin dose and the question guides bedside care."
}
```

### `data/queries.jsonl`
One JSON object per line. Output of `generate_queries.py`, built from
`data/llm_filtered_chunks.jsonl`.

```json
{
  "query_id": "q_0001",
  "query_text": "What dose of oxytocin do I give for active management of third stage?",
  "seed_chunk_id": "dcaeb591065c7c22",
  "source": "msf-essential-obstetric-and-newborn-care",
  "tier": "very_high",
  "query_type": "per_chunk",
  "adversarial_type": null
}
```

`query_type` is one of:
- `per_chunk` — standard question answerable by the seed chunk
- `synthesis` — question spanning multiple chunks on the same clinical topic
- `adversarial` — additive robustness-oriented reformulation of a per_chunk
  question

`adversarial_type` is null for non-adversarial queries. For adversarial queries,
it identifies the stress scenario:

- `abbreviation`
- `typo`
- `lay_synonym`
- `redundant_context`
- `ambiguous`
- `multi_condition`
- `negation`
- `rare_exact`

### `data/candidates.jsonl`
One JSON object per query. Output of `pool_candidates.py`.

```json
{
  "query_id": "q_0001",
  "query_text": "What dose of oxytocin do I give for active management of third stage?",
  "seed_chunk_id": "dcaeb591065c7c22",
  "candidates": [
    {"chunk_id": "dcaeb591065c7c22", "source": "...", "text": "...", "retrievers": ["bm25", "dense"], "scores": {"bm25": 12.3, "dense": 0.87}},
    {"chunk_id": "a1b2c3d4e5f60001", "source": "...", "text": "...", "retrievers": ["dense"], "scores": {"dense": 0.81}}
  ]
}
```

`candidates` is the deduped union of top-k results across all retrievers. Each candidate records which retrievers surfaced it and their raw scores.

### `data/labels.jsonl`
One JSON object per query. Output of `judge_relevance.py`.

```json
{
  "query_id": "q_0001",
  "query_text": "What dose of oxytocin do I give for active management of third stage?",
  "relevant_chunk_ids": ["dcaeb591065c7c22", "a1b2c3d4e5f60001"],
  "partial_chunk_ids": ["ff00112233445566"],
  "adjudication": {
    "dcaeb591065c7c22": "fully",
    "a1b2c3d4e5f60001": "fully",
    "ff00112233445566": "partially"
  }
}
```

`relevant_chunk_ids` = seed positive + all chunks labeled `fully` or `partially` relevant.
`partial_chunk_ids` = subset labeled `partially` (informational; included in relevant_chunk_ids).
`adjudication` = raw judge verdict per candidate for traceability.

### `releases/mamaretrieval-v1/manifest.json`

```json
{
  "version": "v1",
  "corpus_version": "v0.2.0",
  "date": "2026-05-06",
  "query_count": 2990,
  "label_count": 2990,
  "sources_sampled": 19,
  "chunks_path": "processed/chunks_for_rag.txt",
  "notes": "LLM-generated queries only; no hand-written queries."
}
```

---

## 6. Sampling strategy and priority

### Why sample at all

The corpus has 63,650 chunks across 87 sources. Generating queries for every chunk and labeling relevance at that scale would cost hundreds of dollars in API calls and weeks of labeling effort. The target of ~3,000 queries is chosen to be large enough to give reliable retriever rankings while remaining tractable.

### Priority: relevance tier drives query budget

Sources rated higher in `~/Downloads/mamai-medical-guidelines/source-research/rag-source-evaluation.md` receive a larger query budget. The tier weights and resulting query-per-source targets are:

| Tier | Weight | Queries/source | Rationale |
|------|--------|---------------|-----------|
| Very High | 4× | 300 | Best setting fit; most clinically actionable for Zanzibar nurses |
| High | 2.5× | 165 | Strong clinical content, LMIC-appropriate |
| Moderate-High | 2× | 130 | Good clinical chapters; some narrative filler |
| Moderate | 1.5× | 100 | Useful selectively; US/UK-centric or educator-framing |
| Low-moderate | 1× | 65 | Limited clinical content; included for completeness |

The weight ratios (e.g. Very High gets 4.6× more queries than Low-moderate) reflect how much more the evaluation should stress-test retrieval from high-value clinical sources. A retriever that misses a PPH dosing chunk from `msf` is a worse failure than missing an NMC OSCE rubric.

### Within-source sampling: stratified by breadcrumb section

Within each source, do not sample purely at random. Group chunks by their top-level breadcrumb section (e.g. "Postpartum Haemorrhage", "Neonatal Resuscitation"). Sample proportionally across sections so that no single chapter dominates. If a source has 10 sections and you need 150 chunks, sample ~15 from each section, adjusting for section size.

Concretely in `sample_chunks.py`:
1. Parse all chunks for the source.
2. Group by first breadcrumb level (the first `>` segment, or `__root__` if no breadcrumb).
3. Compute section weights proportional to section chunk count.
4. Sample `n_chunks` using weighted random sampling without replacement across sections.
5. Set `random.seed(42)` for reproducibility.

This ensures the query set covers the full clinical scope of each source rather than over-sampling one large chapter.

### Sources excluded from sampling

All sources in the corpus that do NOT appear in `rag-source-evaluation.md`'s included list receive zero queries. This includes the core WHO/NICE/Tanzania clinical guideline PDFs in `raw/Clinical guidelines_International/` and `raw/Clinical guidelines_Zanzibar-Tanzania/`. These are excluded from query generation for now — they will be incorporated in a future mamaretrieval version once explicit relevance tiers are assigned to them.

Sources explicitly excluded by `rag-source-evaluation.md` (rated Exclude or Not relevant) also receive zero queries regardless of their chunk count.

---

## 7. Corpus format

The corpus lives at `~/Downloads/mamai-medical-guidelines/processed/chunks_for_rag.txt`. It is `<sep>`-delimited plaintext with 63,650 chunks total across 87 source PDFs.

Each chunk:
```
<sep>[SOURCE:msf-essential-obstetric-and-newborn-care|PAGE:14|CID:dcaeb591065c7c22]
> Postpartum Haemorrhage > Active Management of Third Stage

Oxytocin 10 IU IM is the uterotonic of choice...
```

Parsing logic:
- Lines starting with `<sep>` are chunk headers; parse `SOURCE`, `PAGE`, `CID` with regex: `r'<sep>\[SOURCE:([^|]+)\|PAGE:(\d+)\|CID:([a-f0-9]+)\]'`
- Everything after the header line up to the next `<sep>` is the chunk text
- The first line(s) of chunk text beginning with `>` are the breadcrumb (section hierarchy); strip these for the `breadcrumb` field and keep the rest as `text`

**Important:** only 19 of the 87 sources are sampled (see `config.yaml`). Skip all others during sampling.

---

## 8. Script implementations

### Phase 1a — `scripts/sample_chunks.py`

**Purpose:** Read the corpus, filter to included sources, sample chunks per source according to tier quotas.

**Input:** `config.yaml`, corpus at `chunks_path`

**Output:** `data/sampled_chunks.jsonl`

**Logic:**
1. Parse all chunks from `chunks_for_rag.txt` into a dict keyed by `source`.
2. For each source in `config.yaml` source_tiers, compute `n_chunks_to_sample = target_queries / questions_per_chunk`. Round up.
3. For sources with fewer chunks than needed (e.g. `icm-essential-competencies` has 15 chunks), sample all chunks and log the shortfall.
4. Filter out boilerplate chunks before sampling. Boilerplate heuristics:
   - Chunk text (excluding breadcrumb) is shorter than 100 characters
   - Chunk text matches patterns like "Suggested citation:", "Endorsed by:", "Table of Contents", "References", "Acknowledgements", "Foreword"
   - Chunk text is a web-resource/contact-link list rather than clinical content
   - Chunk text is a mostly empty form/template table
   - Chunk consists only of a heading line with no body content
5. Sample using stratified random sampling (shuffle filtered chunks, take first N). Set `random.seed(42)` for reproducibility.
6. Write each sampled chunk as one JSON line to `data/sampled_chunks.jsonl`.

**Edge cases:**
- If a source in config is not found in the corpus, log a warning and skip.
- Preserve breadcrumb in the output (useful for synthesis question generation grouping).

---

### Phase 1b — `scripts/llm_filter_chunks.py`

**Purpose:** Call a local LLM for each sampled chunk to generate one seed query
and reject chunks whose seed query is not both answerable by that chunk and
clinically useful.

**Input:** `data/sampled_chunks.jsonl`

**Outputs:** `data/llm_filter_results.jsonl`, `data/llm_filtered_chunks.jsonl`

**Prompt contract:**

The system prompt asks the model to do three things in order:

1. Carefully understand the chunk's clinical topic, guidance, purpose, and
   completeness.
2. Generate exactly one clinical question a practicing midwife or nurse would
   type into a clinical reference system. The prompt limits the question to
   `≤20` words.
3. Judge `answerable_by_chunk` and `clinically_useful` independently, with a
   reason explaining both judgments. The prompt limits the reason to `≤30`
   words.

Educational or explanatory style is not itself a rejection reason. Explanatory
clinical chunks can be clinically useful when they support counseling,
assessment, prevention, risk recognition, diagnosis, management, or referral.
Reject only non-clinical scaffolding, administration, bibliography, professional
conduct / organization advice unrelated to patient counseling or care, and very
sparse fragments.

The model must return exactly one JSON object using one of these patterns:

```json
{"query": "<question ≤20 words>", "reason": "<≤30 words>", "answerable_by_chunk": true, "clinically_useful": true}
{"query": "<question ≤20 words>", "reason": "<≤30 words>", "answerable_by_chunk": true, "clinically_useful": false}
{"query": "<question ≤20 words>", "reason": "<≤30 words>", "answerable_by_chunk": false, "clinically_useful": true}
{"query": null, "reason": "<≤30 words>", "answerable_by_chunk": false, "clinically_useful": false}
```

Keep only chunks where both `answerable_by_chunk` and `clinically_useful` are
true. For kept chunks, copy the sampled chunk record and add `seed_query`,
`llm_answerable_by_chunk`, `llm_clinically_useful`, `llm_filter_reason`,
`llm_filter_schema_version`, and `llm_filter_prompt_hash`.

`--resume` must only reuse previous judgments with the current
`llm_filter_schema_version` and `llm_filter_prompt_hash`; older `suitable`
records, output records, or records from a previous prompt are stale and must
be ignored.

### Phase 1c — `scripts/generate_queries.py`

**Purpose:** Assemble final benchmark queries from LLM-filtered chunks, then add
synthesis and adversarial query records.

**Input:** `data/llm_filtered_chunks.jsonl`, `config.yaml`

**Output:** `data/queries.jsonl`

**Query types:**

#### Per-chunk questions (standard)
For each filtered chunk, use `seed_query` as the `per_chunk` question. Preserve
`chunk_id` as `seed_chunk_id`, plus the source and tier metadata.

#### Synthesis questions
After creating per-chunk records, group filtered chunks by clinical topic (use
the top-level breadcrumb heading). For each topic group with 3+ chunks, generate
1 synthesis question that cannot be answered by any single chunk alone.

User prompt:
```
Given these passages on the topic "{topic}", generate 1 question that
requires information from MULTIPLE passages to answer fully.

Requirements:
- The question must NOT be fully answerable by any single passage.
- Write as a nurse or midwife in Zanzibar would ask.
- The answer should require combining information across passages.

Passages:
{passages}

Return a JSON object: {"query_text": "...", "seed_chunk_ids": ["id1", "id2", ...]}
For seed_chunk_ids, list the chunk IDs whose combination is needed.
```

For synthesis questions, set `seed_chunk_id` to the first of the `seed_chunk_ids` in the output schema (primary seed), and `query_type` to `synthesis`.

#### Adversarial reformulations
After generating per-chunk questions, select up to `adversarial_fraction` (default 15%) for robustness-oriented reformulation. Keep the original per-chunk question and add each reformulation as a separate query record; the final query count may exceed `queries.target_total`.

Research-backed retrieval stress scenarios to cover:

- `abbreviation`: common clinical shorthand. Examples of eligible terms include postpartum haemorrhage/hemorrhage (PPH), magnesium sulphate/sulfate (MgSO4), blood pressure (BP), intravenous (IV), intramuscular (IM), prevention of mother-to-child transmission (PMTCT), antenatal care (ANC), caesarean/cesarean section (CS), fetal heart rate (FHR), and last menstrual period (LMP). Do not force abbreviations into questions where clinicians would not normally use shorthand.
- `typo`: realistic spelling or keyboard mistakes that preserve the clinical intent.
- `lay_synonym`: colloquial patient-facing wording instead of professional medical terminology, e.g. "bleeding too much after birth" for postpartum haemorrhage.
- `redundant_context`: extra bedside narrative around the actual information need.
- `ambiguous`: underspecified wording that still points to the same likely clinical topic.
- `multi_condition`: multiple constraints in one query, such as condition plus risk factor, contraindication, or patient state.
- `negation`: "avoid", "do not", contraindication, or absence-of-symptom wording.
- `rare_exact`: drug names, doses, measurements, procedures, or rare salient terms where exact matching matters.

Skip the adversarial record if the model output changes the clinical intent, is unchanged, is less clear, or sounds clinically unnatural.

Abbreviation prompt:
```
Rephrase the following clinical question to use common abbreviations and
shorthand as a nurse would use them (e.g. PPH instead of postpartum haemorrhage,
MgSO4 instead of magnesium sulphate, PMTCT instead of prevention of
mother-to-child transmission, IM instead of intramuscular, IV instead of
intravenous, BP instead of blood pressure, etc.).

Original: {question}

Return only the rephrased question string.
```

Use similarly constrained prompts for the other `adversarial_type` values. Set
`query_type` to `adversarial`, set `adversarial_type`, and retain the original
`seed_chunk_id`.

Do not include corpus poisoning or prompt injection in Stage 4 query generation.
Those belong in a later security/robustness audit because this benchmark uses a
fixed curated guideline corpus.

Research basis:

- Query-level RAG robustness work reports degradation under minor query variations such as typos, redundancy, formality changes, and ambiguity: https://aclanthology.org/2025.gem-1.38/
- Query-variation robustness work shows modern retrieval transformers can remain sensitive to typos and paraphrases: https://aclanthology.org/2024.findings-emnlp.248/
- Multi-condition IR work reports degradation as query complexity and condition count increase: https://aclanthology.org/2025.findings-emnlp.726/
- Medical retrieval work highlights domain-specific vocabulary, abbreviations, informal terminology, and negation as important clinical retrieval issues: https://www.nist.gov/publications/trec-medical-records-track
- Dense retrieval work highlights weaknesses around rare entities and salient phrase matching: https://aclanthology.org/2022.findings-emnlp.19/
- Medical search work highlights layperson/professional vocabulary gaps: https://link.springer.com/article/10.1007/s10791-015-9258-y

**Implementation notes:**
- Use `llm_filter_chunks.py` with local Ollama/Qwen for seed query generation
  and filtering.
- Use the generation model from `config.yaml` for synthesis and adversarial
  query assembly if those steps require LLM calls.
- Batch requests where possible; use `tqdm` for progress.
- Write to `data/queries.jsonl` incrementally (flush after each source) so partial runs are recoverable.
- Assign `query_id` as `q_{index:05d}` (zero-padded, sequential across all queries).
- Log token usage and estimated cost at the end.

---

### Phase 2a — `scripts/pool_candidates.py`

**Purpose:** For each query, run all configured retrievers and union the top-k results into a candidate set.

**Input:** `data/queries.jsonl`, corpus chunks, `config.yaml`

**Output:** `data/candidates.jsonl`

**Logic:**
1. Load all corpus chunks from `chunks_for_rag.txt` into memory (text + chunk_id + source).
2. Build retriever indices:
   - **BM25:** use `rank_bm25.BM25Okapi` on tokenised chunk texts. Tokenise by whitespace + lowercase.
   - **Dense:** encode all chunks with `sentence-transformers` model from config. Build a FAISS `IndexFlatIP` (inner product, i.e. cosine after normalisation). This will take a few minutes for 63,650 chunks.
3. For each query:
   a. Run each retriever, retrieve top-k chunk IDs and scores.
   b. Union results across retrievers, dedup by chunk_id.
   c. Always include the seed chunk(s) in the candidate set (they are guaranteed positives; including them ensures the LLM judge sees them).
   d. Write one candidates record to `data/candidates.jsonl`.

**Notes:**
- Build indices once and reuse across all queries. Do not re-build per query.
- Write incrementally; use `tqdm`.
- For synthesis queries with multiple seed chunk IDs, include all seeds in the candidate set.

---

### Phase 2b — `scripts/judge_relevance.py`

**Purpose:** For each query, call an LLM to judge whether each candidate chunk is relevant. Produce final `relevant_chunk_ids` labels.

**Input:** `data/candidates.jsonl`, `config.yaml`

**Output:** `data/labels.jsonl`

**Calibration — do this first:**
Before running at scale, manually label 50–100 (query, chunk) pairs as `fully / partially / not relevant`. Run the LLM judge on the same pairs. Compute agreement rate. Proceed only if agreement > 85%. If not, revise the judge prompt.

**Judge prompt:**

System:
```
You are a clinical relevance assessor for a medical retrieval benchmark.
The benchmark is for a midwifery/nursing assistant in Zanzibar, Tanzania.
```

User:
```
Query: {query_text}

Passage (chunk_id: {chunk_id}):
{chunk_text}

Does this passage answer the query?
- "fully": the passage directly and completely answers the query.
- "partially": the passage is relevant and provides useful information but
  does not fully answer the query on its own.
- "not": the passage is not relevant to the query.

Reply with exactly one word: fully, partially, or not.
```

**Logic:**
1. For each query in `data/candidates.jsonl`:
   a. The seed chunk(s) are automatically labeled `fully` — do not call the LLM for them.
   b. Call the LLM judge on all other candidates.
   c. `relevant_chunk_ids` = seed chunk(s) + all chunks labeled `fully` or `partially`.
2. Write one labels record to `data/labels.jsonl`.

**Notes:**
- Use `gpt-4o` (stronger model) for judging — quality matters here.
- Write incrementally; skip already-judged queries on re-run (check if `query_id` already in output file).
- Log total API calls and cost.
- If LLM returns something other than `fully / partially / not`, log and default to `not`.

---

### Phase 3 — `scripts/audit.py`

**Purpose:** Build a 30-query gold-standard subset with exhaustive labels. Compare against pipeline labels to validate label quality. Report the gap as the error bar on all retrieval scores.

**Input:** `data/queries.jsonl`, `data/labels.jsonl`, corpus chunks, `config.yaml`

**Output:** `data/audit/query_ids.txt`, `data/audit/labels_exhaustive.jsonl`, `data/audit/results.md`

**Logic:**

**Step 1 — Select 30 audit queries.**
Stratified random sample: 2 per tier (very_high: 6, high: 4, moderate_high: 2, moderate: 4, low_moderate: 2) plus random fill to 30. Write query IDs to `data/audit/query_ids.txt`.

**Step 2 — Build exhaustive candidate pool.**
For the 30 audit queries, run at minimum 6 retrievers (add alternative dense models, hybrid BM25+dense, a re-ranker if available) and retrieve top-20 from each. Union all results. This pool will be larger than Phase 2a's pool.

**Step 3 — Exhaustive LLM judging.**
Run the LLM judge on every candidate in the exhaustive pool (same prompt as Phase 2b). Then hand-review all LLM-relevant labels (fully or partially) plus a random sample of 20% of LLM-not-relevant labels. Record final human-adjudicated verdicts in `data/audit/labels_exhaustive.jsonl` (same schema as `data/labels.jsonl`).

**Step 4 — Compare and report.**
For each of the 30 audit queries, score a reference retriever using:
- (a) exhaustive labels
- (b) pipeline labels from `data/labels.jsonl`

Compute: Hit Rate@k, MRR, nDCG@10, Recall@5, Precision@5 for k in {1, 3, 5, 10}.

Report the per-metric gap (a) − (b). If gap < 2–3 pp on primary metrics (Hit Rate, MRR), pipeline labels are fit for purpose. If gap is larger, expand the pool size in `pool_candidates.py` or improve the judge prompt before proceeding.

Write findings to `data/audit/results.md` in this structure:
```
## Completeness Audit Results

| Metric | Exhaustive | Pipeline | Gap |
|--------|-----------|----------|-----|
| Hit Rate@5 | ... | ... | ... |
| MRR | ... | ... | ... |
| nDCG@10 | ... | ... | ... |

Verdict: [fit for purpose / expand pool]
```

---

## 9. Release packaging

Once all phases are complete and the audit passes:

1. Create `releases/mamaretrieval-v1/`.
2. Copy `data/queries.jsonl` → `releases/mamaretrieval-v1/queries.jsonl`.
3. Copy `data/labels.jsonl` → `releases/mamaretrieval-v1/labels.jsonl`.
4. Write `releases/mamaretrieval-v1/manifest.json` with:
   - `version`, `corpus_version` (read from mamai-medical-guidelines manifest at `releases/rag-bundle-v0.2.0/manifest.json`), `date`, `query_count`, `label_count`, `sources_sampled`, `notes`.

The `releases/mamaretrieval-v1/` directory is the artifact consumed by the evaluation pipeline.

---

## 10. Key design decisions (rationale)

**No hand-written queries.** Hand-written queries were considered but skipped for resource reasons. Three mitigations are applied in the generation prompt: synthesis questions (cross-chunk), nurse voice (realistic phrasing), and adversarial reformulations covering abbreviations, typos, lay terminology, redundant context, ambiguity, multi-condition queries, negation, and rare exact terms. See `~/Downloads/mamai-mamabench-docs/mamaretrieval.md` for full rationale. Residual caveat: dense-retriever scores may be optimistic; lead with MRR and Hit Rate rather than Recall@k.

**Sampling only from rag-source-evaluation.md included sources.** The corpus contains 87 source PDFs but only 19 are sampled. Sources not covered by `~/Downloads/mamai-medical-guidelines/source-research/rag-source-evaluation.md` (including the core WHO/NICE/Tanzania clinical guidelines) are excluded from query generation for now. This keeps the benchmark focused on sources with explicit relevance ratings.

**Seed chunks are always labeled fully relevant.** The seed chunk is the chunk the question was generated from — by construction it answers the query. This avoids wasting LLM judge calls on known positives, but it means if the generation prompt produced a bad question (not actually answered by the seed chunk), it silently enters the benchmark as a false positive. Mitigate by spot-checking a random sample of seed-chunk judgments.

**`icm-essential-competencies` capped at ~40 queries.** This source has only 15 chunks in the corpus. Maximum queries ≈ 15 × 3 = 45. Accepted shortfall; does not significantly affect the 3,000 query target.

**FAISS index built from scratch.** The corpus embeddings in `embeddings.sqlite` use a Gecko TFLite VF32 format specific to the Android RAG runtime — not usable in Python. Build a fresh FAISS index using `sentence-transformers` for Phase 2a. This means the dense retriever in the evaluation pipeline is not the same model as the production retriever; use a medical-domain model (e.g. `pritamdeka/S-PubMedBert-MS-MARCO`) if available for a fairer evaluation.

---

## 11. Running order

```bash
python scripts/sample_chunks.py      # → data/sampled_chunks.jsonl
python scripts/llm_filter_chunks.py  # → data/llm_filtered_chunks.jsonl
python scripts/generate_queries.py   # → data/queries.jsonl
python scripts/pool_candidates.py    # → data/candidates.jsonl  (CPU-heavy, ~30 min)
python scripts/judge_relevance.py    # → data/labels.jsonl  (~$20-40 GPT-4o)
python scripts/audit.py              # → data/audit/  (requires manual review step)
```

Estimated total cost: ~$30–60 in OpenAI API calls.
Estimated wall time: 2–4 hours (dominated by Phase 2a FAISS indexing and Phase 2b API calls).
