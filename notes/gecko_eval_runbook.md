# Runbook: Gecko retrieval evaluation + label-granularity decision

> **Status (2026-05-16): #23 completed.** Headline result is in `AUDIT_REPORT.md` § "Phase 4 — Gecko deployment-quality measurement". Gecko sits in the middle tier of the 6-retriever comparison — clearly above BM25/MedCPT, clearly below Octen/voyage/LateOn. Lenient HR@3 = 0.880, Precision@3 = 0.657, NDCG@3 = 0.540. Pool was expanded with 79 new labels (`source: "gecko_pool_expansion"`), giving 100% top-3 coverage. **#24 remains open**, but the precision-style metrics turned out to be discriminating enough that finer labels are likely not needed — re-open if a follow-up retrieval-improvement workstream requires it.
>
> Below is the original plan, kept as historical record of the methodology.

Self-contained context-recovery document covering the two open workstreams in `mamaretrieval`:

- **#23**: Measure Gecko (the deployed retriever) on the 100 audit queries — coverage, HR@3, Precision@3, MRR@3, NDCG@3
- **#24**: Decide whether to invest in finer relevance labels (deferred — gated by #23 results)

Written to survive a context compaction. After compaction, read this from the top; it contains everything needed to resume.

---

## 0. Project context (one-screen orientation)

### What MAMAI is

On-device medical-advice Android app for nurses/midwives in Zanzibar. Codebase at `~/Downloads/mamai`. Uses:
- **Gemma 4 E4B** (3.66 GB `.litertlm`) for generation via LiteRT-LM 0.11.0
- **Gecko** (110M params, `Gecko_1024_quant.tflite` 146 MB) for embeddings
- **SQLite** for the vector index (`embeddings.sqlite`, ~89 MB, 63,650 chunks, 1024-dim Gecko embeddings)
- **Top-3 retrieval** by default

### What mamaretrieval is

Sibling benchmark repo at `~/Downloads/mamaretrieval`. Builds retrieval benchmarks for evaluating which retrievers do well on medical-midwifery queries. The Phase 3 audit ran 5 retrievers (BM25, MedCPT, Octen-8B, voyage-4-large, LateOn ColBERT) at top-20 over 100 sampled queries, then had a Qwen3.5-397B judge label every retrieved chunk for relevance on a 0/1/2 scale (`score = D1 × (D2 + D3)` — topic × (meaningful + actionable)).

**Key audit finding (already shipped in `AUDIT_REPORT.md` and `data/audit/results.md`)**: Phase 2a's deployment-targeted retrieval pool (3 retrievers × top-10) captures only ~49% lenient / ~52% strict of the truly-relevant chunks. The audit's Variant C decomposition showed that *depth (deeper top-k)* matters more than *breadth (more retrievers)* for closing the recall gap.

### The blind spot this runbook exists to close

**Gecko has never been evaluated against the audit labels.** All 5 retrievers in the audit are benchmark retrievers, not what the app actually uses on-device. We don't know:
- How recall@3 of Gecko compares to BM25 / Octen / voyage at the same depth
- Whether the deployment retrieval is meaningfully better or worse than the audit's middle retrievers
- Whether the audit's labels even *cover* what Gecko surfaces (pooling bias)

That's what #23 measures.

### Key recent decisions and findings

- **Production retrieval top-k is 3** (from `mamai/config/runtime_config.json`). The latency-bound deployment ceiling is k=3 on CPU (60 s budget) — on GPU there's no latency worry up to k=15 on the test device (Snapdragon 8 Elite).
- **Gemma 4 E2B is being considered** as a smaller/faster alternative to E4B. E2B at-rest CPU memory is 1.7 GB vs E4B's 3.3 GB. E2B's GPU TTFT smoke-tested at ~0.5 s; total at k=3 GPU ≈ 11 s (vs E4B's ~13.5 s). **Decode is bandwidth-bound, so E2B is only ~1.5× faster overall, not 2×.**
- **E2B latency sweep is in flight in a separate Claude CLI session**, executing the runbook at `mamai/evaluation/runbooks/e2b_sweep.md`. Branch `feat/e2b-latency-sweep` in mamai. Don't touch.
- **Relevance metric at k=3**: precision-style metrics (HR@3, Precision@3, MRR@3, NDCG@3) matter more than recall@3 because at small k recall is bounded by k/relevant. NDCG@3 already uses the 0/1/2 grading, so we get *some* relevance discrimination — but **if all 3 returned chunks are score=2, NDCG@3 = 1.0 regardless of order** (no discrimination among them). That limitation is what #24 is about.
- **Pooling bias is real**: any retriever evaluated against the audit's label set is at risk of "false negatives" where its top-3 includes chunks no benchmark retriever surfaced, so they have no label and default to "non-relevant." For Gecko specifically this could systematically underestimate quality. Mitigation: measure *coverage* first, expand the pool only if coverage is below ~80%.

### Key open question this session is *not* going to answer

Whether to switch the production model from E4B → E2B. That decision is being informed by the parallel E2B latency sweep + a separate answer-quality eval (in `mamai/evaluation/reports/eval_report_app_parity_v1.md`, ongoing). The Gecko measurement here is independent of that decision.

### File locations to remember

| What | Path |
|---|---|
| Gecko TFLite model | `~/Downloads/mamai/device_push/models/Gecko_1024_quant.tflite` (146 MB, 1024-dim, INT8 quant) |
| SentencePiece tokenizer | `~/Downloads/mamai/device_push/models/sentencepiece.model` (776 KB) |
| Production embeddings (chunks) | `~/Downloads/mamai/device_push/bundle/embeddings.sqlite` (89 MB, 63,650 chunks × 1024-dim) |
| Audit query IDs | `~/Downloads/mamaretrieval/data/audit/query_ids.txt` (100 lines) |
| Audit query text | `~/Downloads/mamaretrieval/data/queries_audit.jsonl` |
| Audit relevance labels | `~/Downloads/mamaretrieval/data/audit/relevance_labels_audit.jsonl` + Phase 2b labels at `~/Downloads/mamaretrieval/data/relevance_labels.jsonl` |
| Existing per-retriever top-20 JSONLs | `~/Downloads/mamaretrieval/data/audit/{bm25,medcpt,octen,voyage,lateon}_top20.jsonl` |
| Audit metrics script | `~/Downloads/mamaretrieval/scripts/audit_metrics.py` (the place to add Gecko's row) |
| RagPipeline.kt (how the app uses Gecko) | `~/Downloads/mamai/app/android/app/src/main/kotlin/com/example/app/RagPipeline.kt` — reference for embedding parameters (use_gpu_for_embeddings = false per app_config.json) |

---

## #23 — Gecko retrieval evaluation runbook

### Goal

Add a "gecko" row to the audit's per-retriever ranking table covering:
- **Coverage of Gecko's top-3 against the existing audit label pool** (gating measurement — tells us if we can trust the metrics without expanding the pool)
- **HR@3** (Hit Rate — at least 1 relevant chunk in top-3)
- **Precision@3** (fraction of top-3 that are relevant)
- **MRR@3** (mean reciprocal rank of first relevant chunk in top-3)
- **NDCG@3** (graded relevance, uses 0/1/2 scores directly)

Total effort: ~4 hours of laptop work, no device dependency, no cluster dependency *unless* coverage is too low.

### Step-by-step plan

#### Step 1 — Build the Gecko Python embedding harness (~2 hours)

Create `~/Downloads/mamaretrieval/scripts/retrieve_gecko_audit.py`. It should:

1. Load `Gecko_1024_quant.tflite` via TFLite Python (`tflite_runtime` or `tensorflow.lite`)
2. Load `sentencepiece.model` via `sentencepiece` library
3. Replicate exactly how `RagPipeline.kt` calls Gecko:
   - Read the input embedding spec from the tflite file's signature
   - Tokenize each query with sentencepiece
   - Pad/truncate to the input length (1024 tokens typically for this model)
   - Run inference, get the 1024-dim embedding
   - L2-normalize the output (the app does this — verify against `RagPipeline.kt`)

Read 100 query texts from `data/queries_audit.jsonl`. Embed all 100. Output: `data/audit/gecko_queries.npy` shaped `(100, 1024)`.

Sanity check: cosine similarity between the embeddings of two semantically similar queries (e.g. two PPH-related queries) should be > 0.5; between unrelated queries should be < 0.3.

#### Step 2 — Score Gecko against the production index (~1 hour)

Create `~/Downloads/mamaretrieval/scripts/score_gecko_audit.py`. It should:

1. Open `~/Downloads/mamai/device_push/bundle/embeddings.sqlite` read-only
2. Read all 63,650 chunk embeddings into a numpy array (1024-dim, L2-normalized — verify)
3. Compute cosine similarity between the 100 query embeddings and all chunk embeddings (one big GEMM: 100 × 1024 @ 1024 × 63650 = 100 × 63650 similarity matrix)
4. Take top-20 per query, write to `data/audit/gecko_top20.jsonl` in the schema matching the existing per-retriever files (look at `data/audit/octen_top20.jsonl` for format)

Schema (per record, one per query):
```json
{
  "query_id": "q_00001",
  "model": "gecko-1024-quant",
  "top_k": 20,
  "results": [
    {"chunk_id": "chunk_12345", "rank": 1, "score": 0.847},
    ...
  ]
}
```

#### Step 3 — Coverage check (~30 min)

Before any precision/recall computation, measure how much of Gecko's top-3 is in the existing audit label pool. This is the gating measurement.

Compute, per query and aggregated:
- `audit_labeled_set` = set of (query_id, chunk_id) pairs that appear in either `relevance_labels.jsonl` (Phase 2b) or `relevance_labels_audit.jsonl` (Phase 3 audit) restricted to the 100 audit queries
- For each query, Gecko's top-3 chunks
- `coverage_query = |{c in gecko_top3 if (qid, c) in audit_labeled_set}| / 3`
- `coverage_aggregate = mean(coverage_query)` across 100 queries

Report aggregate coverage. Three regimes:

| Coverage @ top-3 | Decision |
|---|---|
| **≥ 80%** | Trust the metrics with a footnote noting slight underestimate |
| **50–80%** | Expand the pool (Step 4) for clean numbers |
| **< 50%** | Must expand — current numbers uninterpretable |

#### Step 4 (conditional) — Expand the label pool

Only if coverage < 80%. Same recipe as the Phase 3 audit itself:

1. Write `scripts/build_judge_input_audit_gecko.py` — produces a per-query candidates JSONL of (query_id, chunk_id) pairs that are in Gecko's top-20 but NOT in the existing audit label set. Format matches `data/audit/candidates_audit_new_only.jsonl`.
2. Launch the Qwen3.5-397B judge on the EPFL cluster using existing scripts: `scripts/submit_judge_relevance.sh` with `INPUT_PATH`, `OUTPUT_DIR`, `OUTPUT_PREFIX` env vars. Use the same prompt hash `f36ff561215b3a6f` and schema version `v-f20c636b` from Phase 2b/3 for label consistency. Per-shard `PYTHONUSERBASE_${shard}` isolation.
3. Estimated cluster time: <1 hour if Gecko adds ~30% new chunks (~600–1500 new pairs).
4. Merge new labels into the audit set: append to `relevance_labels_audit.jsonl`. Add a comment in `AUDIT_REPORT.md` documenting the expansion.

#### Step 5 — Compute metrics (~1 hour)

Extend `~/Downloads/mamaretrieval/scripts/audit_metrics.py`:

1. Add `"gecko": "data/audit/gecko_top20.jsonl"` to `PER_RETRIEVER_INPUTS`.
2. Add a new function `compute_k3_deployment_metrics(retriever_data, relevance_dict)` that returns per-retriever `{HR_3, P_3, MRR_3, NDCG_3}` aggregates.
3. Add a new section to `data/audit/results.md` called "Variant D — k=3 deployment metrics" with the following table:

| Retriever | Coverage@3 | HR@3 | Precision@3 | MRR@3 | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| BM25 | 100% (in pool) | ? | ? | ? | ? |
| MedCPT | 100% | ? | ? | ? | ? |
| Octen | 100% | ? | ? | ? | ? |
| voyage-4-large | 100% | ? | ? | ? | ? |
| LateOn | 100% | ? | ? | ? | ? |
| **gecko (deployed)** | **?** | **?** | **?** | **?** | **?** |

Both thresholds (lenient ≥1 and strict =2) — produce one table per threshold.

Reuse code patterns from existing Variants A/B/C in the script.

#### Step 6 — Write up and commit (~30 min)

1. Update `AUDIT_REPORT.md` with a brief Section about Gecko results — one paragraph each on coverage, the headline metrics, and what we conclude (does deployment retrieval need attention or not).
2. Commits in mamaretrieval, focused per the existing style (`feat:`, `analysis:`, etc.). Don't push without user approval.

### What the result tells us

Three rough outcomes:

| Gecko's HR@3 / Precision@3 / NDCG@3 vs middle of audit retrievers | Implication |
|---|---|
| **Comparable** (e.g. close to Octen) | Deployment retrieval is OK. Focus optimization elsewhere — LLM, prompt engineering, query rewriting. |
| **Slightly below** | Modest gain possible from hybrid retrieval (BM25 + Gecko reciprocal-rank fusion) or a small on-device reranker. ~1–2 weeks of work. |
| **Well below** | Real deployment retrieval gap. Larger investment justified — possibly swapping Gecko, on-device reranking, or query rewriting. |

### Failure-mode guidance

| Symptom | Action |
|---|---|
| Gecko TFLite model fails to load via Python TFLite | Check tflite_runtime version; try with `tensorflow` package instead. The model is INT8-quantized, ensure your runtime supports that. |
| L2-normalization disagreement (cosine similarities all near 1.0 or all near 0) | Check whether the model output is already L2-normalized internally vs needs manual normalization. Reference: `RagPipeline.kt`'s embedder call. |
| `embeddings.sqlite` schema unknown | The `localagents.rag.memory.SqliteVectorStore` is the writer. Open with `sqlite3` CLI to inspect tables. Likely a single table with chunk_id + 1024 floats serialized. |
| Coverage drops below 50% — pool expansion stalls | The cluster judge needs MAMAI_HF_API_KEY env var; the Qwen3.5 model file needs to be HF-downloadable from the cluster nodes. Refer to PR `4345954` ("judge submit: parameterize OUTPUT_DIR/PREFIX/NODE_POOL, per-shard userbase") for env var setup. |

---

## #24 — Relevance-label granularity decision (deferred — gated by #23)

### The problem

Current relevance labels are 3-level (0/1/2 via `D1 × (D2 + D3)`). This handles "is this useful at all" well, but **can't discriminate among multiple score-2 chunks in top-3**. If a retriever returns 3 actionable chunks, NDCG@3 = 1.0 regardless of which is ranked first — we lose the signal about whether the retriever picked the *best* actionable chunk first.

For RAG specifically, even more important than ranking-within-relevant is **diversity**: 3 redundant actionable chunks waste context budget; 3 complementary chunks help the generator.

### Four options previously discussed

| Option | What | Effort | When useful |
|---|---|---|---|
| **(a) Finer-grained re-judging** (0–5 scale) | New Qwen3.5 prompt with richer rubric | ~1 hour cluster time on 6,400 labels | If we believe more grades discriminate top results |
| **(b) Aspect / sub-topic labels** | Tag each chunk with which sub-question it answers (e.g. "PPH dosing", "PPH initial assessment") | ~3 hours cluster time + meaningful prompt work | Enables diversity metrics (α-NDCG, sub-topic recall) |
| **(c) Pairwise preferences** | For each (query, top-3), judge ranks them against each other | Per-retriever, fast | Cleanest ranking signal, requires per-retriever judging |
| **(d) End-to-end RAG quality eval** | Generate answers with each retriever × k context, judge **answer** quality | 100 q × N retrievers × Gemma + Qwen judge | **Measures what we actually care about**, bypasses labeling problem entirely |

### Current position: defer until #23 lands

The decision tree:

1. Run #23 first.
2. If Gecko looks comparable to other audit retrievers on the precision-style metrics → no labeling problem to solve → close #24.
3. If Gecko's retrieval metrics look similar to others but downstream RAG answers are worse → proxies are failing → invest in **option (d)** (end-to-end eval).
4. If Gecko looks worse on metrics AND we want to discriminate among retrievers with similar HR@3 / Precision@3 → consider **option (a)** finer grading first as the cheapest improvement.
5. Skip (b) and (c) for now — significantly more work, less likely to be justified.

### What signal #23 will provide

- Gecko's HR@3 and NDCG@3 absolute values, vs the audit's middle retrievers
- Whether Gecko's top-3 contains many "all score-2" cases (= where #24's labeling issue actually bites) — count this as a side-output of Variant D
- Pool coverage — tells us about the *judging completeness*, which is different from but related to the label-granularity question

### When this decision needs to be made

Not yet. Re-open #24 after #23 results are in. Likely a 1-day analysis sprint to decide which (a)-(d) path to take.

---

## Post-compaction recovery checklist

If you're reading this after a context compaction:

1. ✅ Read this whole file
2. ✅ Check the current state of `mamaretrieval`:
   - `git status` — should be clean on `main`
   - `git log --oneline -5` — most recent commits should be the audit work (commits like `1a35b22`, `bd287a0`)
3. ✅ Check that the inputs in §0 "File locations to remember" all exist on disk
4. ✅ Check whether `data/audit/gecko_top20.jsonl` already exists (would indicate #23 partially done)
5. ✅ Read the existing audit files to refresh on what 5 retrievers and what metrics are there:
   - `AUDIT_REPORT.md`
   - `data/audit/README.md`
   - `data/audit/results.md`
6. ✅ Coordinate with E2B latency sweep: check `~/Downloads/mamai` is on branch `main` or `feat/e2b-latency-sweep` and whether new E2B JSONs have landed in `~/Downloads/mamai/evaluation/latency_results/` — those tell us about the parallel work in the other Claude CLI session.

Then proceed with Step 1 of the #23 plan.

---

_Last updated: 2026-05-15. Open issues: #23 (Gecko measurement, not started), #24 (label granularity decision, deferred). E2B latency sweep in flight in a separate CLI session — independent workstream._
