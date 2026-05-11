# Audit Retrieval Runbook

How to run **BM25 + MedCPT + Octen top-20 retrieval** on the EPFL light cluster
for the Phase 3 completeness audit (30-query subset).

This runbook covers only the cluster retrieval step. The audit also requires
LLM judging, hand review, and metric comparison — see
`IMPLEMENTATION_GUIDE.md` §8 Phase 3 for the full sequence.

---

## What this is

Phase 2a ran top-10 retrieval over all 3,185 queries. For the **completeness
audit**, the same retrievers re-run at top-20 on a 30-query stratified subset.
The wider pool gives the LLM judge more chances to find every relevant chunk,
so we can measure how many true positives Phase 2a's pool missed.

- **Inputs:**
  - `data/audit/query_ids.txt` — 30 audit query IDs (one per line)
  - `data/queries_audit.jsonl` — those 30 queries' full records
- **Output:**
  - `data/candidates_audit.jsonl` — one record per audit query,
    ~30–60 unique candidates each (top-20 from each of 3 retrievers, unioned)

---

## Prerequisites

- `ssh light` works, with the `light-yiren` project configured for runai.
- The mamaretrieval repo lives on the cluster at
  `/mnt/light/scratch/users/yiren/mamaretrieval/`.
- Corpus embedding cache from Phase 2a exists at
  `/lightscratch/users/yiren/mamaretrieval/.cache/` (Octen + MedCPT).
  If it's missing, the job will recompute it (~15 min on H100).

---

## Step 1 — Select the 30-query audit subset

Stratified random sample per `IMPLEMENTATION_GUIDE.md` §8 Phase 3 Step 1:

| Tier | Count |
|------|------:|
| very_high | 6 |
| high | 4 |
| moderate_high | 2 |
| moderate | 4 |
| low_moderate | 2 |
| random fill (any tier) | 12 |
| **total** | **30** |

This selection script does not exist yet. Create `scripts/select_audit_queries.py`
to read `data/queries.jsonl` and write:

- `data/audit/query_ids.txt` — one query ID per line
- `data/queries_audit.jsonl` — the 30 full query records, same schema as
  `data/queries.jsonl`

Use a fixed `--seed` so the selection is reproducible.

---

## Step 2 — Patch the in-pod runner to accept a custom queries path

`scripts/run_pool_candidates_job.sh` currently hardcodes
`--queries data/queries.jsonl`. Add a `QUERIES_PATH` env var so the same
runner can be reused for the audit subset:

```diff
 TOP_K="${TOP_K:-10}"
+QUERIES_PATH="${QUERIES_PATH:-data/queries.jsonl}"
 BATCH_SIZE="${BATCH_SIZE:-64}"
```

```diff
 python3 -u scripts/pool_candidates.py \
-  --queries data/queries.jsonl \
+  --queries "$QUERIES_PATH" \
   --corpus "$CORPUS_PATH" \
   --output "data/candidates_shard${SHARD_INDEX}.jsonl" \
```

Also propagate the env var through the submit script — add one line
inside the `runai submit` env block in `scripts/submit_pool_candidates.sh`:

```diff
     -e TOP_K="$TOP_K" \
+    -e QUERIES_PATH="$QUERIES_PATH" \
     -e BATCH_SIZE="$BATCH_SIZE" \
```

These three lines are the entire infrastructure change; Phase 2a's defaults
are unaffected.

---

## Step 3 — Submit the audit job

For 30 queries, **one shard is enough.** The bottleneck is corpus encoding,
not query count, and with a warm embedding cache the job runs in minutes.

Check GPU pool availability first (prefer H200 > H100 > A100):

```bash
ssh light 'runai list nodes | grep -E "h200|h100"'
```

Submit:

```bash
cd ~/Downloads/mamaretrieval

SHARD_COUNT=1 \
JOB_PREFIX=mamaretrieval-pool-audit \
TOP_K=20 \
QUERIES_PATH=data/queries_audit.jsonl \
bash scripts/submit_pool_candidates.sh
```

To target H200 instead of H100, edit `--node-pool h100` → `--node-pool h200`
in `submit_pool_candidates.sh` (or parameterize it via an env var).

The submit script will:
1. rsync `scripts/`, `config.yaml`, and `data/queries_audit.jsonl` to the cluster.
2. Launch `mamaretrieval-pool-audit-shard0` on the chosen node pool.

---

## Step 4 — Monitor

```bash
ssh light 'runai list jobs --project light-yiren'
ssh light 'runai logs mamaretrieval-pool-audit-shard0 -f --project light-yiren'
```

Expected wall time:

| Cache state | Wall time |
|---|---|
| Embedding cache warm (Octen + MedCPT already on `.cache/`) | ~2–5 min |
| Cold cache (corpus has to be re-encoded) | ~15–20 min on H100, ~10 min on H200 |

---

## Step 5 — Sync and rename

With one shard, no merge is needed — just rename the shard output to its
audit-specific name:

```bash
rsync -av \
  light:/mnt/light/scratch/users/yiren/mamaretrieval/data/candidates_shard0.jsonl \
  data/candidates_audit.jsonl
```

**Sanity checks:**

```bash
wc -l data/candidates_audit.jsonl                  # should be 30
python3 -c "
import json
n = c = 0
for line in open('data/candidates_audit.jsonl'):
    r = json.loads(line); n += 1; c += len(r['candidates'])
print(f'{n} queries, {c} total candidates, avg {c/n:.1f}/query')
"
# expect: 30 queries, ~900–1800 total candidates, avg 30–60/query
```

---

## What this runbook does not cover

The Phase 3 spec calls for additional retrievers beyond BM25/MedCPT/Octen
to build a truly exhaustive pool. These are not on the cluster path and need
separate work:

- **voyage-4-large** — API-only (Voyage AI). Doesn't run on the cluster;
  call the API from a separate script and merge results into
  `candidates_audit.jsonl` post-hoc.
- **BGE-reranker** — cross-encoder. Not a retriever; applies as a
  precision-boosting re-rank after the union pool is built.
- **LateOn** (ColBERT-style late interaction) — requires the `pylate`
  library + a PLAID index. Significant integration work; deferred.

After the audit's candidate pool is built, the downstream steps are:

1. Run the Phase 2b LLM judge on the audit pool (same prompt, different input).
2. Hand-review all LLM-relevant labels (score ≥ 1) plus a 20% random sample
   of LLM-irrelevant labels (score = 0).
3. Compute Hit Rate / MRR / nDCG@10 / Recall@5 / Precision@5 using
   (a) exhaustive labels and (b) pipeline labels from `data/relevance_labels.jsonl`.
4. Report the gap. Acceptance: < 2–3 pp on Hit Rate and MRR.

See `IMPLEMENTATION_GUIDE.md` §8 Phase 3 for the full spec.
