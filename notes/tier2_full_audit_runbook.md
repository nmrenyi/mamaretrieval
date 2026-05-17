# Tier 2 full audit — runbook

**Goal**: produce per-retriever Variant D metrics (HR@3, P@3, plus weighted variants) on the full **3,185-query** set with the v2 graded rubric. Sharpens the Tier 1 (n=100) numbers and enables per-segment analysis.

**Status (2026-05-17)**: not started. Tier 1 pilot complete and validated (see `notes/rubric_design_worked_examples.md` Tier 1 pilot validation section and `data/audit/results_v2.md`).

**Related issues**: GitHub #15 (the Phase 4 full-run umbrella), #17 (gecko gap remediation — kicked off after this runbook completes).

---

## Inventory — what we have vs need

| Retriever | Status for 3,185 queries | Action |
|---|---|---|
| BM25 | Embedded in `data/candidates.jsonl` as `bm25_rank` | Extract top-3 — no new compute |
| MedCPT | Embedded as `medcpt_rank` | Extract top-3 — no new compute |
| Octen | Embedded as `octen_rank` | Extract top-3 — no new compute |
| voyage | Only 100 audit queries done (`data/audit/voyage_top20.jsonl`) | **Run on remaining 3,085** via voyage API |
| LateOn | Only 100 audit queries done | **Run on remaining 3,085** on cluster |
| gecko | Only 100 audit queries done | **Run on remaining 3,085** on cluster |

Query files:
- Full set: `data/queries.jsonl` (3,185 queries)
- Audit subset: `data/queries_audit.jsonl` / `data/audit/query_ids.txt` (100)
- Tier 2 must run on the 3,085 NOT in the audit subset (we already have the audit-100 rankings — preserve them and merge).

---

## Phase A — Get top-20 for all 6 retrievers on all 3,185 queries

Goal: produce six full-corpus top-20 files: `data/full/bm25_top20.jsonl`, `medcpt_top20.jsonl`, `octen_top20.jsonl`, `voyage_top20.jsonl`, `lateon_top20.jsonl`, `gecko_top20.jsonl`. Each with one row per query.

Schema (matches `data/audit/*_top20.jsonl`):
```json
{"query_id": "q_XXXXX", "retriever": "name", "top_k": 20, "results": [{"chunk_id": "...", "rank": 1, "score": ...}, ...]}
```

### A1. Extract BM25 / MedCPT / Octen from Phase 2b candidates

Write `scripts/extract_phase2b_rankings.py`:
- Reads `data/candidates.jsonl` (3,185 rows; each has `candidates[]` with `bm25_rank`, `medcpt_rank`, `octen_rank` fields, possibly null)
- For each retriever name in {bm25, medcpt, octen}:
  - For each query: filter chunks with non-null `<retriever>_rank`, sort by rank, keep top-20
  - Write to `data/full/<retriever>_top20.jsonl`
- Phase 2b only stored union top-20; some queries may have <20 entries per retriever. Document this in the comment.
- Effort: ~30 min including check.

### A2. Voyage on the remaining 3,085 queries

- Identify the existing voyage code path used for the audit 100 (look at how `data/audit/voyage_top20.jsonl` was produced — should be a script under `scripts/`).
- Use `voyage-4-large` (same model that produced the audit numbers — keep comparable).
- Embed the 3,085 query texts via voyage API, run cosine similarity against the pre-computed corpus index, write top-20 per query.
- Merge with existing audit-100 file → `data/full/voyage_top20.jsonl` (3,185 rows total).
- Cost estimate: voyage charges ~$0.12/M tokens for queries on voyage-4-large; 3,085 queries × ~30 tokens ≈ 92k tokens ≈ **~$0.01–0.05** in query embeddings. Corpus embedding was a one-time cost already paid.
- Effort: ~1-2 hours including the API call orchestration. Can run from laptop in parallel with the cluster jobs.

### A3. LateOn on remaining 3,085 queries

- Existing pattern: see `scripts/submit_*.sh` for the LateOn audit-100 job (the prior pipeline).
- Submit one job on the cluster with the 3,085-query subset as input.
- Keep `node-pool h100` for consistency.
- Wall-clock: ~1-3 hours including model load + preemption risk.

### A4. Gecko on remaining 3,085 queries

- Existing runbook: `notes/gecko_eval_runbook.md`.
- Submit via `scripts/submit_gecko_full.sh` (created 2026-05-17 — modeled on `submit_lateon_audit.sh`).
- Cluster job, single H100 (TFLite quant is CPU-bound but we ask for 1 GPU for scheduling).
- Default sqlite is `/lightscratch/users/yiren/mamai-medical-guidelines/processed/embeddings.sqlite` (63,650 chunks). The `model_backup/` sqlite only has 2,826 chunks — wrong corpus version; do not use.
- Wall-clock: ~1-2 hours, dominated by per-query TFLite embedding.

**Gotcha (2026-05-17)**: the gecko `sanity_check_pairs()` self-test reports `sanity: FAIL` on the cluster — the "unrelated PPH vs jaundice" pair scores 0.308 vs the 0.3 threshold. The failure is borderline (could be TFLite numeric drift across hardware) and the script continues. More importantly, the **margin** between same-topic paraphrases (0.53 low end) and unrelated (0.31) is only ~0.22, vs the 0.6+ a high-capacity model would show. This corroborates the Tier 1 finding that gecko is at its discrimination ceiling for medical chunks — see GitHub issue #17 comment (2026-05-17) for the full diagnosis and implications for the remediation roadmap.

### A5. Sanity-check

- Each `data/full/<retriever>_top20.jsonl` has 3,185 rows
- Each row has `results` with up to 20 chunks
- Spot-check 5 queries — confirm rankings are sensible

### Sequencing for Phase A

```
Parallel:
  ├── A2 voyage API     (laptop, ~1-2 hours)
  ├── A3 LateOn cluster (~1-3 hours wall)
  └── A4 Gecko cluster  (~1-3 hours wall)
Sequential after:
  └── A1 extraction (fast, ~30 min)
  └── A5 sanity check
```

---

## Phase B — Build the v2 candidates file

`scripts/build_v2_full_candidates.py`:
- Mirror `scripts/build_judge_input_audit_*.py` (or whatever pattern Tier 1 used to build `data/audit/candidates_v2_pilot.jsonl`)
- For each query in `data/queries.jsonl`:
  - Union of all 6 retrievers' top-3
  - Dedup chunks
  - Write `{query_id, query_text, candidates: [{chunk_id}, ...]}`
- Output: `data/audit/candidates_v2_full.jsonl`
- Expected size: ~36,600 unique (q, c) pairs (11.5 avg from Tier 1 × 3,185 queries; the average per query may grow with more diverse queries)
- Effort: ~30 min Python.

---

## Phase C — Judge run on the cluster

Use the existing `scripts/submit_judge_relevance.sh` + `scripts/run_judge_relevance_job.sh`. Same prompt hash, same caps as Tier 1 (validated 95% threshold agreement vs Opus-4.7 at ≥3).

Submission command:
```bash
JOB_PREFIX=mamaretrieval-judge-v2-full-h100 \
SHARD_COUNT=1 \
NODE_POOL=h100 \
RUBRIC=v2_graded \
INPUT_PATH=data/audit/candidates_v2_full.jsonl \
OUTPUT_DIR=data/audit \
OUTPUT_PREFIX=v2_full_h100 \
THINKING_BUDGET=10000 \
THINKING_TOKEN_BUDGET=25000 \
MAX_TOKENS=0 \
bash scripts/submit_judge_relevance.sh
```

### Sharding decision

Default: single shard (Tier 1 worked fine). If cluster is busy and you want to cut wall-clock:
- SHARD_COUNT=2 or 4
- Each pod judges its slice of queries
- More preemption surface area but parallel progress
- Recommend single shard unless explicit need for speed

### Supervisor

Reuse `scripts/supervise_pilot.sh` — change the JOB and OUT/RAW paths to `mamaretrieval-judge-v2-full-h100-shard0` and `data/audit/v2_full_h100_shard0.{jsonl,raw.jsonl}`. The auto-resubmit logic handles terminal failures; pod-level preemption is handled automatically by runai.

### Wall-clock estimate

- 36,600 pairs / 8 parallel workers = ~4,575 sequential calls
- ~5 sec/pair average → **~6.5 hours of judging**
- + 15-20 min model load per pod (per attempt)
- + preemption cycles (Tier 1 had 2 cycles, ~20 min each)
- **Realistic: 8-13 hours wall-clock**

### Resume safety

`RESUME_ARGS=(--resume)` is on by default (LIMIT=0). If a pod dies mid-run, the next pod picks up from existing output file. Don't delete `data/audit/v2_full_h100_shard0.jsonl` between attempts.

---

## Phase D — Compute metrics

Once the job succeeds:
```bash
rsync -av light:/mnt/light/scratch/users/yiren/mamaretrieval/data/audit/v2_full_h100_shard0.jsonl \
              light:/mnt/light/scratch/users/yiren/mamaretrieval/data/audit/v2_full_h100_shard0.raw.jsonl \
              data/audit/

python3 scripts/audit_metrics_v2.py \
  --labels data/audit/v2_full_h100_shard0.jsonl \
  --report data/audit/results_v2_full.md \
  --raw    data/audit/results_v2_full.json
```

Sanity checks before trusting the report:
- Line count matches expected (~36,600)
- `_error` field empty on every row (any non-empty means JSON parse failure — re-judge those)
- Score distribution is reasonable (not 100% zeros, has spread)

### Optional: bootstrap CIs

At n=3,185 you can compute 95% bootstrap CIs per retriever per metric. Not necessary for the headline but useful for "is voyage statistically better than octen?". Add to `audit_metrics_v2.py` if needed.

---

## Phase E — Document & commit

1. Append a "Tier 2 full audit (n=3,185)" subsection to `notes/rubric_design_worked_examples.md`:
   - New scoreboard
   - Any retriever ranking changes vs Tier 1 (probably none)
   - Notable per-query patterns if found
2. Commit with `worked-examples: add Tier 2 full audit scoreboard`
3. Push
4. Comment on GitHub #15 (Phase 4 full run) and close
5. Activate #17 (gecko gap remediation) — Phase A there is the threshold calibration

---

## Decisions to make before kicking off

1. **All 6 retrievers, or skip any?** Default: all 6 (~$0.05 for voyage + a few cluster hours).
2. **One shard or several?** Default: one shard (resume + supervisor handles preemption).
3. **Re-validate the judge?** Default: no (Tier 1 validated; same prompt hash `9d2abdfb76b030ea`).
4. **Voyage API key still valid?** Verify before kicking off A2.
5. **Cluster availability?** Check `runai list jobs` for current contention before submitting heavy jobs.

---

## Risks & contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| Voyage API rate-limited or down | low | Retry with backoff, then accept partial |
| Cluster contention causing many preemptions | medium (we saw this in Tier 1) | Supervisor handles; could shard if it gets bad |
| LateOn / Gecko cluster job fails | medium | Diagnose with `runai logs`; resubmit |
| Judge gives spurious `_error` rows | low (Tier 1 had 0) | Re-judge the errored pairs |
| Per-retriever ranking changes vs Tier 1 | very low | Document if it happens; would suggest n=100 was unrepresentative |
| Judge job preempted past 24h | medium | Resume mode keeps progress; just keep restarting until done |

---

## What "done" looks like

- `data/audit/v2_full_h100_shard0.jsonl` has ~36,600 records, 0 errors
- `data/audit/results_v2_full.md` exists, matches the script's output
- `notes/rubric_design_worked_examples.md` has the Tier 2 scoreboard appended
- Commits pushed
- GitHub #15 closed
- #17 ready to start

---

## Reference — Tier 1 pilot baseline (for comparison)

| Retriever | HR (≥3) | P (≥3) | HR (≥5) | P (≥5) | wHR | wP |
|---|---:|---:|---:|---:|---:|---:|
| voyage | 0.990 | 0.820 | 0.730 | 0.430 | 0.847 | 0.657 |
| octen  | 0.990 | 0.760 | 0.720 | 0.413 | 0.845 | 0.624 |
| lateon | 0.990 | 0.727 | 0.710 | 0.380 | 0.833 | 0.586 |
| gecko  | 0.840 | 0.493 | 0.490 | 0.210 | 0.693 | 0.404 |
| bm25   | 0.740 | 0.413 | 0.390 | 0.163 | 0.613 | 0.336 |
| medcpt | 0.610 | 0.287 | 0.310 | 0.117 | 0.523 | 0.259 |

n=100 queries, k=3. All metrics on the same denominator (queries with no relevant chunk in pool contribute HR=0 / P=0).
