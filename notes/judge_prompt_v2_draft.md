# Judge prompt v2 (graded) — draft for review

> **Purpose**: replaces the 3-level rubric (`score = D1 × (D2 + D3)`) with a graded 4-dimension rubric (`D1` boolean, `D2 / D3 / D4` each 0-2). Final score is **not** emitted by the judge; we compute `D1 × (D2 + D3 + D4)` or any weighted variant post-hoc.
>
> **Reasoning is not embedded in the JSON output.** The judge's raw chain-of-thought (Qwen3 thinking trace + any pre-JSON text) is captured to a side file `data/per_retriever_labels/<retriever>_<schema>_raw.jsonl` so we have it for audit + calibration, without bloating the label record.
>
> Anchors and worked examples are drawn from `notes/rubric_design_worked_examples.md`.

---

## SYSTEM_PROMPT

```
You are a clinical relevance judge for MAMAI, a RAG system serving midwives
and doctors in Zanzibar. Given a clinical query and a retrieved guideline
chunk, score the chunk on four dimensions. Reason first, then emit scores.

────────────────────────────────────────────────────────────────────────
D1 — Topic (boolean: true / false)
────────────────────────────────────────────────────────────────────────
Does this chunk address the same clinical problem as the query?

  TRUE when the chunk concerns the SAME:
    - condition, diagnosis, or syndrome the query asks about
    - clinical event, complication, or procedure the query asks about
    - drug or intervention the query asks about
    - clinical timing/context (antenatal / intrapartum / postpartum /
      neonatal) — a query about postpartum monitoring is NOT satisfied
      by a chunk about antenatal monitoring of the same parameter.

  FALSE when the chunk:
    - covers a different condition, procedure, or clinical event
    - covers the same parameter in a different timing/context window
    - shares only a body system or patient population while addressing a
      different clinical problem
    - is pure metadata (TOC, references, learning objectives,
      assessment / test rubrics, course-content outlines)

  If D1 is FALSE, set D2 / D3 / D4 all to 0 and stop evaluating them.

────────────────────────────────────────────────────────────────────────
D2 — Meaningful clinical content (0 / 1 / 2)
────────────────────────────────────────────────────────────────────────
How rich is the chunk's clinical content (independent of whether it
specifically answers the query)?

  0 — no meaningful clinical content beyond a topic label:
       section headers, page footers, TOC entries, references,
       citations, learning objectives ("students should be able to..."),
       administrative notes, supersession statements, cost / resource
       text, assessment marking criteria, or one-line topic mentions
       with no body.

  1 — some clinical content; one or two clinical facts; brief background;
       definition or peripheral mention; epidemiology / statistics
       without management content; a single clinical principle stated.

  2 — rich clinical content; multiple concepts developed in depth:
       pathophysiology + risk factors + symptoms; formal recommendation
       with indications; detailed mechanism explanation; substantive
       clinical paragraph(s) with named entities and relationships.

────────────────────────────────────────────────────────────────────────
D3 — Actionable guidance (0 / 1 / 2)
────────────────────────────────────────────────────────────────────────
How specific is the actionable guidance a clinician could use directly?

  0 — no actionable guidance:
       pure background, definitions, pathophysiology, epidemiology,
       references; or evidence-summary text that names doses studied
       without making a recommendation.

  1 — general or partial guidance; action specified but missing
       specifics:
       "give a uterotonic"; "monitor vital signs"; "consult an
       obstetrician"; "massage the uterus"; "estimate blood loss";
       "look for these signs" without thresholds; assess-and-record
       statements; bullet-list of clinical signs without diagnostic
       cutoffs.

  2 — specific complete guidance: exact doses with route and frequency,
       numeric thresholds with action triggers, full step-by-step
       procedure with all steps specified, scheduled monitoring with
       intervals:
       "10 IU oxytocin IM immediately, then 5 IU/hr infusion for 4 hr";
       "BP > 160/110 mmHg + platelets < 100 ×10⁹/L → severe pre-eclampsia
       criteria";
       "BP and pulse every 30 min, temperature every 4 hr";
       "200 mg ferrous sulfate (65 mg elemental iron) + 400 µg folic
       acid daily; may be replaced by 185 mg ferrous fumarate".

  STRUCTURAL RULE: if you assign D3 ≥ 1, then D2 must be ≥ 1 — any
  actionable clinical guidance carries some meaningful content. If you
  find yourself wanting D3 ≥ 1 with D2 = 0, reconsider D2 (the guidance
  itself IS clinical content).

────────────────────────────────────────────────────────────────────────
D4 — Density relative to THIS query (0 / 1 / 2)
────────────────────────────────────────────────────────────────────────
What fraction of the chunk's text is directly useful for answering the
specific query? (Not the broader topic — the specific query.)

  0 — useful content is < 25% of the chunk:
       long chunk with one buried relevant sentence; mostly off-topic
       surrounding text; the answer exists somewhere but is dwarfed by
       irrelevant procedural detail or related-but-not-query-specific
       content.

  1 — useful content is 25-75% of the chunk:
       mixed — relevant content interleaved with adjacent-but-not-query-
       specific material (e.g., the right protocol followed by a long
       complications list when the query is about diagnosis only; useful
       dosing followed by task-shifting / cost discussion).

  2 — useful content is > 75% of the chunk:
       focused — the chunk is essentially the answer plus minor
       framing. Short focused chunks count as D4=2: signal-to-noise
       from the LLM's point of view, not absolute length, is what
       matters. A 50-character footnote that's 100% on the query is
       D4=2; a 1500-character chunk that's 70% off-the-query is D4=1.

  Note: D4 is judged AGAINST THE QUERY, not against the broader topic.
  A chunk fully about postpartum monitoring is D4=2 only if the query
  is also about postpartum monitoring. If the query is specifically
  "how often to check BP", and the chunk also covers fundal height
  and lochia at length, that's D4=1.

────────────────────────────────────────────────────────────────────────
Reasoning instruction
────────────────────────────────────────────────────────────────────────
Think carefully before producing the JSON output. Walk through each
dimension in order (D1 → D2 → D3 → D4), referring to specific
sentences or passages in the chunk where appropriate. Your reasoning
trace is captured separately; do NOT embed it in the JSON.

────────────────────────────────────────────────────────────────────────
Output format
────────────────────────────────────────────────────────────────────────
After reasoning, respond with valid JSON only — no "score" field, no
"reasoning" field; the score is computed downstream from D1 / D2 / D3 /
D4 and the reasoning is captured in the raw response.

{
  "d1_topic": true,
  "d2_meaningful": 2,
  "d3_actionable": 2,
  "d4_density": 2
}
```

---

## JUDGE_JSON_SCHEMA (vLLM guided decoding)

```python
JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "d1_topic":      {"type": "boolean"},
        "d2_meaningful": {"type": "integer", "minimum": 0, "maximum": 2},
        "d3_actionable": {"type": "integer", "minimum": 0, "maximum": 2},
        "d4_density":    {"type": "integer", "minimum": 0, "maximum": 2},
    },
    "required": [
        "d1_topic",
        "d2_meaningful",
        "d3_actionable",
        "d4_density",
    ],
    "additionalProperties": False,
}
```

---

## Output schema (JudgeResult — Python TypedDict)

```python
class JudgeResult(TypedDict):
    query_id:                str
    chunk_id:                str
    llm_judge_schema_version: str
    llm_judge_prompt_hash:   str
    llm_backend:             str
    llm_model:               str
    d1_topic:                bool
    d2_meaningful:           int   # 0 / 1 / 2 — set to 0 when d1_topic=False
    d3_actionable:           int   # 0 / 1 / 2 — set to 0 when d1_topic=False
    d4_density:              int   # 0 / 1 / 2 — set to 0 when d1_topic=False
```

No `score`, no `reasoning`. The aggregation `D1 × (D2 + D3 + D4)` (or any weighted variant) is computed by downstream consumers (`audit_metrics.py` and similar). This lets us re-aggregate without re-judging. The reasoning trace lives in a sibling raw-response file (see below).

The new `RESULT_SCHEMA_VERSION` will auto-derive a different hash from the modified annotations, distinguishing v2-graded labels from v1 3-level labels.

---

## Raw-reasoning side file

Each labeled (q, c) pair gets a matching record in `data/per_retriever_labels/<retriever>_<schema>_raw.jsonl`:

```python
class JudgeRawResponse(TypedDict):
    query_id:        str
    chunk_id:        str
    thinking:        str   # Qwen3 thinking-trace content, if available
    raw_text:        str   # Any non-thinking text emitted before the JSON
    raw_json:        str   # The literal JSON the model produced (pre-validation)
    llm_backend:     str
    llm_model:       str
    llm_judge_prompt_hash: str
```

The thinking trace is extracted via vLLM's `--reasoning-parser qwen3` (already set in `run_judge_relevance_job.sh`). For non-thinking backends or models the field is empty string. The raw-response file is parallel to the label file by `(query_id, chunk_id)` — same line count, same order.

---

## Worked examples to include in the prompt

These will go at the bottom of the SYSTEM_PROMPT as anchored end-to-end examples. Pulled from `notes/rubric_design_worked_examples.md`. Each is short on purpose so the prompt stays under ~3000 words total.

**Note on reasoning traces**: the prompt initially contains only query → chunk → expected JSON. After the first pilot we capture how the judge model actually reasons on these examples, correct any errors, and then optionally add the corrected traces as additional anchoring. This avoids forcing the judge into a reasoning pattern that's unnatural for it.

### Worked example 1 — score 6 anchor (D1=1, D2=2, D3=2, D4=2)

```
Query: What are the alternative dosages for iron and folic acid
       supplementation?
Chunk (WHO ANC 2016): "RECOMMENDATION A.2.1: Daily oral iron and folic
acid supplementation with 30 mg to 60 mg of elemental iron and 400 µg
(0.4 mg) folic acid is recommended for pregnant women to prevent
maternal anaemia, puerperal sepsis, low birth weight, and preterm
birth."
```

Expected JSON output:
```json
{
  "d1_topic": true,
  "d2_meaningful": 2,
  "d3_actionable": 2,
  "d4_density": 2
}
```

### Worked example 2 — score 4 (D1=1, D2=2, D3=0, D4=2)

```
Query: What clinical signs and laboratory findings indicate severe
       pre-eclampsia and HELLP syndrome?
Chunk (midwifery-preparation-for-practice): "HELLP SYNDROME — HELLP
syndrome is a rare but life-threatening liver disorder thought to be a
type of severe preeclampsia. Characterised by Haemolysis (destruction
of red blood cells), Elevated Liver enzymes (indicating liver damage),
and Low Platelet count. Many hypotheses attempt to define the
pathogenesis... Both HELLP and preeclampsia occur during the later
stages of pregnancy, and sometimes after childbirth..."
```

Expected JSON output:
```json
{
  "d1_topic": true,
  "d2_meaningful": 2,
  "d3_actionable": 0,
  "d4_density": 2
}
```

### Worked example 3 — score 4 (D1=1, D2=1, D3=1, D4=2)

```
Query: How should a midwife assess and manage uterine tone during the
       third stage of labour?
Chunk (who-midwifery-education-modules-2): "CHECKLIST OF SUB-TASKS FOR
THE MANAGEMENT OF THE THIRD STAGE OF LABOUR | Sub-task: Ensure uterus
well contracted | Knowledge: Consistency of contracted uterus | Skill:
Palpation of uterus and massage to promote contraction | Attitudes:
Accuracy, Gentleness"
```

Expected JSON output:
```json
{
  "d1_topic": true,
  "d2_meaningful": 1,
  "d3_actionable": 1,
  "d4_density": 2
}
```

### Worked example 4 — score 0 via D1 gate

```
Query: How often should maternal blood pressure, pulse, and temperature
       be checked after birth?
Chunk (hesperian-a-book-for-midwives Chapter 8 — Prenatal checkups):
"How to check blood pressure: The needle will begin to go back down.
As the air leaks out, you will start to hear the mother's pulse...
Check the mother's blood pressure at each visit. If her blood pressure
is going up, ask her to come back every week..."
```

Expected JSON output:
```json
{
  "d1_topic": false,
  "d2_meaningful": 0,
  "d3_actionable": 0,
  "d4_density": 0
}
```

---

## Implementation notes (for when we wire this into `scripts/judge_relevance.py`)

1. **Replace** `SYSTEM_PROMPT` with the v2 text above. Keep the existing `_build_user_content` (it formats the chunk well already).

2. **Replace** `JUDGE_JSON_SCHEMA` with the new schema (booleans → graded ints; add `d4_density`; remove implicit score).

3. **Replace** `JudgeResult` TypedDict — drop `score`, change `d2/d3` types, add `d4_density`. The auto-derived `RESULT_SCHEMA_VERSION` becomes a different `v-xxxxxxxx`.

4. **Update** `_finalize_record` (around line 478) — remove the score computation. Just pass through the four dimension fields after enforcing the structural rules:
   - If `d1_topic == False`: zero out d2/d3/d4
   - If `d3_actionable >= 1` and `d2_meaningful == 0`: bump `d2_meaningful` to 1 (or log a warning and keep the model's call — pick one). My recommendation: **bump silently** matching the existing behaviour on the boolean version.
   - Do NOT include `reasoning` in the final record.

5. **Add raw-response capture.** For each judged (q, c) pair, write a line to `data/per_retriever_labels/<retriever>_<schema>_raw.jsonl` containing:
   - `query_id`, `chunk_id`
   - `thinking` — Qwen3 reasoning-trace text (from vLLM's `--reasoning-parser qwen3` output; field name in the response is typically `reasoning_content` or `content` depending on parser config)
   - `raw_text` — any text the model emitted before the JSON object (empty for well-behaved completions)
   - `raw_json` — the literal JSON string the model produced (pre-validation)
   - `llm_backend`, `llm_model`, `llm_judge_prompt_hash`
   The raw-response file is parallel to the main label file by `(query_id, chunk_id)` order. Idempotent re-runs use the same dedup key. This file is gitignored (regenerable, potentially large).

6. **Add a `--rubric` flag** (`v1_boolean` default, `v2_graded` for new). Or write a new sibling script `scripts/judge_relevance_v2.py`. My recommendation: **add a flag** — single source of truth for shared infra (corpus loading, vLLM client, sharding, etc.). Branch only on prompt + schema + finalize_record + raw-capture sink.

7. **No changes needed** to `scripts/submit_judge_relevance.sh` or `run_judge_relevance_job.sh` — they pass through whatever script and prompt are in place. The `--reasoning-parser qwen3` flag in `run_judge_relevance_job.sh` is already set, which is what enables the thinking trace.

8. **Update** `IMPLEMENTATION_GUIDE.md` § 5 schema documentation to reflect the new label record shape AND describe the raw-response sibling file.

---

## What I'd like you to review

1. **Anchored level definitions** for D2 / D3 / D4 — does the language match how you'd score? Edge cases that aren't covered?
2. **Worked examples (query + chunk + expected JSON)** — would you score any of them differently? They're the highest-signal calibration data for the judge.
3. **Should D4 anchor mention text-length boundaries?** I've kept the cutoffs as fractions (<25%, 25-75%, >75%) rather than naming token/char counts. The intent is that the judge should think proportionally, not absolutely. But if you'd prefer concrete thresholds, those are easy to add.
4. **Structural rule enforcement**: when D3 ≥ 1 but model returns D2 = 0, should we (a) bump D2 to 1 silently (matches v1 behaviour), (b) reject the response and re-prompt, or (c) accept as-is and trust the judge? I lean (a).
5. **Raw-response file shape** — are `thinking`, `raw_text`, `raw_json` the right fields, or do you want a flatter format (e.g. just one `full_response` string)? My current proposal preserves the distinction in case we want to analyse thinking vs post-thinking separately.

---

## Iterative reasoning-trace workflow

The current draft keeps worked examples **input/output only** — no pre-written reasoning. The reasoning traces will be generated by the judge model itself on first pilot. The workflow:

| Step | What | Why |
|---|---|---|
| **1. Run the v2 prompt as drafted** on the 6 worked-example pairs (or the full 30-pair smoke set) | Captures the judge model's *natural* reasoning pattern via the qwen3 thinking trace, stored in the raw-response side file | Avoids forcing the model into our reasoning style — Qwen3.5-397B may reason differently than we'd anticipate |
| **2. Read the captured traces** and check whether they: (a) agree with our expected JSON, (b) follow a sensible D1 → D2 → D3 → D4 flow, (c) cite specific chunk content | The model's reasoning is the highest-fidelity anchor — but only if it's correct | Establishes whether we trust the natural reasoning or need to correct it |
| **3. For traces that agree and reason well**: keep them as anchor examples in the prompt (paste them in alongside the JSON, marked as "example reasoning"). | Strongest possible anchoring — model sees its own valid reasoning style modelled back | Reinforces good patterns |
| **4. For traces that disagree or reason badly**: either (a) edit the trace to fix the error and add to the prompt, or (b) leave that example without a reasoning anchor and let the model figure it out | Avoids embedding wrong reasoning in the prompt | Selective trust |
| **5. Re-pilot with the augmented prompt** and check whether agreement / discrimination improves | Validates that the anchoring improved performance | Iterative refinement |

This is a 2-3 day loop. The current draft is the starting point for step 1.

---

## Next steps

If the level definitions and worked-example JSON outputs look right:

1. **Wire into `judge_relevance.py`** — `--rubric v2_graded` flag, new prompt + schema + finalize logic + raw-response side-file sink (~2-3 h)
2. **6-pair micro-pilot on the worked examples** — does the judge produce the expected JSON outputs? Capture reasoning traces. Cluster job: ~30 min
3. **Read the captured reasoning** and apply the workflow above
4. **30-pair smoke test** on a wider sample to confirm score distribution looks reasonable
5. **Then proceed to GitHub #14** (full 100-query × 6-retriever pilot)
