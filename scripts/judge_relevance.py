#!/usr/bin/env python
"""LLM-based relevance judge for Phase 2b.

For each (query, candidate_chunk) pair in the candidate pool, asks an LLM to
assess relevance on three binary dimensions:

  D1 — Topic:      Same clinical problem as the query
  D2 — Meaningful: Contains clinical information (background counts)
  D3 — Actionable: Contains specific clinical guidance (dosing, protocols)

Score computed in post-processing:
  - D1=False                        → score 0  (off topic, or on-topic but
  - D1=True, D2=False, D3=False     → score 0   meaningless)
  - D1=True, D2=True,  D3=False     → score 1  (on topic, background)
  - D1=True, D2=True,  D3=True      → score 2  (on topic, actionable)

Note: D3=True implies D2=True (enforced in post-processing, not by the model).

Usage:
    python scripts/judge_relevance.py                          # full run
    python scripts/judge_relevance.py --limit 5               # test: 5 queries
    python scripts/judge_relevance.py --resume                # continue after interruption
    python scripts/judge_relevance.py --workers 8             # more concurrency
    python scripts/judge_relevance.py --no-think              # faster smoke test
    python scripts/judge_relevance.py --shard 0 5             # cluster shard 0 of 5
    python scripts/judge_relevance.py \\
        --backend openai --model Qwen/Qwen3-32B               # vLLM backend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OLLAMA_URL        = "http://localhost:11434/api/chat"
OPENAI_BASE_URL   = "http://127.0.0.1:8000/v1"
DEFAULT_OLLAMA_MODEL  = "qwen3:32b"
DEFAULT_OPENAI_MODEL  = "Qwen/Qwen3-32B"

DEFAULT_INPUT  = Path("data/candidates.jsonl")
DEFAULT_OUTPUT = Path("data/relevance_labels.jsonl")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical relevance judge for MAMAI, a RAG system serving midwives \
and doctors in Zanzibar. Assess whether a guideline chunk is relevant to a \
clinical query by answering three questions.

D1 — Topic: Does this chunk address the same clinical problem as the query?

  Answer YES if the chunk directly concerns:
  - the same condition, diagnosis, or syndrome
  - the same clinical event or complication
  - the same drug, intervention, or procedure the query asks about
  - the same physiological process that is the direct subject of the query

  Answer NO if the chunk:
  - covers a different condition or procedure entirely
  - covers a broader category that includes the topic but does not address it
  - shares only a body system or patient population but addresses a different
    clinical problem

D2 — Meaningful: Does this chunk contain clinical information?
  YES: background principles, risk factors, pathophysiology, definitions,
       diagnostic criteria, management principles, or any clinical information
       relevant to the topic.
  NO:  administrative text, assessment rubrics, educator notes,
       table-of-contents listings, or content with no clinical information.

D3 — Actionable: Does this chunk contain specific clinical guidance a midwife
or doctor could act on?
  YES: dosing, protocols, diagnostic thresholds, management steps,
       contraindications, clinical criteria, specific recommendations.
  NO:  general background, broad principles, definitions, or narrative
       descriptions without specific guidance.

Respond with JSON only, reasoning first:
{"reasoning": "...", "d1_topic": true/false, "d2_meaningful": true/false, "d3_actionable": true/false}"""

# Enforced by vLLM guided decoding. Field order in "required" drives
# generation order — reasoning is emitted before the boolean verdicts.
JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning":     {"type": "string"},
        "d1_topic":      {"type": "boolean"},
        "d2_meaningful": {"type": "boolean"},
        "d3_actionable": {"type": "boolean"},
    },
    "required": ["reasoning", "d1_topic", "d2_meaningful", "d3_actionable"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class JudgeResult(TypedDict):
    query_id:                str
    chunk_id:                str
    llm_judge_schema_version: str
    llm_judge_prompt_hash:   str
    llm_backend:             str
    llm_model:               str
    reasoning:               str
    d1_topic:                bool
    d2_meaningful:           bool
    d3_actionable:           bool
    score:                   int   # 0 / 1 / 2, or -1 on error


RESULT_SCHEMA_VERSION = "v-" + hashlib.sha256(
    "\x00".join(sorted(JudgeResult.__annotations__)).encode()
).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_user_content(query_text: str, chunk: dict[str, Any]) -> str:
    source = chunk.get("source", "").strip()
    page   = chunk.get("page")
    text   = chunk.get("text", "").strip()

    meta_parts: list[str] = []
    if source:
        meta_parts.append(f"Source: {source}")
    if page is not None:
        meta_parts.append(f"Page: {page}")
    meta = " | ".join(meta_parts)

    chunk_header = f"{meta}\n---" if meta else "---"
    return f"## Query\n{query_text}\n\n## Guideline chunk\n{chunk_header}\n{text}\n---"


_SENTINEL_CHUNK: dict[str, Any] = {
    "source": "__sentinel__",
    "page": 0,
    "text": "Sentinel chunk text.",
}
_SENTINEL_QUERY = "What is the first-line treatment for postpartum hemorrhage?"

PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + "\x00" + _build_user_content(_SENTINEL_QUERY, _SENTINEL_CHUNK)
     ).encode("utf-8")
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# V2 graded rubric — score = D1 × (D2 + D3 + D4), each dim 0-2; computed
# downstream from the 4 dimensions (NOT emitted by the judge).
# ---------------------------------------------------------------------------

V2_SYSTEM_PROMPT = """\
You are a clinical relevance judge for MAMAI, a RAG system serving midwives \
and doctors in Zanzibar. Given a clinical query and a retrieved guideline \
chunk, score the chunk on four dimensions. Think carefully, then emit JSON.

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
What fraction of the chunk text is directly useful for answering the
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

────────────────────────────────────────────────────────────────────────
Worked examples
────────────────────────────────────────────────────────────────────────

### Example 1
Query: What are the alternative dosages for iron and folic acid supplementation?
Chunk (WHO ANC 2016): "RECOMMENDATION A.2.1: Daily oral iron and folic acid \
supplementation with 30 mg to 60 mg of elemental iron and 400 µg (0.4 mg) \
folic acid is recommended for pregnant women to prevent maternal anaemia, \
puerperal sepsis, low birth weight, and preterm birth."
Example reasoning:
D1: Iron and folic acid dosing in antenatal care — same topic, same context.
D1 = true.
D2: A formal recommendation with named indications (maternal anaemia,
puerperal sepsis, LBW, preterm birth) — multiple clinical concepts in one
paragraph. D2 = 2.
D3: Specific dose ranges (30-60 mg elemental iron, 400 µg folic acid daily)
with route (oral) — matches the "exact doses with route and frequency"
anchor. D3 = 2.
D4: The entire chunk is the recommendation statement. D4 = 2.
Expected JSON:
{"d1_topic": true, "d2_meaningful": 2, "d3_actionable": 2, "d4_density": 2}

### Example 2
Query: What clinical signs and laboratory findings indicate severe \
pre-eclampsia and HELLP syndrome?
Chunk (midwifery-preparation-for-practice): "HELLP SYNDROME — HELLP syndrome \
is a rare but life-threatening liver disorder thought to be a type of severe \
preeclampsia. Characterised by Haemolysis (destruction of red blood cells), \
Elevated Liver enzymes (indicating liver damage), and Low Platelet count. \
Many hypotheses attempt to define the pathogenesis... Both HELLP and \
preeclampsia occur during the later stages of pregnancy, and sometimes after \
childbirth..."
Example reasoning:
D1: HELLP syndrome and severe preeclampsia — same topic. D1 = true.
D2: Defines HELLP, breaks down the acronym with clinical meaning (RBC
destruction, liver damage, platelet drop), and discusses timing and
relationship to preeclampsia. Substantive clinical definitions. D2 = 2.
D3: The chunk defines the syndrome but provides no diagnostic thresholds
(e.g., AST > X, platelets < Y) and no management steps. Pure
background/pathophysiology. D3 = 0.
D4: The chunk is short and focused on HELLP. The pathogenesis text is
adjacent context but the core (definition + acronym) directly addresses
the query. D4 = 2.
Expected JSON:
{"d1_topic": true, "d2_meaningful": 2, "d3_actionable": 0, "d4_density": 2}

### Example 3
Query: How should a midwife assess and manage uterine tone during the third \
stage of labour?
Chunk (who-midwifery-education-modules-2): "CHECKLIST OF SUB-TASKS FOR THE \
MANAGEMENT OF THE THIRD STAGE OF LABOUR | Sub-task: Ensure uterus well \
contracted | Knowledge: Consistency of contracted uterus | Skill: Palpation \
of uterus and massage to promote contraction | Attitudes: Accuracy, \
Gentleness"
Example reasoning:
D1: Uterine tone in the third stage of labour — same topic, same timing.
D1 = true.
D2: Checklist row identifies the sub-task ("Ensure uterus well contracted"),
the knowledge ("Consistency"), and the skill ("Palpation and massage"). Brief
listing, no elaboration. D2 = 1.
D3: Names actions (palpate, massage) but no procedural specifics — how to
palpate, frequency, when soft vs firm, technique nuance. Partial guidance.
D3 = 1.
D4: The entire table row is on-topic for the query. D4 = 2.
Expected JSON:
{"d1_topic": true, "d2_meaningful": 1, "d3_actionable": 1, "d4_density": 2}

### Example 4
Query: How often should maternal blood pressure, pulse, and temperature be \
checked after birth?
Chunk (hesperian-a-book-for-midwives Chapter 8 — Prenatal checkups): "How to \
check blood pressure: The needle will begin to go back down. As the air leaks \
out, you will start to hear the mother's pulse... Check the mother's blood \
pressure at each visit. If her blood pressure is going up, ask her to come \
back every week..."
Example reasoning:
D1: Query asks about postpartum monitoring; chunk is from "Chapter 8
Prenatal checkups" and discusses BP "at each visit" and "come back every
week" — antenatal context. Same parameter, different clinical timing.
D1 = false.
D2-D4: zeroed per the structural rule (D1 = false).
Expected JSON:
{"d1_topic": false, "d2_meaningful": 0, "d3_actionable": 0, "d4_density": 0}"""


V2_JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "d1_topic":      {"type": "boolean"},
        "d2_meaningful": {"type": "integer", "minimum": 0, "maximum": 2},
        "d3_actionable": {"type": "integer", "minimum": 0, "maximum": 2},
        "d4_density":    {"type": "integer", "minimum": 0, "maximum": 2},
    },
    "required": [
        "d1_topic", "d2_meaningful", "d3_actionable", "d4_density",
    ],
    "additionalProperties": False,
}


class V2JudgeResult(TypedDict):
    query_id:                str
    chunk_id:                str
    llm_judge_schema_version: str
    llm_judge_prompt_hash:   str
    llm_backend:             str
    llm_model:               str
    d1_topic:                bool
    d2_meaningful:           int   # 0 / 1 / 2 — zeroed if d1_topic is False
    d3_actionable:           int   # 0 / 1 / 2 — zeroed if d1_topic is False
    d4_density:              int   # 0 / 1 / 2 — zeroed if d1_topic is False
    _error:                  str   # empty string on success; error msg on failure


V2_RESULT_SCHEMA_VERSION = "v-" + hashlib.sha256(
    "\x00".join(sorted(V2JudgeResult.__annotations__)).encode()
).hexdigest()[:8]

V2_PROMPT_HASH = hashlib.sha256(
    (V2_SYSTEM_PROMPT + "\x00" + _build_user_content(_SENTINEL_QUERY, _SENTINEL_CHUNK)
     ).encode("utf-8")
).hexdigest()[:16]


def _v2_apply_rules(d1: bool, d2: int, d3: int, d4: int) -> tuple[bool, int, int, int]:
    """Apply v2 structural rules.

    Rules:
      - D1=False → D2=D3=D4=0
      - D3 >= 1 → bump D2 to 1 if D2=0 (actionable carries meaningful)
      - All scores clamped to [0, 2]
    """
    if not d1:
        return False, 0, 0, 0
    d2 = max(0, min(2, d2))
    d3 = max(0, min(2, d3))
    d4 = max(0, min(2, d4))
    if d3 >= 1 and d2 == 0:
        d2 = 1
    return d1, d2, d3, d4


def _v2_safe_int(value: Any, default: int = 0) -> int:
    """Tolerant int coercion for v2 dim parsing (no schema enforcement)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        # Strip leading "score=", "D2=", etc., then keep just digits/sign
        for prefix in ("score=", "score: ", "D2=", "D3=", "D4="):
            if s.startswith(prefix):
                s = s[len(prefix):]
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return default
    return default


def _validate_v2_result(result: dict[str, Any]) -> V2JudgeResult:
    expected = set(V2JudgeResult.__annotations__)
    actual   = set(result.keys())
    missing  = expected - actual
    extra    = actual - expected
    if missing or extra:
        raise ValueError(
            f"V2JudgeResult schema mismatch — missing: {missing}, extra: {extra}"
        )
    return result  # type: ignore[return-value]


class V2RawCapture(TypedDict):
    query_id:              str
    chunk_id:              str
    thinking:              str
    raw_json:              str
    llm_backend:           str
    llm_model:             str
    llm_judge_prompt_hash: str


# Public dispatch table — keyed on --rubric value.
RUBRICS: dict[str, dict[str, Any]] = {
    "v1_boolean": {
        "system_prompt":   SYSTEM_PROMPT,
        "json_schema":     JUDGE_JSON_SCHEMA,
        "prompt_hash":     PROMPT_HASH,
        "schema_version":  RESULT_SCHEMA_VERSION,
    },
    "v2_graded": {
        "system_prompt":   V2_SYSTEM_PROMPT,
        "json_schema":     V2_JUDGE_JSON_SCHEMA,
        "prompt_hash":     V2_PROMPT_HASH,
        "schema_version":  V2_RESULT_SCHEMA_VERSION,
    },
}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_CORPUS_HEADER_RE = re.compile(
    r"<sep>\[SOURCE:(?P<source>[^|]+)\|PAGE:(?P<page>[^|]+)\|CID:(?P<cid>[^\]]+)\]"
)


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    """Parse corpus file → {chunk_id: {source, page, text}}."""
    corpus: dict[str, dict[str, Any]] = {}
    current_cid: str | None = None
    current_meta: dict[str, Any] = {}
    text_lines: list[str] = []

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = _CORPUS_HEADER_RE.match(line)
            if m:
                if current_cid is not None:
                    corpus[current_cid] = {
                        **current_meta,
                        "text": "\n".join(text_lines).strip(),
                    }
                current_cid = m.group("cid")
                raw_page = m.group("page")
                try:
                    page: Any = int(raw_page)
                except ValueError:
                    page = raw_page
                current_meta = {"source": m.group("source"), "page": page}
                text_lines = []
            else:
                text_lines.append(line)

    if current_cid is not None:
        corpus[current_cid] = {
            **current_meta,
            "text": "\n".join(text_lines).strip(),
        }

    return corpus


def _corpus_path_from_config(config_path: Path) -> Path | None:
    """Read corpus.chunks_path from config.yaml, expanding ~ if present."""
    if not config_path.exists():
        return None
    try:
        import yaml  # type: ignore[import]
        with config_path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        raw = cfg.get("corpus", {}).get("chunks_path")
        if raw:
            return Path(raw).expanduser()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",   default=str(DEFAULT_INPUT),
                   help="candidates.jsonl from Phase 2a. Default: %(default)s")
    p.add_argument("--corpus",  default=None,
                   help="Path to chunks_for_rag.txt. Falls back to corpus.chunks_path "
                        "in --config.")
    p.add_argument("--config",  default="config.yaml",
                   help="config.yaml for corpus path fallback. Default: %(default)s")
    p.add_argument("--output",  default=str(DEFAULT_OUTPUT),
                   help="Output JSONL: one record per (query, chunk) pair. "
                        "Default: %(default)s")
    p.add_argument("--rubric", choices=["v1_boolean", "v2_graded"], default="v1_boolean",
                   help="Rubric to use. v1_boolean: D1/D2/D3 booleans + score 0-2 "
                        "(original Phase 2b/3). v2_graded: D1 boolean + D2/D3/D4 each "
                        "0-2 (Phase 4 deployment-precision); score computed downstream. "
                        "Default: %(default)s")
    p.add_argument("--raw-output", default=None,
                   help="Path for raw-response side file (v2 only). Captures the "
                        "Qwen3 thinking trace and raw JSON per (q,c) pair. "
                        "Default: <output>.raw.jsonl when --rubric=v2_graded.")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                   help="LLM serving backend. Use 'openai' for vLLM. Default: %(default)s")
    p.add_argument("--model",   default=None,
                   help="Model name. Defaults to %(default)s per backend.")
    p.add_argument("--ollama-url", default=OLLAMA_URL,
                   help=f"Ollama chat endpoint. Default: {OLLAMA_URL}")
    p.add_argument("--base-url",   default=OPENAI_BASE_URL,
                   help=f"OpenAI-compatible base URL. Default: {OPENAI_BASE_URL}")
    p.add_argument("--api-key",    default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                   help="API key for OpenAI-compatible backend.")
    p.add_argument("--workers",    type=int, default=4,
                   help="Concurrent LLM requests. Default: %(default)s")
    p.add_argument("--timeout",    type=int, default=120,
                   help="Per-request timeout in seconds. Default: %(default)s")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max generated tokens per call (covers thinking + output). "
                        "Default: %(default)s")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation temperature. Default: %(default)s (deterministic)")
    p.add_argument("--no-think",        action="store_true",
                   help="Disable model thinking (faster smoke tests).")
    p.add_argument("--thinking-budget", type=int, default=0,
                   help="SOFT cap on thinking trace (via Qwen3 chat_template_kwargs). "
                        "Model is encouraged to wrap up at N tokens but may overshoot. "
                        "0 = unlimited.")
    p.add_argument("--thinking-token-budget", type=int, default=0,
                   help="HARD cap on thinking trace (vLLM ThinkingTokenBudgetLogitsProcessor, "
                        "PR #20859, available in vLLM 0.21.0+). Forcibly terminates the "
                        "<think>...</think> block at N tokens via logits masking. "
                        "0 = no hard cap. Recommended: set SOFT below HARD as belt-and-suspenders.")
    p.add_argument("--limit",      type=int, default=0,
                   help="Process at most N queries (0 = all). Default: %(default)s")
    p.add_argument("--shuffle",    action="store_true",
                   help="Shuffle queries before processing. Use with --limit for a "
                        "diverse random sample across sources.")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Random seed for --shuffle. Default: %(default)s")
    p.add_argument("--resume",     action="store_true",
                   help="Skip pairs already present in --output.")
    p.add_argument("--shard",      type=int, nargs=2, metavar=("INDEX", "COUNT"),
                   help="Process shard INDEX (0-based) of COUNT equal parts, "
                        "splitting by query.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_model(backend: str, model: str | None) -> str:
    if model:
        return model
    return DEFAULT_OPENAI_MODEL if backend == "openai" else DEFAULT_OLLAMA_MODEL


def _openai_chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    return exc.read().decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _pair_key(query_id: str, chunk_id: str) -> str:
    return f"{query_id}::{chunk_id}"


def _matches_contract(rec: dict[str, Any], rubric_name: str, backend: str, model: str) -> bool:
    spec = RUBRICS[rubric_name]
    return (
        rec.get("llm_judge_schema_version") == spec["schema_version"]
        and rec.get("llm_judge_prompt_hash") == spec["prompt_hash"]
        and rec.get("llm_backend") == backend
        and rec.get("llm_model") == model
    )


def _is_successful_record(rec: dict[str, Any], rubric_name: str) -> bool:
    """v1: score >= 0; v2: _error is the empty string."""
    if rubric_name == "v1_boolean":
        return rec.get("score", -1) >= 0
    return rec.get("_error", "") == ""


def _load_done_keys(
    output_path: Path, rubric_name: str, backend: str, model: str
) -> tuple[set[str], int]:
    """Return (done_keys, stale_count). Only successful labels for the given rubric count."""
    if not output_path.exists():
        return set(), 0
    done: set[str] = set()
    stale = 0
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _matches_contract(rec, rubric_name, backend, model) and _is_successful_record(rec, rubric_name):
                done.add(_pair_key(rec["query_id"], rec["chunk_id"]))
            else:
                stale += 1
    return done, stale


# ---------------------------------------------------------------------------
# Score computation (post-processing)
# ---------------------------------------------------------------------------

def _compute_score(d1: bool, d2: bool, d3: bool) -> tuple[bool, bool, int]:
    """Apply logical constraints and return (d2_final, d3_final, score).

    Rules:
      - D1=False → D2=False, D3=False, score=0
      - D3=True  → D2=True  (actionable implies meaningful)
      - score    = D1 * (D2 + D3)
    """
    if not d1:
        return False, False, 0
    if d3:
        d2 = True
    return d2, d3, int(d2) + int(d3)


def _validate_result(result: dict[str, Any]) -> JudgeResult:
    expected = set(JudgeResult.__annotations__)
    actual   = set(result.keys())
    missing  = expected - actual
    extra    = actual - expected
    if missing or extra:
        raise ValueError(
            f"JudgeResult schema mismatch — missing: {missing}, extra: {extra}"
        )
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# LLM call functions
# ---------------------------------------------------------------------------

def _call_openai(
    query_id:    str,
    query_text:  str,
    chunk_id:    str,
    chunk:       dict[str, Any],
    rubric_name: str,
    model:       str,
    base_url:    str,
    api_key:     str,
    timeout:     int,
    max_tokens:  int,
    temperature: float,
    think:            bool,
    thinking_budget:        int = 0,
    thinking_token_budget:  int = 0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Returns (result_record, raw_capture_or_None).

    raw_capture is only populated for rubric_name == 'v2_graded'.
    """
    user_content = _build_user_content(query_text, chunk)
    spec = RUBRICS[rubric_name]

    chat_template_kwargs: dict[str, Any] = {"enable_thinking": think}
    if think and thinking_budget > 0:
        # SOFT cap via Qwen3 chat template: model encouraged to wrap up.
        chat_template_kwargs["thinking_budget"] = thinking_budget

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": spec["system_prompt"]},
            {"role": "user",   "content": user_content},
        ],
        "temperature":   temperature,
        "chat_template_kwargs": chat_template_kwargs,
    }
    # max_tokens=0 (or negative) → omit from request, let vLLM use
    # the default (max_model_len - input_tokens). With thinking_budget
    # set, runaway protection comes from there, not from max_tokens.
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    # HARD cap on thinking trace via vLLM ThinkingTokenBudgetLogitsProcessor.
    # Forcibly closes <think>...</think> at N tokens; JSON output continues
    # unaffected after the forced close.
    if thinking_token_budget > 0:
        payload["thinking_token_budget"] = thinking_token_budget
    if rubric_name == "v1_boolean":
        # Strict schema enforcement (legacy behavior).
        payload["guided_json"] = spec["json_schema"]
    # v2_graded: no structured-output constraint of any kind. vLLM's
    # `enable_in_reasoning=False` means BOTH guided_json AND
    # response_format=json_object suppress the thinking trace. We rely on
    # the prompt's "respond with valid JSON only" instruction + tolerant
    # parsing for the output.

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        _openai_chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = _read_http_error_body(exc)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    msg    = response["choices"][0]["message"]
    raw    = (msg.get("content") or "").strip()
    # vLLM 0.9+ exposes the reasoning trace as `message.reasoning`. Older
    # versions used `reasoning_content`. Read both so the parser works on
    # any vLLM version.
    thinking = (msg.get("reasoning") or msg.get("reasoning_content") or "")

    # The reasoning parser may leak think tokens into content in two ways:
    #   (a) full block: <think>...</think>JSON  — strip the whole block
    #   (b) orphaned closing tag: </think>JSON  — strip just the closing tag
    # In both cases, capture the thinking content to `thinking` if the
    # reasoning_content field wasn't populated separately.
    if raw.startswith("<think>"):
        end = raw.find("</think>")
        if end != -1:
            if not thinking:
                thinking = raw[len("<think>"):end].strip()
            raw = raw[end + len("</think>"):].strip()
        else:
            raw = ""
    elif raw.startswith("</think>"):
        raw = raw[len("</think>"):].strip()

    # Strip markdown code fence (```json\n...\n``` or ```\n...\n```)
    if raw.startswith("```"):
        newline = raw.find("\n")
        if newline != -1:
            raw = raw[newline + 1:].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    if not raw:
        snippet = thinking[:3000] if thinking else "(no reasoning_content in response)"
        raise json.JSONDecodeError(
            f"Empty content after thinking. reasoning_content[:3000]:\n{snippet}",
            "", 0,
        )

    # Without schema enforcement, the model may emit free-text reasoning
    # before/after the JSON object. Find the first `{`, parse one JSON
    # object via raw_decode (tolerates trailing text), capture any
    # pre-JSON text into `thinking` if not already populated.
    json_start = raw.find("{")
    if json_start == -1:
        raise json.JSONDecodeError(
            f"No JSON object found in content. raw_repr={repr(raw[:300])}",
            raw, 0,
        )
    if json_start > 0:
        leading = raw[:json_start].strip()
        if leading and not thinking:
            thinking = leading
        raw = raw[json_start:]
    try:
        parsed, _consumed = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError as _je:
        raise json.JSONDecodeError(
            f"JSON parse failed. raw_repr={repr(raw[:300])}: {_je.msg}",
            raw, _je.pos,
        ) from _je

    if rubric_name == "v1_boolean":
        d1 = bool(parsed["d1_topic"])
        d2 = bool(parsed["d2_meaningful"])
        d3 = bool(parsed["d3_actionable"])
        d2_final, d3_final, score = _compute_score(d1, d2, d3)
        result = _validate_result({
            "query_id":                query_id,
            "chunk_id":                chunk_id,
            "llm_judge_schema_version": RESULT_SCHEMA_VERSION,
            "llm_judge_prompt_hash":   PROMPT_HASH,
            "llm_backend":             "openai",
            "llm_model":               model,
            "reasoning":               str(parsed.get("reasoning", "")),
            "d1_topic":                d1,
            "d2_meaningful":           d2_final,
            "d3_actionable":           d3_final,
            "score":                   score,
        })
        return result, None

    # v2_graded — tolerant parsing (no schema enforcement)
    d1_raw = bool(parsed.get("d1_topic", False))
    d2_raw = _v2_safe_int(parsed.get("d2_meaningful"))
    d3_raw = _v2_safe_int(parsed.get("d3_actionable"))
    d4_raw = _v2_safe_int(parsed.get("d4_density"))
    d1_f, d2_f, d3_f, d4_f = _v2_apply_rules(d1_raw, d2_raw, d3_raw, d4_raw)
    result = _validate_v2_result({
        "query_id":                query_id,
        "chunk_id":                chunk_id,
        "llm_judge_schema_version": V2_RESULT_SCHEMA_VERSION,
        "llm_judge_prompt_hash":   V2_PROMPT_HASH,
        "llm_backend":             "openai",
        "llm_model":               model,
        "d1_topic":                d1_f,
        "d2_meaningful":           d2_f,
        "d3_actionable":           d3_f,
        "d4_density":              d4_f,
        "_error":                  "",
    })
    raw_capture = {
        "query_id":              query_id,
        "chunk_id":              chunk_id,
        "thinking":              thinking,
        "raw_json":              raw,
        "llm_backend":           "openai",
        "llm_model":             model,
        "llm_judge_prompt_hash": V2_PROMPT_HASH,
    }
    return result, raw_capture


def _call_ollama(
    query_id:    str,
    query_text:  str,
    chunk_id:    str,
    chunk:       dict[str, Any],
    rubric_name: str,
    model:       str,
    url:         str,
    timeout:     int,
    max_tokens:  int,
    temperature: float,
    think:       bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    user_content = _build_user_content(query_text, chunk)
    spec = RUBRICS[rubric_name]

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": spec["system_prompt"]},
            {"role": "user",   "content": user_content},
        ],
        "stream": False,
        "think": think,
        "format": "json",   # valid JSON only; no schema enforcement
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read())

    raw      = response["message"]["content"].strip()
    thinking = response["message"].get("thinking", "") or ""
    parsed   = json.loads(raw)

    if rubric_name == "v1_boolean":
        d1 = bool(parsed.get("d1_topic",     False))
        d2 = bool(parsed.get("d2_meaningful", False))
        d3 = bool(parsed.get("d3_actionable", False))
        d2_final, d3_final, score = _compute_score(d1, d2, d3)
        result = _validate_result({
            "query_id":                query_id,
            "chunk_id":                chunk_id,
            "llm_judge_schema_version": RESULT_SCHEMA_VERSION,
            "llm_judge_prompt_hash":   PROMPT_HASH,
            "llm_backend":             "ollama",
            "llm_model":               model,
            "reasoning":               str(parsed.get("reasoning", "")),
            "d1_topic":                d1,
            "d2_meaningful":           d2_final,
            "d3_actionable":           d3_final,
            "score":                   score,
        })
        return result, None

    # v2_graded — tolerant parsing
    d1_raw = bool(parsed.get("d1_topic", False))
    d2_raw = _v2_safe_int(parsed.get("d2_meaningful"))
    d3_raw = _v2_safe_int(parsed.get("d3_actionable"))
    d4_raw = _v2_safe_int(parsed.get("d4_density"))
    d1_f, d2_f, d3_f, d4_f = _v2_apply_rules(d1_raw, d2_raw, d3_raw, d4_raw)
    result = _validate_v2_result({
        "query_id":                query_id,
        "chunk_id":                chunk_id,
        "llm_judge_schema_version": V2_RESULT_SCHEMA_VERSION,
        "llm_judge_prompt_hash":   V2_PROMPT_HASH,
        "llm_backend":             "ollama",
        "llm_model":               model,
        "d1_topic":                d1_f,
        "d2_meaningful":           d2_f,
        "d3_actionable":           d3_f,
        "d4_density":              d4_f,
        "_error":                  "",
    })
    raw_capture = {
        "query_id":              query_id,
        "chunk_id":              chunk_id,
        "thinking":              thinking,
        "raw_json":              raw,
        "llm_backend":           "ollama",
        "llm_model":             model,
        "llm_judge_prompt_hash": V2_PROMPT_HASH,
    }
    return result, raw_capture


def _process_one(
    query_id:    str,
    query_text:  str,
    chunk_id:    str,
    chunk:       dict[str, Any],
    rubric_name: str,
    backend:     str,
    model:       str,
    ollama_url:  str,
    base_url:    str,
    api_key:     str,
    timeout:     int,
    max_tokens:  int,
    temperature: float,
    think:            bool,
    thinking_budget:        int = 0,
    thinking_token_budget:  int = 0,
    retries:          int = 2,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Judge one pair, retrying transient errors up to `retries` times.

    Returns (result_record, raw_capture_or_None). raw_capture is only
    populated for rubric_name == 'v2_graded'.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if backend == "openai":
                return _call_openai(
                    query_id, query_text, chunk_id, chunk, rubric_name,
                    model, base_url, api_key, timeout, max_tokens, temperature, think,
                    thinking_budget, thinking_token_budget,
                )
            return _call_ollama(
                query_id, query_text, chunk_id, chunk, rubric_name,
                model, ollama_url, timeout, max_tokens, temperature, think,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            last_err = exc
            break  # malformed output — no point retrying
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(2 ** attempt)

    err_msg = f"error: {type(last_err).__name__}: {last_err}"
    if rubric_name == "v1_boolean":
        return _validate_result({
            "query_id":                query_id,
            "chunk_id":                chunk_id,
            "llm_judge_schema_version": RESULT_SCHEMA_VERSION,
            "llm_judge_prompt_hash":   PROMPT_HASH,
            "llm_backend":             backend,
            "llm_model":               model,
            "reasoning":               err_msg,
            "d1_topic":                False,
            "d2_meaningful":           False,
            "d3_actionable":           False,
            "score":                   -1,
        }), None

    # v2_graded error record
    return _validate_v2_result({
        "query_id":                query_id,
        "chunk_id":                chunk_id,
        "llm_judge_schema_version": V2_RESULT_SCHEMA_VERSION,
        "llm_judge_prompt_hash":   V2_PROMPT_HASH,
        "llm_backend":             backend,
        "llm_model":               model,
        "d1_topic":                False,
        "d2_meaningful":           0,
        "d3_actionable":           0,
        "d4_density":              0,
        "_error":                  err_msg,
    }), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    model = _resolve_model(args.backend, args.model)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # Resolve corpus path: CLI arg → config.yaml fallback
    if args.corpus:
        corpus_path = Path(args.corpus).expanduser()
    else:
        corpus_path = _corpus_path_from_config(Path(args.config))
        if corpus_path is None:
            print(
                "ERROR: --corpus not provided and could not read corpus.chunks_path "
                f"from {args.config}.",
                file=sys.stderr,
            )
            return 1

    for path, label in [
        (input_path,  "candidates (--input)"),
        (corpus_path, "corpus (--corpus)"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    # Load corpus
    print(f"Loading corpus from {corpus_path} ...")
    corpus = _load_corpus(corpus_path)
    print(f"  {len(corpus):,} chunks loaded.")

    # Load queries with candidate lists
    queries = _read_jsonl(input_path)
    print(f"Loaded {len(queries):,} queries from {input_path}")

    # Shard by query index (all candidates of a query stay in the same shard)
    if args.shard is not None:
        shard_idx, shard_count = args.shard
        if not (0 <= shard_idx < shard_count):
            print(f"ERROR: invalid shard {shard_idx}/{shard_count}", file=sys.stderr)
            return 1
        shard_size = math.ceil(len(queries) / shard_count)
        start  = shard_idx * shard_size
        queries = queries[start : start + shard_size]
        print(
            f"Shard {shard_idx}/{shard_count}: {len(queries):,} queries "
            f"(indices {start}–{start + len(queries) - 1})"
        )

    if args.shuffle:
        import random
        random.Random(args.shuffle_seed).shuffle(queries)
        print(f"Shuffled with seed={args.shuffle_seed}")

    if args.limit > 0:
        queries = queries[: args.limit]
        print(f"Limited to {args.limit} queries.")

    # Load already-labeled pairs for --resume
    done_keys: set[str] = set()
    if args.resume:
        done_keys, stale = _load_done_keys(output_path, args.rubric, args.backend, model)
        print(f"Resuming: {len(done_keys):,} pairs already labeled.")
        if stale:
            print(f"  {stale:,} stale / errored records will be re-processed.")

    # Flatten queries → individual work items, skipping done and missing chunks
    todo: list[tuple[str, str, str]] = []  # (query_id, query_text, chunk_id)
    missing_chunks = 0
    for q in queries:
        qid   = q["query_id"]
        qtext = q["query_text"]
        for cand in q["candidates"]:
            cid = cand["chunk_id"]
            if _pair_key(qid, cid) in done_keys:
                continue
            if cid not in corpus:
                missing_chunks += 1
                continue
            todo.append((qid, qtext, cid))

    if missing_chunks:
        print(
            f"Warning: {missing_chunks:,} candidate chunk IDs not found in corpus "
            "— skipped.",
            file=sys.stderr,
        )
    if not todo:
        print("Nothing to process.")
        return 0

    print(
        f"Processing {len(todo):,} pairs | rubric={args.rubric} "
        f"backend={args.backend} model={model} workers={args.workers} "
        f"timeout={args.timeout}s max_tokens={args.max_tokens} "
        f"temperature={args.temperature} think={not args.no_think}"
    )
    print(f"Output → {output_path}")

    # Resolve raw-output path (v2 only)
    raw_path: Path | None = None
    if args.rubric == "v2_graded":
        raw_path = Path(args.raw_output) if args.raw_output else output_path.with_suffix(".raw.jsonl")
        print(f"Raw responses → {raw_path}")

    # --- Run ---
    out_lock     = Lock()
    raw_lock     = Lock()
    # v1 has 3 score buckets (0/1/2); v2 has 7 (0..6 computed downstream from D1*(D2+D3+D4))
    score_counts: dict[int, int] = {k: 0 for k in (range(3) if args.rubric == "v1_boolean" else range(7))}
    errors       = 0
    start_time   = time.monotonic()

    out_mode = "a" if args.resume else "w"
    out_fh   = output_path.open(out_mode, encoding="utf-8")
    raw_fh   = raw_path.open(out_mode, encoding="utf-8") if raw_path else None

    def _on_done(item: tuple[dict[str, Any], dict[str, Any] | None]) -> None:
        nonlocal errors
        result, raw_capture = item

        with out_lock:
            out_fh.write(json.dumps(result) + "\n")
            out_fh.flush()

        if raw_capture is not None and raw_fh is not None:
            with raw_lock:
                raw_fh.write(json.dumps(raw_capture) + "\n")
                raw_fh.flush()

        # Compute display score
        if args.rubric == "v1_boolean":
            is_err = result.get("score", -1) < 0
            disp_score = result.get("score", -1)
        else:
            is_err = result.get("_error", "") != ""
            d1 = bool(result.get("d1_topic", False))
            disp_score = (
                (int(result["d2_meaningful"]) + int(result["d3_actionable"]) + int(result["d4_density"]))
                if d1 else 0
            )

        if is_err:
            errors += 1
        else:
            score_counts[disp_score] = score_counts.get(disp_score, 0) + 1

        total   = sum(score_counts.values()) + errors
        elapsed = time.monotonic() - start_time
        rate    = total / elapsed if elapsed > 0 else 0
        eta     = (len(todo) - total) / rate if rate > 0 else float("inf")
        eta_str = f"{eta / 60:.0f}m" if eta < 7200 else f"{eta / 3600:.1f}h"

        status  = "ERR" if is_err else f"s={disp_score}"
        dist    = " ".join(f"{k}:{v}" for k, v in sorted(score_counts.items()))
        print(
            f"[{total:>6}/{len(todo)}] {status} | "
            f"{rate:.1f}/s ETA {eta_str} | "
            f"scores({dist}) err={errors} | "
            f"{result['query_id'][:8]}..{result['chunk_id'][:8]}"
        )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    qid, qtext, cid, corpus[cid],
                    args.rubric,
                    args.backend, model,
                    args.ollama_url, args.base_url, args.api_key,
                    args.timeout, args.max_tokens, args.temperature,
                    not args.no_think, args.thinking_budget, args.thinking_token_budget,
                ): (qid, cid)
                for qid, qtext, cid in todo
            }
            for future in as_completed(futures):
                _on_done(future.result())
    finally:
        out_fh.close()
        if raw_fh is not None:
            raw_fh.close()

    elapsed = time.monotonic() - start_time
    total   = sum(score_counts.values()) + errors
    print()
    print(f"Done in {elapsed:.0f}s — {total:,} pairs judged, {errors} errors")
    print("Score distribution: " +
          "  ".join(f"score={k}: {v}" for k, v in sorted(score_counts.items())))

    # On resume: deduplicate output, keeping only current-schema successful labels
    if args.resume:
        seen:          set[str]          = set()
        deduped:       list[dict[str, Any]] = []
        error_records: list[dict[str, Any]] = []
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not _matches_contract(rec, args.rubric, args.backend, model):
                    continue
                if not _is_successful_record(rec, args.rubric):
                    error_records.append(rec)
                    continue
                key = _pair_key(rec["query_id"], rec["chunk_id"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(rec)
        with output_path.open("w", encoding="utf-8") as fh:
            for rec in deduped:
                fh.write(json.dumps(rec) + "\n")
        if error_records:
            error_path = output_path.parent / (output_path.stem + "_errors.jsonl")
            with error_path.open("w", encoding="utf-8") as fh:
                for rec in error_records:
                    fh.write(json.dumps(rec) + "\n")
            print(f"Error records saved:  {len(error_records):,} → {error_path}")
        print(f"Deduplicated output: {len(deduped):,} labeled pairs → {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
