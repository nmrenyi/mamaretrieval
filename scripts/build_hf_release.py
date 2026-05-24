#!/usr/bin/env python3
"""Build the v0.2.0 HuggingFace release bundle for ``nmrenyi/mamaretrieval``.

Reads the Tier-3 artefacts under ``data/`` and emits a release tree
under ``releases/mamaretrieval-hf-v0.2.0/`` split across two folders:

``data/``  — the primary eval artefacts (parquet configs):
- ``queries``    — 3,185 query records
- ``rankings``   — top-20 of each of 6 retrievers (long table)
- ``judgments``  — 230,964 v2-graded labels (no reasoning)
- ``chunks``     — chunk text

``audit/`` — provenance / reasoning trail (referenced from manifest, not
required to use the eval but useful for understanding how it was made):
- ``judgments_with_reasoning.parquet`` — judgments + the judge's ``thinking``
- ``query_generation_prompt.txt``      — verbatim generator prompt
- ``judge_relevance_prompt.txt``       — verbatim judge prompt

To rebuild the prior v0.1.0 bundle (top-3 union, 36k labels), check out the
script at commit 02127e3 (``git show 02127e3:scripts/build_hf_release.py``).

Chunk text comes from the released ``rag-bundle-v0.2.0`` tarball on the
``mamai-medical-guidelines`` GitHub release — the script downloads, verifies
SHA256, extracts ``chunks_for_rag.txt``, and caches both under ``.cache/``
so re-runs don't re-download. This means anyone with the repo can rebuild
the release end-to-end from scratch with nothing local.

Run from the repo root:
    /Users/renyi/miniforge3/bin/python scripts/build_hf_release.py

(The miniforge interpreter is pinned because the system python3 — Homebrew
3.14 on this machine — lacks pyarrow and pyyaml.)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import date
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamaretrieval.corpus import iter_chunks  # noqa: E402


JUDGE_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
QUERY_GENERATOR_MODEL = "Qwen/Qwen3.6-27B-FP8"
CORPUS_BUNDLE = "rag-bundle-v0.2.0"
CORPUS_BUNDLE_COMMIT = "a1abe003cce742b46954375d17abb28a3e27110f"
RAG_BUNDLE_URL = (
    "https://github.com/nmrenyi/mamai-medical-guidelines/releases/download/"
    "v0.2.0/rag-bundle-v0.2.0.tar.gz"
)
RAG_BUNDLE_SHA256 = "eca38fee3b191e98aa32771925ef65f17a2664ca7f453fd54256896afe3603a8"
RAG_BUNDLE_INNER_CHUNKS_PATH = "rag-bundle-v0.2.0/debug/chunks_for_rag.txt"
CHUNKS_FILE_SHA256 = "64c9e2d6a33a609585dbcaef41fe9196bff70f5afdd4a432a059410f90505917"
CACHE_DIR = REPO_ROOT / ".cache"
RETRIEVER_MODELS: dict[str, str] = {
    "bm25":   "BM25 (lexical baseline)",
    "medcpt": "ncbi/MedCPT (Query+Article Encoders)",
    "octen":  "Octen/Octen-Embedding-8B",
    "voyage": "voyage-4-large",
    "lateon": "lightonai/GTE-ModernColBERT-v1",
    "gecko":  "gecko-1024-quant-v0.2.0 (on-device TFLite)",
}
RETRIEVERS = list(RETRIEVER_MODELS.keys())
TOP_K = 20
VERSION = "v0.2.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _curl_resume(url: str, dest: Path) -> None:
    """Download `url` to `dest` with resume support — if `dest` exists from
    a previous interrupted run, curl's -C - picks up where it left off rather
    than restarting. Streams progress to the existing stderr so the build log
    shows the bar live."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-L", "-C", "-", "--fail", "-o", str(dest), url],
        check=True,
    )


def fetch_rag_bundle_chunks() -> Path:
    """Download the released rag-bundle, verify its SHA256, extract just the
    chunks_for_rag.txt file, and return its path. Both the tarball and the
    extracted chunks file are cached under .cache/, so re-runs fast-path
    via the chunks file's own SHA256 — no re-download or re-extract.

    The two hardcoded SHA256s pin the exact bytes used to produce this
    release; if either changes, the build aborts loudly rather than silently
    rebuilding against a different corpus."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = CACHE_DIR / RAG_BUNDLE_INNER_CHUNKS_PATH
    tarball = CACHE_DIR / "rag-bundle-v0.2.0.tar.gz"

    if chunks_path.exists() and _sha256(chunks_path) == CHUNKS_FILE_SHA256:
        return chunks_path

    if not tarball.exists():
        print(f"      downloading {RAG_BUNDLE_URL} (~503 MB)", flush=True)
        _curl_resume(RAG_BUNDLE_URL, tarball)

    actual = _sha256(tarball)
    if actual != RAG_BUNDLE_SHA256:
        raise SystemExit(
            f"SHA256 mismatch for {tarball}\n"
            f"  expected: {RAG_BUNDLE_SHA256}\n"
            f"  got:      {actual}\n"
            f"Delete the cached tarball and re-run, or update "
            f"RAG_BUNDLE_SHA256 if intentional."
        )

    print(f"      extracting {RAG_BUNDLE_INNER_CHUNKS_PATH}", flush=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extract(tar.getmember(RAG_BUNDLE_INNER_CHUNKS_PATH), path=CACHE_DIR)

    actual_chunks = _sha256(chunks_path)
    if actual_chunks != CHUNKS_FILE_SHA256:
        raise SystemExit(
            f"Extracted chunks file has unexpected SHA256\n"
            f"  expected: {CHUNKS_FILE_SHA256}\n"
            f"  got:      {actual_chunks}"
        )
    return chunks_path


def _load_script(name: str) -> ModuleType:
    """Load a script under scripts/ by path — that directory isn't a package,
    so a plain `import` doesn't work. Used to pull the verbatim prompts +
    prompt_hash + schema_version straight from the scripts that actually
    produced the data, so manifest provenance can't drift."""
    spec = importlib.util.spec_from_file_location(
        f"_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def build_queries(repo_root: Path, out_dir: Path) -> tuple[int, set[str]]:
    rows = list(read_jsonl(repo_root / "data" / "queries.jsonl"))
    table = pa.table(
        {
            "query_id": [r["query_id"] for r in rows],
            "query_text": [r["query_text"] for r in rows],
            "seed_chunk_id": [r["seed_chunk_id"] for r in rows],
        }
    )
    pq.write_table(table, out_dir / "queries.parquet", compression="snappy")
    seed_chunks = {r["seed_chunk_id"] for r in rows}
    return len(rows), seed_chunks


def build_rankings(repo_root: Path, out_dir: Path) -> tuple[int, set[str]]:
    rows: list[dict] = []
    chunk_ids: set[str] = set()
    for retriever in RETRIEVERS:
        path = repo_root / "data" / "full" / f"{retriever}_top20.jsonl"
        for record in read_jsonl(path):
            query_id = record["query_id"]
            for result in record["results"][:TOP_K]:
                rows.append(
                    {
                        "query_id": query_id,
                        "retriever": retriever,
                        "rank": result["rank"],
                        "chunk_id": result["chunk_id"],
                        "score": result.get("score"),
                    }
                )
                chunk_ids.add(result["chunk_id"])

    score_type = pa.float64()
    table = pa.table(
        {
            "query_id": pa.array([r["query_id"] for r in rows], pa.string()),
            "retriever": pa.array([r["retriever"] for r in rows], pa.string()),
            "rank": pa.array([r["rank"] for r in rows], pa.int32()),
            "chunk_id": pa.array([r["chunk_id"] for r in rows], pa.string()),
            "score": pa.array([r["score"] for r in rows], score_type),
        }
    )
    pq.write_table(table, out_dir / "rankings.parquet", compression="snappy")
    return len(rows), chunk_ids


def build_judgments(
    repo_root: Path, data_dir: Path, audit_dir: Path
) -> tuple[int, set[str]]:
    # v0.2.0: full top-20 union = Tier 2 top-3 labels + Tier 3 ranks 4-20 labels.
    # Raw thinking traces are split across 4 shard files (2 from Tier 2, 2 from
    # Tier 3) and indexed by (query_id, chunk_id) so the join is order-agnostic.
    clean_path = repo_root / "data" / "audit" / "v2_top20_all.jsonl"
    raw_shards = [
        repo_root / "data" / "audit" / "v2_full_h100_shard0.raw.jsonl",
        repo_root / "data" / "audit" / "v2_full_h100_shard1.raw.jsonl",
        repo_root / "data" / "audit" / "v2_top20_new_h100_shard0.raw.jsonl",
        repo_root / "data" / "audit" / "v2_top20_new_h100_shard1.raw.jsonl",
    ]
    thinking: dict[tuple[str, str], str] = {}
    for shard in raw_shards:
        for record in read_jsonl(shard):
            thinking[(record["query_id"], record["chunk_id"])] = record["thinking"]

    chunk_ids: set[str] = set()
    qids: list[str] = []
    cids: list[str] = []
    d1s: list[bool] = []
    d2s: list[int] = []
    d3s: list[int] = []
    d4s: list[int] = []
    scores: list[int] = []
    thinking_col: list[str] = []
    for record in read_jsonl(clean_path):
        d1 = bool(record["d1_topic"])
        d2 = int(record["d2_meaningful"])
        d3 = int(record["d3_actionable"])
        d4 = int(record["d4_density"])
        qids.append(record["query_id"])
        cids.append(record["chunk_id"])
        d1s.append(d1)
        d2s.append(d2)
        d3s.append(d3)
        d4s.append(d4)
        scores.append((1 if d1 else 0) * (d2 + d3 + d4))
        thinking_col.append(thinking[(record["query_id"], record["chunk_id"])])
        chunk_ids.add(record["chunk_id"])

    structured = pa.table(
        {
            "query_id": pa.array(qids, pa.string()),
            "chunk_id": pa.array(cids, pa.string()),
            "d1_topic": pa.array(d1s, pa.bool_()),
            "d2_meaningful": pa.array(d2s, pa.int8()),
            "d3_actionable": pa.array(d3s, pa.int8()),
            "d4_density": pa.array(d4s, pa.int8()),
            "score": pa.array(scores, pa.int8()),
        }
    )
    pq.write_table(structured, data_dir / "judgments.parquet", compression="snappy")

    with_reasoning = structured.append_column(
        "thinking", pa.array(thinking_col, pa.string())
    )
    pq.write_table(
        with_reasoning,
        audit_dir / "judgments_with_reasoning.parquet",
        compression="snappy",
    )
    return len(qids), chunk_ids


def build_query_generator_prompt(out_dir: Path, audit_dir: Path) -> dict:
    """Write audit/query_generation_prompt.txt (verbatim SYSTEM prompt + the
    user-message template rendered with placeholder tokens) and return the
    manifest metadata for the generator."""
    filter_mod = _load_script("llm_filter_chunks")
    placeholder_chunk = {
        "chunk_id": "<chunk_id>",
        "source": "<source>",
        "page": "<page>",
        "breadcrumb": "<breadcrumb>",
        "text": "<chunk text>",
    }
    user_template = filter_mod._build_user_content(placeholder_chunk)

    prompt_path = audit_dir / "query_generation_prompt.txt"
    prompt_path.write_text(
        "# Query generator prompt (per_chunk path)\n"
        "# ---------------------------------------------------------------\n"
        "# Sourced verbatim from scripts/llm_filter_chunks.py at release time.\n"
        "# prompt_hash and schema_version in manifest.json pin this content.\n"
        "# ---------------------------------------------------------------\n"
        "\n"
        "## SYSTEM message\n"
        "\n"
        f"{filter_mod.SYSTEM_PROMPT}\n"
        "\n"
        "## USER message template (rendered per chunk)\n"
        "\n"
        f"{user_template}\n"
    )
    return {
        "model": QUERY_GENERATOR_MODEL,
        "prompt_hash": filter_mod.PROMPT_HASH,
        "schema_version": filter_mod.RESULT_SCHEMA_VERSION,
        "prompt_file": str(prompt_path.relative_to(out_dir)),
    }


def build_judge_prompt(out_dir: Path, audit_dir: Path) -> dict:
    """Write audit/judge_relevance_prompt.txt (verbatim V2 SYSTEM prompt + the
    user-message template rendered with placeholders) and return the manifest
    metadata for the judge."""
    judge_mod = _load_script("judge_relevance")
    placeholder_chunk = {
        "source": "<source>",
        "page": "<page>",
        "text": "<chunk text>",
    }
    user_template = judge_mod._build_user_content("<query_text>", placeholder_chunk)

    prompt_path = audit_dir / "judge_relevance_prompt.txt"
    prompt_path.write_text(
        "# Judge prompt (v2 graded rubric)\n"
        "# ---------------------------------------------------------------\n"
        "# Sourced verbatim from scripts/judge_relevance.py at release time.\n"
        "# prompt_hash and schema_version in manifest.json pin this content.\n"
        "# ---------------------------------------------------------------\n"
        "\n"
        "## SYSTEM message\n"
        "\n"
        f"{judge_mod.V2_SYSTEM_PROMPT}\n"
        "\n"
        "## USER message template (rendered per query × chunk pair)\n"
        "\n"
        f"{user_template}\n"
    )
    return {
        "model": JUDGE_MODEL,
        "prompt_hash": judge_mod.V2_PROMPT_HASH,
        "schema_version": judge_mod.V2_RESULT_SCHEMA_VERSION,
        "prompt_file": str(prompt_path.relative_to(out_dir)),
    }


def build_chunks(repo_root: Path, out_dir: Path, referenced: set[str]) -> int:
    """Emit chunks.parquet (chunk_id + text) covering every chunk_id in
    ``referenced`` (union of seed_chunk_ids, ranking chunks, and judgment
    chunks).

    Also verifies every chunk in ``referenced`` is present in the corpus, so
    chunk-id drift between the rag-bundle and the rankings/judgments fails
    loudly here rather than silently downstream.
    """
    corpus_path = fetch_rag_bundle_chunks()

    rows: list[dict] = []
    seen: set[str] = set()
    for chunk in iter_chunks(corpus_path):
        if chunk.chunk_id not in referenced or chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        rows.append({"chunk_id": chunk.chunk_id, "text": chunk.text})

    missing = referenced - seen
    if missing:
        raise SystemExit(
            f"{len(missing)} referenced chunk_ids not found in corpus "
            f"({CORPUS_BUNDLE} @ {CORPUS_BUNDLE_COMMIT[:8]}); verify the bundle "
            "version matches the one used to produce the judgments."
        )

    pq.write_table(
        pa.Table.from_pylist(rows),
        out_dir / "chunks.parquet",
        compression="snappy",
    )

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=f"releases/mamaretrieval-hf-{VERSION}",
        help="Output directory (relative to repo root unless absolute).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    data_dir = out_dir / "data"
    audit_dir = out_dir / "audit"
    # Wipe and recreate the two build-owned subtrees so re-runs are idempotent
    # and stale artefacts from a prior layout (e.g. an old prompts/ dir or a
    # config file that moved between dirs) can't linger. Anything else under
    # out_dir (README, hand-written notes, etc.) is left untouched.
    for d in (data_dir, audit_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    print("[1/6] queries")
    n_queries, seed_chunks = build_queries(repo_root, data_dir)
    print(f"      wrote {n_queries} rows, {len(seed_chunks)} unique seed chunks")

    print(f"[2/6] rankings (top-{TOP_K} × {len(RETRIEVERS)} retrievers)")
    n_rankings, ranking_chunks = build_rankings(repo_root, data_dir)
    print(f"      wrote {n_rankings} rows, {len(ranking_chunks)} unique chunks")

    print("[3/6] judgments (data/) + judgments_with_reasoning (audit/)")
    n_judgments, judgment_chunks = build_judgments(repo_root, data_dir, audit_dir)
    print(f"      wrote {n_judgments} rows, {len(judgment_chunks)} unique chunks")

    referenced = seed_chunks | ranking_chunks | judgment_chunks
    print(f"[4/6] chunks (refs: {len(referenced)})")
    n_chunks = build_chunks(repo_root, data_dir, referenced)
    print(f"      wrote {n_chunks} rows")

    print("[5/6] query generator prompt")
    query_generator = build_query_generator_prompt(out_dir, audit_dir)
    print(f"      wrote {out_dir / query_generator['prompt_file']}")

    print("[6/6] judge prompt")
    judge = build_judge_prompt(out_dir, audit_dir)
    print(f"      wrote {out_dir / judge['prompt_file']}")

    manifest = {
        "name": "mamaretrieval",
        "version": VERSION,
        "release_date": date.today().isoformat(),
        "description": (
            "Per-retriever evaluation of 6 retrievers on 3,185 midwifery / "
            "OBGYN queries against the rag-bundle-v0.2.0 corpus, graded by an "
            "LLM judge under a 4-dimension rubric "
            "(D1 topic, D2 meaningful, D3 actionable, D4 density; "
            "score = d1 × (d2 + d3 + d4) ∈ [0..6])."
        ),
        "scope": (
            f"Tier 3: union of top-{TOP_K} from {len(RETRIEVERS)} retrievers — "
            "230,964 judged (q, c) pairs. Top-3 union (Tier 2's scope) is a "
            "strict subset, recoverable from rankings.parquet with rank <= 3."
        ),
        "previous_version": {
            "version": "v0.1.0",
            "scope": f"Tier 2: union of top-3, 36,418 judged pairs.",
            "build_commit": "02127e3",
        },
        "judge": judge,
        "query_generator": query_generator,
        "retrievers": [
            {"name": name, "model": model}
            for name, model in RETRIEVER_MODELS.items()
        ],
        "top_k": TOP_K,
        "corpus": {
            "bundle": CORPUS_BUNDLE,
            "producer_commit": CORPUS_BUNDLE_COMMIT,
        },
        "configs": {
            "queries": {"rows": n_queries},
            "rankings": {"rows": n_rankings, "depth": TOP_K},
            "judgments": {"rows": n_judgments},
            "judgments_with_reasoning": {
                "rows": n_judgments,
                "extra_columns": ["thinking"],
            },
            "chunks": {"rows": n_chunks},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"      wrote {out_dir/'manifest.json'}")

    preview_dir = out_dir.parent / f"{out_dir.name}-previews"
    build_previews([data_dir, audit_dir], preview_dir)


def build_previews(parquet_dirs: list[Path], preview_dir: Path) -> None:
    """Write a sibling folder of JSON first-row previews for every parquet
    found across the given dirs — for eyeballing only, not part of the HF
    bundle (the upload command targets out_dir, this is its sibling)."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = sorted(p for d in parquet_dirs for p in d.glob("*.parquet"))
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        first_row = table.slice(0, 1).to_pylist()[0] if table.num_rows else None
        preview = {
            "config": parquet_path.stem,
            "source_file": parquet_path.name,
            "num_rows": table.num_rows,
            "columns": [
                {"name": field.name, "type": str(field.type)}
                for field in table.schema
            ],
            "first_row": first_row,
        }
        out = preview_dir / f"{parquet_path.stem}.head.json"
        out.write_text(json.dumps(preview, indent=2, ensure_ascii=False) + "\n")
        print(f"      preview {out}")


if __name__ == "__main__":
    main()
