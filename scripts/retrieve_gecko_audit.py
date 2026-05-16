#!/usr/bin/env python3
"""Retrieve Gecko top-K candidates for the Phase-3 audit queries.

Reads the v0.2.0 RAG bundle (matching the corpus the other audit retrievers
operate on) and the on-device Gecko TFLite model. Embeds each audit query
with Gecko, runs cosine top-K against the pre-computed chunk embeddings
in the v0.2.0 embeddings.sqlite, and writes per-query results in the same
schema as the other `data/audit/<retriever>_top20.jsonl` files.

Key choices (matching `mamai/evaluation/retrieval.py` and `RagPipeline.kt`):
- Tokenize with SentencePiece, pad/truncate to model max_length with id=0
- L2-normalize both query and chunk embeddings before dot product
- chunk_id (audit CID) is taken from the v0.2.0 chunks_for_rag.txt at the
  same ROWID position as the sqlite row (verified contract)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_BUNDLE_ROOT = Path("/Users/renyi/Downloads/mamai-medical-guidelines/releases/rag-bundle-v0.2.0")
DEFAULT_SQLITE = DEFAULT_BUNDLE_ROOT / "runtime/embeddings.sqlite"
DEFAULT_CHUNKS_TXT = Path("/Users/renyi/Downloads/mamai-medical-guidelines/processed/chunks_for_rag.txt")
DEFAULT_MODEL = Path("/Users/renyi/Downloads/mamai/device_push/models/Gecko_1024_quant.tflite")
DEFAULT_TOKENIZER = Path("/Users/renyi/Downloads/mamai/device_push/models/sentencepiece.model")

HEADER_RE = re.compile(r"^<sep>\[SOURCE:([^|]+)\|PAGE:(\d+)\|CID:([a-f0-9]+)\]$")


def load_chunk_ids(chunks_txt: Path) -> list[str]:
    """Return CIDs in file order. ROWID i in sqlite -> CIDs[i-1]."""
    cids: list[str] = []
    with chunks_txt.open() as f:
        for line in f:
            m = HEADER_RE.match(line.rstrip("\n"))
            if m:
                cids.append(m.group(3))
    return cids


def load_corpus_embeddings(sqlite_path: Path, expected_dim: int = 768) -> np.ndarray:
    """Load chunk embeddings from embeddings.sqlite. Blob = 4-byte 'VF32' + N float32."""
    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute("SELECT ROWID, embeddings FROM rag_vector_store ORDER BY ROWID").fetchall()
    conn.close()
    n = len(rows)
    embs = np.empty((n, expected_dim), dtype=np.float32)
    for i, (rowid, blob) in enumerate(rows):
        if rowid != i + 1:
            raise RuntimeError(f"ROWID gap at index {i}: rowid={rowid}")
        n_floats = (len(blob) - 4) // 4
        if n_floats != expected_dim:
            raise RuntimeError(f"row {rowid}: expected {expected_dim}-dim, got {n_floats}")
        embs[i] = np.frombuffer(blob[4:], dtype=np.float32)
    return embs


class GeckoEmbedder:
    """Gecko TFLite embedder replicating RagPipeline.kt's pipeline."""

    def __init__(self, model_path: Path, tokenizer_path: Path):
        import sentencepiece as spm
        try:
            from ai_edge_litert import interpreter as tflite
        except ImportError:  # fallback
            try:
                import tflite_runtime.interpreter as tflite  # type: ignore
            except ImportError:
                import tensorflow as tf  # type: ignore
                tflite = tf.lite

        self.interp = tflite.Interpreter(model_path=str(model_path), num_threads=4)
        self.interp.allocate_tensors()
        self.ins = self.interp.get_input_details()
        self.outs = self.interp.get_output_details()
        self.max_len = int(self.ins[0]["shape"][1])
        self.dim = int(self.outs[0]["shape"][-1])

        self.tok = spm.SentencePieceProcessor()
        self.tok.Load(str(tokenizer_path))
        print(f"Gecko: max_tokens={self.max_len}, dim={self.dim}", flush=True)

    def embed(self, text: str) -> np.ndarray:
        ids = self.tok.encode_as_ids(text)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [0] * (self.max_len - len(ids))
        x = np.array([ids], dtype=np.int32)
        self.interp.set_tensor(self.ins[0]["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.outs[0]["index"]).flatten().astype(np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def sanity_check_pairs(embedder: GeckoEmbedder) -> dict[str, float]:
    """Confirm cosine similarities make semantic sense."""
    pairs = {
        "same-topic (PPH dosing)": (
            "What is the dose of oxytocin for postpartum hemorrhage?",
            "How much oxytocin should be administered to treat PPH?",
        ),
        "same-topic (anc visits)": (
            "How many antenatal care visits are recommended in pregnancy?",
            "What is the recommended number of ANC visits?",
        ),
        "unrelated (PPH vs jaundice)": (
            "What is the dose of oxytocin for postpartum hemorrhage?",
            "How is neonatal jaundice diagnosed in newborns?",
        ),
    }
    results: dict[str, float] = {}
    for label, (a, b) in pairs.items():
        ea, eb = embedder.embed(a), embedder.embed(b)
        ea = ea / np.linalg.norm(ea)
        eb = eb / np.linalg.norm(eb)
        cos = float(np.dot(ea, eb))
        results[label] = cos
        print(f"  cosine[{label}] = {cos:.3f}", flush=True)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", default="data/queries_audit.jsonl")
    p.add_argument("--query-ids", default="data/audit/query_ids.txt")
    p.add_argument("--sqlite", default=str(DEFAULT_SQLITE))
    p.add_argument("--chunks-txt", default=str(DEFAULT_CHUNKS_TXT))
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output", default="data/audit/gecko_top20.jsonl")
    p.add_argument("--queries-npy", default="data/audit/gecko_queries.npy")
    p.add_argument("--skip-retrieve", action="store_true",
                   help="Only run the sanity check (Step 1 isolation)")
    args = p.parse_args()

    print(f"=== Gecko v0.2.0 audit retrieval ===", flush=True)
    print(f"  model: {args.model}", flush=True)
    print(f"  tokenizer: {args.tokenizer}", flush=True)
    print(f"  corpus sqlite: {args.sqlite}", flush=True)
    print(f"  corpus chunks: {args.chunks_txt}", flush=True)
    print(flush=True)

    print(f"Loading Gecko model...", flush=True)
    t0 = time.time()
    embedder = GeckoEmbedder(Path(args.model), Path(args.tokenizer))
    print(f"  load time: {time.time()-t0:.1f}s", flush=True)

    print(f"\n=== Sanity check ===", flush=True)
    sanity = sanity_check_pairs(embedder)
    same_min = min(v for k, v in sanity.items() if k.startswith("same-topic"))
    unrel = sanity["unrelated (PPH vs jaundice)"]
    sanity_ok = same_min > 0.5 and unrel < 0.3
    print(f"  same-topic min: {same_min:.3f} (want > 0.5)", flush=True)
    print(f"  unrelated:      {unrel:.3f} (want < 0.3)", flush=True)
    print(f"  sanity: {'PASS' if sanity_ok else 'FAIL'}", flush=True)
    if not sanity_ok:
        print("WARN: sanity check did not pass thresholds; continuing for inspection", file=sys.stderr)

    if args.skip_retrieve:
        return 0 if sanity_ok else 1

    print(f"\n=== Load corpus ===", flush=True)
    t0 = time.time()
    chunk_ids = load_chunk_ids(Path(args.chunks_txt))
    print(f"  CIDs from chunks_for_rag.txt: {len(chunk_ids)} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    corpus_embs = load_corpus_embeddings(Path(args.sqlite))
    print(f"  corpus embeddings: shape={corpus_embs.shape} dtype={corpus_embs.dtype} ({time.time()-t0:.1f}s)", flush=True)
    if corpus_embs.shape[0] != len(chunk_ids):
        print(f"ERROR: chunk count mismatch: sqlite={corpus_embs.shape[0]} txt={len(chunk_ids)}", file=sys.stderr)
        return 1

    corpus_norm = l2_normalize(corpus_embs)

    print(f"\n=== Load audit queries ===", flush=True)
    queries_by_id: dict[str, dict] = {}
    for line in Path(args.queries).open():
        rec = json.loads(line)
        queries_by_id[rec["query_id"]] = rec
    audit_ids = [
        line.strip()
        for line in Path(args.query_ids).read_text().splitlines()
        if line.strip()
    ]
    missing = [q for q in audit_ids if q not in queries_by_id]
    if missing:
        print(f"ERROR: missing in queries: {missing[:5]}", file=sys.stderr)
        return 1
    print(f"  audit queries: {len(audit_ids)}", flush=True)

    print(f"\n=== Embed queries ===", flush=True)
    t0 = time.time()
    Q = np.stack([embedder.embed(queries_by_id[qid]["query_text"]) for qid in audit_ids])
    print(f"  query embeddings: shape={Q.shape} ({time.time()-t0:.1f}s)", flush=True)
    Q_norm = l2_normalize(Q)

    np.save(args.queries_npy, Q)
    print(f"  saved unnormalized embeddings -> {args.queries_npy}", flush=True)

    print(f"\n=== Cosine search top-{args.top_k} ===", flush=True)
    t0 = time.time()
    scores = Q_norm @ corpus_norm.T  # (Q, N)
    k = args.top_k
    top_idx_unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores_unsorted = np.take_along_axis(scores, top_idx_unsorted, axis=1)
    sort_within = np.argsort(-top_scores_unsorted, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, sort_within, axis=1)
    top_scores = np.take_along_axis(top_scores_unsorted, sort_within, axis=1)
    print(f"  search: {time.time()-t0:.2f}s", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for qi, qid in enumerate(audit_ids):
            results = [
                {
                    "chunk_id": chunk_ids[int(idx)],
                    "rank": rank + 1,
                    "score": float(top_scores[qi, rank]),
                }
                for rank, idx in enumerate(top_idx[qi])
            ]
            f.write(json.dumps({
                "query_id": qid,
                "model": "gecko-1024-quant-v0.2.0",
                "embedding_dim": int(embedder.dim),
                "top_k": k,
                "results": results,
            }) + "\n")
    print(f"\nWrote {len(audit_ids)} records to {out_path}", flush=True)
    return 0 if sanity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
