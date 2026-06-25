#!/usr/bin/env python3
"""Retrieve EmbeddingGemma top-20 for the full 3,185-query benchmark.

Mirrors the deployed on-device retriever exactly so the resulting scoreboard row
is comparable to the other six retrievers in AUDIT_REPORT_v2.md.

Encoding (byte-for-byte the shipped LiteRT int8 build, see
``mamai-medical-guidelines/scripts/reembed_embeddinggemma.py``):
  query prompt "task: search result | query: ", seq-len 256, BOS+ids+EOS,
  right-pad to 256 int32, 768-dim output, L2-normalised, cosine similarity.

Corpus vectors are NOT recomputed: the deployed document vectors for the full
63,650-chunk corpus live in rag-bundle-v0.3.0/runtime/embeddings.sqlite (table
``rag_vector_store``, blob = b"VF32" + 768x float32). That store has no chunk_id
column, so we recover it the same way the corpus did:
``chunk_id = sha256(text_after_first_header_line)[:16]`` (verified to match the
benchmark CIDs).

Usage:
  .venv-gecko/bin/python scripts/retrieve_embeddinggemma_full.py \
      --sqlite ~/Downloads/mamai-medical-guidelines/releases/rag-bundle-v0.3.0/runtime/embeddings.sqlite \
      --tflite ~/Downloads/mamai-demo/assets/embeddinggemma-300M_seq256_mixed-precision.tflite \
      --tokenizer ~/Downloads/mamai-demo/assets/sentencepiece.model \
      --queries data/queries.jsonl \
      --out data/full/embeddinggemma_top20.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np

SEQ = 256
QUERY_PROMPT = "task: search result | query: "
BOS, EOS = 2, 1
VF32 = b"VF32"
MODEL_NAME = "embeddinggemma-300m-seq256-mixed-precision-litert"
EMB_DIM = 768
TOP_K = 20


def cid_from_text(text: str) -> str:
    """Recover the benchmark chunk_id from a rag_vector_store text cell.

    The stored text is ``[SOURCE:..|PAGE:..]\\n<body>``; the CID is
    ``sha256(<body>)[:16]`` (the header line is not part of the hashed text)."""
    _, _, body = text.partition("\n")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def load_corpus(sqlite_path: Path) -> tuple[list[str], np.ndarray]:
    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute(
        "SELECT text, embeddings FROM rag_vector_store ORDER BY ROWID"
    ).fetchall()
    conn.close()
    chunk_ids: list[str] = []
    mat = np.empty((len(rows), EMB_DIM), dtype=np.float32)
    for i, (text, blob) in enumerate(rows):
        if blob[:4] != VF32:
            raise ValueError(f"row {i}: unexpected blob prefix {blob[:4]!r}")
        mat[i] = struct.unpack("<768f", blob[4:])
        chunk_ids.append(cid_from_text(text))
    # Vectors are already L2-normalised on write, but re-normalise defensively.
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    dupes = len(chunk_ids) - len(set(chunk_ids))
    print(f"corpus: {len(chunk_ids)} vectors, {dupes} duplicate chunk_ids", flush=True)
    return chunk_ids, mat


def embed_queries(texts: list[str], tflite: str, tok: str) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter
    import sentencepiece as spm

    it = Interpreter(model_path=tflite, num_threads=max(1, _cpu() - 2))
    it.allocate_tensors()
    in_idx = it.get_input_details()[0]["index"]
    out_idx = it.get_output_details()[0]["index"]
    sp = spm.SentencePieceProcessor()
    sp.load(tok)

    out = np.empty((len(texts), EMB_DIM), dtype=np.float32)
    t0 = time.time()
    for i, q in enumerate(texts):
        ids = [BOS] + sp.encode_as_ids(QUERY_PROMPT + q) + [EOS]
        ids = ids[:SEQ] + [0] * max(0, SEQ - len(ids))
        it.set_tensor(in_idx, np.array([ids], dtype=np.int32))
        it.invoke()
        v = it.get_tensor(out_idx).flatten().astype(np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-9)
        if (i + 1) % 500 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"  embedded {i + 1}/{len(texts)} {r:.1f}/s", flush=True)
    print(f"embedded {len(texts)} queries in {int(time.time() - t0)}s", flush=True)
    return out


def _cpu() -> int:
    import os
    return os.cpu_count() or 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--out", default="data/full/embeddinggemma_top20.jsonl")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    queries = [json.loads(l) for l in Path(a.queries).open() if l.strip()]
    qids = [q["query_id"] for q in queries]
    qtexts = [q["query_text"] for q in queries]
    print(f"queries: {len(qids)}", flush=True)

    chunk_ids, doc_emb = load_corpus(Path(a.sqlite).expanduser())
    q_emb = embed_queries(qtexts, str(Path(a.tflite).expanduser()),
                          str(Path(a.tokenizer).expanduser()))

    chunk_ids_arr = np.array(chunk_ids)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open("w") as fh:
        # Block the query matmul to bound peak memory (3185 x 63650 fits, but be tidy).
        BLOCK = 256
        for start in range(0, len(qids), BLOCK):
            qe = q_emb[start:start + BLOCK]
            sims = qe @ doc_emb.T  # (b, N)
            # top-k via argpartition then sort the k winners by score desc
            kk = a.top_k
            part = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
            for row in range(qe.shape[0]):
                idx = part[row]
                idx = idx[np.argsort(-sims[row, idx])]
                results = [
                    {"chunk_id": str(chunk_ids_arr[j]), "rank": rank + 1,
                     "score": float(sims[row, j])}
                    for rank, j in enumerate(idx)
                ]
                rec = {"query_id": qids[start + row], "model": MODEL_NAME,
                       "embedding_dim": EMB_DIM, "top_k": kk, "results": results}
                fh.write(json.dumps(rec) + "\n")
    print(f"wrote {out_path} ({len(qids)} queries) in {int(time.time() - t0)}s",
          flush=True)


if __name__ == "__main__":
    main()
