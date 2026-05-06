# mamaretrieval

`mamaretrieval` builds a retrieval benchmark for the MAMAI medical RAG system.
The final artifact is a set of clinical queries paired with relevant guideline
chunk IDs, used to evaluate retrievers and to provide oracle context for
generator faithfulness checks.

The benchmark is built from the production guideline corpus in:

```text
~/Downloads/mamai-medical-guidelines/processed/chunks_for_rag.txt
```

The current implementation follows:

- `IMPLEMENTATION_GUIDE.md` for the detailed benchmark specification
- `IMPLEMENTATION_PLAN.md` for staged implementation checkpoints

## Current Status

Stages 1-2 are scaffolded. The repository has project metadata,
configuration, dependency declarations, a reusable corpus parser, and a corpus
inspection script.

## Planned Pipeline

```bash
python scripts/inspect_corpus.py
python scripts/sample_chunks.py
python scripts/generate_queries.py
python scripts/pool_candidates.py
python scripts/judge_relevance.py
python scripts/audit.py
```

Pipeline outputs are written under `data/`. Release artifacts are written under
`releases/` after audit.

## Corpus Contract

The expected corpus is the `rag-bundle-v0.2.0` guideline bundle:

- `63,650` chunks
- `87` sources
- headers formatted as `<sep>[SOURCE:<source>|PAGE:<page>|CID:<chunk_id>]`

The guideline repository is read-only for this project.
