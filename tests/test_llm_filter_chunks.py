from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import llm_filter_chunks


class LlmFilterResumeTests(unittest.TestCase):
    def test_load_done_ids_ignores_stale_prompt_or_schema_records(self) -> None:
        current = {
            "chunk_id": "a" * 16,
            "llm_filter_schema_version": llm_filter_chunks.RESULT_SCHEMA_VERSION,
            "llm_filter_prompt_hash": llm_filter_chunks.PROMPT_HASH,
            "query": "What dose of oxytocin is used?",
            "reason": "Chunk answers it and it guides care.",
            "answerable_by_chunk": True,
            "clinically_useful": True,
        }
        old_suitable_schema = {
            "chunk_id": "b" * 16,
            "suitable": True,
            "query": "What dose of oxytocin is used?",
            "reason": "old schema",
        }
        old_prompt_hash = {
            "chunk_id": "c" * 16,
            "llm_filter_schema_version": llm_filter_chunks.RESULT_SCHEMA_VERSION,
            "llm_filter_prompt_hash": "old-prompt",
            "query": "What dose of oxytocin is used?",
            "reason": "old prompt",
            "answerable_by_chunk": True,
            "clinically_useful": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [current, old_suitable_schema, old_prompt_hash]
                )
                + "\n",
                encoding="utf-8",
            )

            done_ids, stale_count = llm_filter_chunks._load_done_ids(path)

        self.assertEqual(done_ids, {"a" * 16})
        self.assertEqual(stale_count, 2)

    def test_output_contract_requires_current_prompt_metadata(self) -> None:
        current_output = {
            "chunk_id": "a" * 16,
            "llm_filter_schema_version": llm_filter_chunks.RESULT_SCHEMA_VERSION,
            "llm_filter_prompt_hash": llm_filter_chunks.PROMPT_HASH,
            "seed_query": "What dose of oxytocin is used?",
            "llm_answerable_by_chunk": True,
            "llm_clinically_useful": True,
        }
        stale_output = dict(current_output, llm_filter_prompt_hash="old-prompt")

        self.assertTrue(llm_filter_chunks._matches_current_output_contract(current_output))
        self.assertFalse(llm_filter_chunks._matches_current_output_contract(stale_output))


if __name__ == "__main__":
    unittest.main()
