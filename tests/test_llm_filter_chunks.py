from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from scripts import llm_filter_chunks


class LlmFilterResumeTests(unittest.TestCase):
    def test_load_done_ids_ignores_stale_prompt_or_schema_records(self) -> None:
        current = {
            "chunk_id": "a" * 16,
            "llm_filter_schema_version": llm_filter_chunks.RESULT_SCHEMA_VERSION,
            "llm_filter_prompt_hash": llm_filter_chunks.PROMPT_HASH,
            "llm_backend": "ollama",
            "llm_model": "qwen3.5:9b",
            "query": "What dose of oxytocin is used?",
            "reason": "Chunk gives oxytocin dose and route for postpartum haemorrhage prevention.",
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
            "llm_backend": "ollama",
            "llm_model": "qwen3.5:9b",
            "query": "What dose of oxytocin is used?",
            "reason": "old prompt",
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

            done_ids, stale_count = llm_filter_chunks._load_done_ids(
                path, "ollama", "qwen3.5:9b"
            )

        self.assertEqual(done_ids, {"a" * 16})
        self.assertEqual(stale_count, 2)

    def test_load_done_ids_ignores_different_backend_or_model(self) -> None:
        current = {
            "chunk_id": "a" * 16,
            "llm_filter_schema_version": llm_filter_chunks.RESULT_SCHEMA_VERSION,
            "llm_filter_prompt_hash": llm_filter_chunks.PROMPT_HASH,
            "llm_backend": "openai",
            "llm_model": "Qwen/Qwen3.6-27B-FP8",
            "query": "What dose of oxytocin is used?",
            "reason": "Chunk gives oxytocin dose and route for postpartum haemorrhage prevention.",
        }
        stale_model = dict(current, chunk_id="b" * 16, llm_model="qwen3.5:9b")
        stale_backend = dict(current, chunk_id="c" * 16, llm_backend="ollama")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [current, stale_model, stale_backend]
                )
                + "\n",
                encoding="utf-8",
            )

            done_ids, stale_count = llm_filter_chunks._load_done_ids(
                path, "openai", "Qwen/Qwen3.6-27B-FP8"
            )

        self.assertEqual(done_ids, {"a" * 16})
        self.assertEqual(stale_count, 2)

    def test_output_contract_requires_current_prompt_metadata(self) -> None:
        current_output = {
            "chunk_id": "a" * 16,
            "llm_filter_schema_version": llm_filter_chunks.RESULT_SCHEMA_VERSION,
            "llm_filter_prompt_hash": llm_filter_chunks.PROMPT_HASH,
            "llm_backend": "openai",
            "llm_model": "Qwen/Qwen3.6-27B-FP8",
            "seed_query": "What dose of oxytocin is used?",
            "llm_filter_reason": "Chunk gives oxytocin dose and route for postpartum haemorrhage.",
        }
        stale_output = dict(current_output, llm_filter_prompt_hash="old-prompt")
        stale_model = dict(current_output, llm_model="qwen3.5:9b")

        self.assertTrue(
            llm_filter_chunks._matches_current_output_contract(
                current_output, "openai", "Qwen/Qwen3.6-27B-FP8"
            )
        )
        self.assertFalse(
            llm_filter_chunks._matches_current_output_contract(
                stale_output, "openai", "Qwen/Qwen3.6-27B-FP8"
            )
        )
        self.assertFalse(
            llm_filter_chunks._matches_current_output_contract(
                stale_model, "openai", "Qwen/Qwen3.6-27B-FP8"
            )
        )

    def test_openai_chat_url_accepts_base_or_full_endpoint(self) -> None:
        self.assertEqual(
            llm_filter_chunks._openai_chat_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )

    def test_call_openai_parses_vllm_chat_completion(self) -> None:
        chunk = {
            "chunk_id": "a" * 16,
            "breadcrumb": "Guide > Topic",
            "text": "Give oxytocin 10 IU IM after birth to prevent postpartum haemorrhage.",
        }
        response = {
            "choices": [
                {
                    "message": {
                        "reasoning": "The chunk states the drug, dose, route, and indication.",
                        "content": json.dumps(
                            {
                                "query": "What oxytocin dose prevents postpartum haemorrhage after birth?",
                                "reason": "Chunk gives dose, route, and indication; clinically relevant.",
                            }
                        ),
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(response).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = llm_filter_chunks._call_openai(
                chunk,
                "Qwen/Qwen3.6-27B-FP8",
                "http://127.0.0.1:8000/v1",
                "EMPTY",
                30,
                8192,
                0.6,
                True,
            )

        self.assertEqual(result["llm_backend"], "openai")
        self.assertEqual(result["llm_model"], "Qwen/Qwen3.6-27B-FP8")
        self.assertEqual(
            result["query"],
            "What oxytocin dose prevents postpartum haemorrhage after birth?",
        )
        self.assertIn("dose", result["reason"])
        self.assertEqual(
            llm_filter_chunks._openai_chat_url(
                "http://127.0.0.1:8000/v1/chat/completions"
            ),
            "http://127.0.0.1:8000/v1/chat/completions",
        )

    def test_call_openai_surfaces_http_error_body(self) -> None:
        chunk = {
            "chunk_id": "a" * 16,
            "breadcrumb": "Guide > Topic",
            "text": "Give oxytocin 10 IU IM after birth.",
        }
        http_error = urllib.error.HTTPError(
            url="http://127.0.0.1:8000/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error":"maximum context length exceeded"}'),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with self.assertRaises(RuntimeError) as ctx:
                llm_filter_chunks._call_openai(
                    chunk,
                    "Qwen/Qwen3.6-27B-FP8",
                    "http://127.0.0.1:8000/v1",
                    "EMPTY",
                    30,
                    32768,
                    0.6,
                    True,
                )

        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertIn("maximum context length exceeded", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
