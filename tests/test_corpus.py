from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mamaretrieval.corpus import CorpusFormatError, iter_chunks, parse_header, split_breadcrumb


class CorpusParserTests(unittest.TestCase):
    def test_parse_header(self) -> None:
        header = parse_header("<sep>[SOURCE:source-a|PAGE:12|CID:abcdef1234567890]")

        self.assertEqual(header.source, "source-a")
        self.assertEqual(header.page, 12)
        self.assertEqual(header.chunk_id, "abcdef1234567890")

    def test_parse_header_rejects_malformed_line(self) -> None:
        with self.assertRaises(CorpusFormatError):
            parse_header("<sep>[SOURCE:source-a|PAGE:12]", line_number=7)

    def test_split_breadcrumb_when_present(self) -> None:
        breadcrumb, text = split_breadcrumb(
            [
                "",
                "> Parent > Child",
                "",
                "## Heading",
                "",
                "Body text.",
                "",
            ]
        )

        self.assertEqual(breadcrumb, "Parent > Child")
        self.assertEqual(text, "## Heading\n\nBody text.")

    def test_split_breadcrumb_when_absent(self) -> None:
        breadcrumb, text = split_breadcrumb(["", "# Heading", "", "Body text.", ""])

        self.assertEqual(breadcrumb, "")
        self.assertEqual(text, "# Heading\n\nBody text.")

    def test_split_multiple_leading_breadcrumb_lines(self) -> None:
        breadcrumb, text = split_breadcrumb(["> Parent", "> Child", "", "Body text."])

        self.assertEqual(breadcrumb, "Parent Child")
        self.assertEqual(text, "Body text.")

    def test_iter_chunks_parses_breadcrumb_and_plain_chunks(self) -> None:
        corpus = "\n".join(
            [
                "<sep>[SOURCE:source-a|PAGE:1|CID:aaaaaaaaaaaaaaaa]",
                "> Topic > Subtopic",
                "",
                "First body.",
                "<sep>[SOURCE:source-b|PAGE:2|CID:bbbbbbbbbbbbbbbb]",
                "# Heading",
                "",
                "Second body.",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunks.txt"
            path.write_text(corpus, encoding="utf-8")
            chunks = list(iter_chunks(path))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source, "source-a")
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[0].chunk_id, "aaaaaaaaaaaaaaaa")
        self.assertEqual(chunks[0].breadcrumb, "Topic > Subtopic")
        self.assertEqual(chunks[0].text, "First body.")
        self.assertEqual(chunks[1].source, "source-b")
        self.assertEqual(chunks[1].breadcrumb, "")
        self.assertEqual(chunks[1].text, "# Heading\n\nSecond body.")

    def test_iter_chunks_rejects_content_before_first_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunks.txt"
            path.write_text("unexpected text\n", encoding="utf-8")

            with self.assertRaises(CorpusFormatError):
                list(iter_chunks(path))


if __name__ == "__main__":
    unittest.main()

