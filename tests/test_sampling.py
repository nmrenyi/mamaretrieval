from __future__ import annotations

import random
import unittest

from mamaretrieval.corpus import CorpusChunk
from mamaretrieval.sampling import (
    ROOT_SECTION,
    SampledChunk,
    SourceTarget,
    boilerplate_reason,
    build_source_targets,
    sample_source_chunks,
    top_level_section,
)


def chunk(
    chunk_id: str,
    *,
    source: str = "source-a",
    breadcrumb: str = "Section > Child",
    text: str | None = None,
) -> CorpusChunk:
    body = text or (
        "## Clinical heading\n\n"
        "This chunk contains clinically useful body text about treatment, "
        "assessment, referral, and follow-up for a bedside care scenario."
    )
    return CorpusChunk(
        chunk_id=chunk_id,
        source=source,
        page=1,
        breadcrumb=breadcrumb,
        text=body,
    )


class SamplingTests(unittest.TestCase):
    def test_top_level_section_uses_first_breadcrumb_segment(self) -> None:
        self.assertEqual(
            top_level_section("Postpartum Haemorrhage > Active Management"),
            "Postpartum Haemorrhage",
        )

    def test_top_level_section_uses_root_for_missing_breadcrumb(self) -> None:
        self.assertEqual(top_level_section(""), ROOT_SECTION)

    def test_sampled_chunk_record_includes_section(self) -> None:
        sampled = SampledChunk(
            chunk("a" * 16, breadcrumb="Postpartum Haemorrhage > Treatment"),
            tier="very_high",
            section="Postpartum Haemorrhage",
        )

        record = sampled.to_record()

        self.assertEqual(record["section"], "Postpartum Haemorrhage")

    def test_boilerplate_filters_short_text(self) -> None:
        reason = boilerplate_reason(chunk("a" * 16, text="Too short."))

        self.assertEqual(reason, "short_text")

    def test_boilerplate_filters_front_matter(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "b" * 16,
                text=(
                    "## Suggested citation\n\n"
                    "This publication should be cited using the following long "
                    "bibliographic citation and publication metadata."
                ),
            )
        )

        self.assertEqual(reason, "heading_suggested_citation")

    def test_boilerplate_filters_heading_only_chunks(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "c" * 16,
                text=(
                    "# This is a very long heading that has enough characters "
                    "to pass the short-text threshold but still has no body text"
                ),
            )
        )

        self.assertEqual(reason, "heading_only")

    def test_boilerplate_filters_front_matter_breadcrumb_section(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "d" * 16,
                breadcrumb="How to use this book > Challenges",
            )
        )

        self.assertEqual(reason, "section_how_to_use_this_book")

    def test_boilerplate_filters_in_this_chapter_tables(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "e" * 16,
                text=(
                    "## In this chapter:\n\n"
                    "| Topic | Page |\n"
                    "|---|---|\n"
                    "| Labour care | 12 |\n"
                    "| Postpartum care | 20 |\n"
                    "This table previews chapter sections and page numbers."
                ),
            )
        )

        self.assertEqual(reason, "heading_in_this_chapter")

    def test_boilerplate_filters_activity_chunks(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "f" * 16,
                text=(
                    "# Activity\n\n"
                    "Find out what the local emergency number is and discuss "
                    "the answer with your classmates during the teaching session."
                ),
            )
        )

        self.assertEqual(reason, "heading_activity")

    def test_boilerplate_filters_facilitator_schedule_chunks(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "1" * 16,
                breadcrumb="Facilitator's schedule and preparation activities",
                text=(
                    "### i 4 hours and 40 minutes\n\n"
                    "| Title | Minutes | Activities | Preparations | Page |\n"
                    "|---|---|---|---|---|\n"
                    "| Group work | 30 | Discuss | Prepare handouts | 10 |"
                ),
            )
        )

        self.assertEqual(reason, "section_facilitator's_schedule_and_preparation_activities")

    def test_boilerplate_filters_glossary_fragments(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "2" * 16,
                text=(
                    "## **A**\n\n"
                    "**Anaemia** A condition where blood has too little haemoglobin.\n\n"
                    "**Antenatal** Before birth.\n\n"
                    "**Antibiotic** A drug used to treat infection."
                ),
            )
        )

        self.assertEqual(reason, "glossary_fragment")

    def test_boilerplate_filters_cataloguing_pages(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "3" * 16,
                text=(
                    "# Education material for teachers of midwifery\n\n"
                    "WHO Library Cataloguing-in-Publication Data. This page "
                    "contains publication metadata and cataloguing details."
                ),
            )
        )

        self.assertEqual(reason, "cataloguing")

    def test_boilerplate_filters_web_resources(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "4" * 16,
                text=(
                    "#### WEB RESOURCES FOR CLINICIANS\n\n"
                    "| Resource | URL |\n"
                    "|---|---|\n"
                    "| Fertility chart | www.example.org |\n"
                    "| Patient handout | www.example.net |"
                ),
            )
        )

        self.assertEqual(reason, "heading_web_resources_for_clinicians")

    def test_boilerplate_filters_empty_form_tables(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "5" * 16,
                text=(
                    "### Evaluating care: Vaginal bleed at term\n\n"
                    "| Recommendations: |\n"
                    "|------------------|\n"
                    "|                  |\n"
                    "|                  |\n"
                    "|                  |\n"
                    "|                  |\n"
                    "|                  |\n"
                    "The instructions and available resources are provided for "
                    "the mock clinical skill station, along with timing."
                ),
            )
        )

        self.assertEqual(reason, "empty_form_table")

    def test_boilerplate_keeps_contentful_tables(self) -> None:
        reason = boilerplate_reason(
            chunk(
                "6" * 16,
                text=(
                    "### Management options\n\n"
                    "| Finding | Action |\n"
                    "|---|---|\n"
                    "| Fever with uterine tenderness | Start antibiotics |\n"
                    "| Heavy bleeding after birth | Assess for postpartum haemorrhage |\n"
                    "These findings guide urgent bedside assessment and treatment."
                ),
            )
        )

        self.assertIsNone(reason)

    def test_build_source_targets_uses_ceiling_query_budget(self) -> None:
        targets = build_source_targets(
            {
                "queries": {"questions_per_chunk": 2},
                "source_tiers": {
                    "high": {
                        "queries_per_source": 165,
                        "sources": ["source-a"],
                    }
                },
            }
        )

        self.assertEqual(targets[0].chunks_to_sample, 83)

    def test_sample_source_chunks_caps_when_usable_chunks_are_fewer_than_target(
        self,
    ) -> None:
        source_chunks = [chunk("a" * 15 + str(i), text="Too short.") for i in range(2)]
        source_chunks.append(chunk("d" * 16))
        target = SourceTarget("source-a", "moderate", 100, 50)

        sampled, report = sample_source_chunks(
            source_chunks, target, random.Random(42)
        )

        self.assertEqual(len(sampled), 1)
        self.assertEqual(report.usable_chunks, 1)
        self.assertEqual(report.sampled_chunks, 1)
        self.assertEqual(report.shortfall, 49)

    def test_sample_source_chunks_is_deterministic(self) -> None:
        source_chunks = [
            chunk(f"{i:016x}", breadcrumb="Alpha > A") for i in range(20)
        ] + [
            chunk(f"{i + 20:016x}", breadcrumb="Beta > B") for i in range(20)
        ]
        target = SourceTarget("source-a", "very_high", 300, 10)

        first, first_report = sample_source_chunks(
            source_chunks, target, random.Random(42)
        )
        second, second_report = sample_source_chunks(
            source_chunks, target, random.Random(42)
        )

        self.assertEqual(
            [item.chunk.chunk_id for item in first],
            [item.chunk.chunk_id for item in second],
        )
        self.assertEqual(first_report.sampled_chunks, second_report.sampled_chunks)
        self.assertEqual(len(first), 10)


if __name__ == "__main__":
    unittest.main()
