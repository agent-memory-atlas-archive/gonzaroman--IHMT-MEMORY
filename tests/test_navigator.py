"""Tree-walking retrieval, logarithmic cost and the interactive clue loop."""

from __future__ import annotations

import unittest
from typing import List, Optional

from ihmt.semantic_navigator import ClueRequest

from .base import StoreTestCase


class DescentTests(StoreTestCase):
    """The navigator walks root -> branch -> leaf instead of scanning."""

    def setUp(self) -> None:
        super().setUp()
        for name in ("InventoryService.java", "narrative_es.txt", "journal_personal.txt", "recipe_paella.md"):
            self.ingest_example(name)
        self.memory.flush()

    def test_a_specific_query_finds_the_right_leaf(self) -> None:
        response = self.memory.search("reserveStock soft hold quantity", top_k=3)

        self.assertTrue(response.results)
        self.assertIn("reserveStock", response.best.content)
        self.assertEqual(response.best.domain, "software.java")

    def test_the_result_carries_the_path_it_was_found_through(self) -> None:
        response = self.memory.search("reserveStock soft hold quantity")
        path = response.best.path

        self.assertEqual(path[0], "root")
        self.assertEqual(path[-1], response.best.leaf_id)
        self.assertGreaterEqual(len(path), 3, "root -> branch -> leaf at minimum")

    def test_retrieval_opens_far_fewer_files_than_a_scan(self) -> None:
        stats = self.memory.stats()
        response = self.memory.search("socarrat paella reposo", top_k=3)

        total_files = stats["leaves"] + stats["nodes"] + 1
        self.assertTrue(response.results)
        self.assertLess(
            response.node_reads + response.leaf_reads,
            total_files / 2,
            "the descent must not degenerate into a scan",
        )

    def test_node_reads_stay_bounded_by_beam_times_depth(self) -> None:
        response = self.memory.search("Marta cuaderno inventario", top_k=3)
        ceiling = 1 + self.memory.config.beam_width * (self.memory.store.max_layer() + 1)

        self.assertLessEqual(response.node_reads, ceiling)

    def test_a_query_in_another_language_reaches_its_own_domain(self) -> None:
        response = self.memory.search("cómo se hace el socarrat de la paella", top_k=2)

        self.assertTrue(response.results)
        self.assertEqual(response.best.domain, "process.cooking")

    def test_an_empty_query_returns_nothing_rather_than_everything(self) -> None:
        response = self.memory.search("   ")
        self.assertEqual(response.results, [])

    def test_a_query_matching_nothing_returns_nothing(self) -> None:
        # Returning zero-scoring leaves would dress "nothing is stored about
        # this" up as either a confident answer or a pointless disambiguation.
        response = self.memory.search("zxqwv kubernetes helm unmatched")

        self.assertEqual(response.results, [])
        self.assertLess(response.confidence, self.memory.config.confidence_threshold)
        self.assertIsNone(response.clue_request)

    def test_a_query_sharing_one_stray_word_is_no_clear_match(self) -> None:
        # Only "Madrid" is stored; the other six words of the question are not.
        # Offering the Madrid entries as a choice would be a false AMBIGUOUS.
        response = self.memory.search("billete avión Madrid Tokio noviembre escala Doha")

        self.assertTrue(response.weak, f"coverage {response.coverage}")
        self.assertFalse(response.found)
        self.assertIsNone(response.clue_request)

    def test_a_specific_query_is_not_weak(self) -> None:
        response = self.memory.search("reserveStock soft hold quantity")

        self.assertFalse(response.weak)
        self.assertTrue(response.found)



class CrossLevelRankingTests(StoreTestCase):
    """Leaves found at different depths are compared on what they hold."""

    # The default branch factor: the shape in which the bug showed up, with the
    # PostgreSQL decision inside a branch and the staging note still at the top.
    branch_factor = 8
    target_tokens = 2000
    max_tokens = 3000

    NOTES = [
        "2026-01-12: Started Tidewater, a booking app for small sailing schools. Backend in Python 3.12 with FastAPI, frontend in SvelteKit, deployed on Fly.io.",
        "2026-01-12: I live in Lisbon. I work remotely, and most Tidewater customers are sailing schools in Portugal and Spain.",
        "2026-01-14: Decided on PostgreSQL over SQLite for Tidewater because bookings need row-level locking when two students grab the last seat on a boat.",
        "2026-01-20: Dev setup: macOS, uv for Python environments, pnpm for the frontend, Postgres through Docker Compose (docker compose up db). Tests run with pytest -q.",
        "2026-02-03: Prefer small pull requests (under 300 lines) and squash merges. Commit messages in English, imperative mood.",
        "2026-02-10: Luis Ortega is the backend lead at Tidewater. He reviews every database migration before it is merged.",
        "2026-02-18: Luis Fernández is the freelance designer who made the Tidewater landing page and the logo. Paid per project, invoices at the end of each month.",
        "2026-03-09: Decision: money is stored as integer cents everywhere, never floats. A rounding bug on a 3-day course price (199.99 x 3) charged a customer one cent less.",
        "2026-03-15: Release procedure for Tidewater: 1) merge to main, 2) CI builds the image, 3) fly deploy --strategy rolling to staging, 4) smoke test the booking flow, 5) promote to production with fly deploy -a tidewater-prod.",
        "2026-03-28: Timezones: all timestamps stored in UTC; schools see local time. The Lisbon school once got bookings shifted one hour on the DST change because a cron used local time.",
        "2026-04-06: Stripe webhooks must be idempotent: we store the event id in processed_events and skip duplicates. Stripe retries for up to 3 days.",
        "2026-04-19: Prefers unittest-style plain asserts in pytest, no extra plugins except pytest-asyncio.",
        "2026-05-02: The booking calendar is slow above 400 slots per week. Fixed by loading one week at a time and caching availability for 60 seconds.",
        "2026-05-21: Maria Costa runs the Porto sailing school, our first paying customer. She wants SMS reminders 24 hours before each class.",
        "2026-06-04: SMS reminders go through Twilio. Decided against WhatsApp for now: templates need approval and schools found it confusing.",
        "2026-06-30: Moved CI from GitHub Actions minutes to a self-hosted runner on a Hetzner box because the test suite with Postgres took 14 minutes.",
        "2026-07-12: Accessibility: every form field has a visible label; the booking flow was tested with VoiceOver on macOS and iOS.",
        "2026-08-27: Backups: nightly pg_dump to an S3 bucket in eu-west-1, kept 30 days. Restore tested on 2026-08-27 in 11 minutes.",
        "2026-09-08: Do not use the ORM for the availability query: the raw SQL with a lateral join is 20x faster. Keep it in queries/availability.sql.",
        "2026-09-15: The mobile app is postponed until spring 2027. The web app works well on phones and schools asked for reporting first.",
        "2026-09-29: Monthly report for schools: bookings, cancellations and revenue per instructor, exported as CSV and PDF.",
        "2026-10-02: Luis Ortega suggested partitioning the bookings table by year once it passes 5 million rows. Not needed yet (1.2 million today).",
        "2026-07-01: I moved to Porto. It puts me closer to the first schools using Tidewater, and the timezone is the same, so no cron changes.",
        "2026-03-02: Staging database runs PostgreSQL 15 on a shared Fly.io cluster.",
        "2026-08-20: Staging upgraded to PostgreSQL 17; PostgreSQL 15 is no longer used anywhere.",
    ]

    def setUp(self) -> None:
        super().setUp()
        # Saved one by one as save_memory does, without a final flush: the
        # last notes stay at the top level while older ones sit in a branch.
        for note in self.NOTES:
            self.memory.ingest_text(note, source="mcp://claude-code", tags=["source:claude-code"])

    def test_a_leaf_holding_the_whole_query_beats_one_sharing_a_word_at_another_level(self) -> None:
        response = self.memory.search("staging database PostgreSQL version", top_k=3)

        self.assertIn("Staging", response.best.content)
        self.assertNotIn("SQLite", response.best.content)


class ClueLoopTests(StoreTestCase):
    """An ambiguous query asks for a clue instead of guessing."""

    def setUp(self) -> None:
        super().setUp()
        self.ingest_example("journal_personal.txt")
        self.ingest_example("narrative_es.txt")
        self.memory.flush()

    def test_a_bare_first_name_is_reported_as_ambiguous(self) -> None:
        response = self.memory.search("Luis", top_k=4)

        self.assertTrue(response.ambiguous)
        self.assertIsNotNone(response.clue_request)
        self.assertGreaterEqual(len(response.clue_request.options), 2)
        self.assertIn("Luis", response.clue_request.prompt())

    def test_the_clue_loop_resolves_the_ambiguity(self) -> None:
        asked: List[ClueRequest] = []

        def provider(request: ClueRequest) -> str:
            asked.append(request)
            return "vacaciones en Benidorm"

        response = self.memory.ask("Luis", provider, top_k=3)

        self.assertEqual(len(asked), 1, "the user should be asked exactly once")
        self.assertFalse(response.needs_clue, "the clue must resolve the query")
        self.assertIn("Benidorm", response.best.content)
        self.assertEqual(response.clues_used, ["vacaciones en Benidorm"])

    def test_the_joint_query_beats_either_term_alone(self) -> None:
        joint = self.memory.navigator.cross_reference("Luis", "vacaciones en Benidorm")
        self.assertIn("Benidorm", joint.best.content)
        self.assertIn("+", joint.query)

    def test_declining_to_give_a_clue_ends_the_loop_honestly(self) -> None:
        def refuse(request: ClueRequest) -> Optional[str]:
            return None

        response = self.memory.ask("Luis", refuse)

        self.assertTrue(response.needs_clue, "without a clue the answer stays flagged as ambiguous")
        self.assertEqual(response.clues_used, [])

    def test_repeating_the_same_clue_does_not_loop_forever(self) -> None:
        calls = {"n": 0}

        def repeat(request: ClueRequest) -> str:
            calls["n"] += 1
            return "Luis"

        self.memory.ask("Luis", repeat, max_rounds=5)
        self.assertLessEqual(calls["n"], 2, "a clue that adds nothing must stop the loop")

    def test_outline_renders_the_hierarchy(self) -> None:
        lines = self.memory.outline(max_depth=2)
        self.assertTrue(lines[0].startswith("root.json"))
        self.assertGreater(len(lines), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
