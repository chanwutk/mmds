from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples.lecture_event_localization import build_query as build_two_stage_query  # noqa: E402
from examples.lecture_event_localization_transcript_only import (  # noqa: E402
    build_query as build_transcript_only_query,
)
from examples.lecture_event_localization_video_only import (  # noqa: E402
    build_query as build_video_only_query,
)
from mmds import StaticPromptExecutor, execute  # noqa: E402
from mmds.model import DatasetExpr, PromptSpec  # noqa: E402
from mmds.model import MMDSValidationError  # noqa: E402


def nodes_by_name(plan: DatasetExpr) -> dict[str, DatasetExpr]:
    result: dict[str, DatasetExpr] = {}
    node: DatasetExpr | None = plan
    while node is not None:
        if node.name is not None:
            result[node.name] = node
        node = node.source
    return result


def input_path(plan: DatasetExpr) -> str | None:
    node = plan
    while node.source is not None:
        node = node.source
    return node.input_path


class LectureEventLocalizationPlanTests(unittest.TestCase):
    def test_all_queries_use_generic_lecture_data_by_default(self) -> None:
        plans = [
            build_video_only_query(),
            build_transcript_only_query(),
            build_two_stage_query(),
        ]

        self.assertEqual(
            [input_path(plan) for plan in plans],
            ["data/lectures.jsonl"] * 3,
        )

    def test_plan_has_explicit_rewrite_stages(self) -> None:
        plan = build_two_stage_query("data/example.jsonl")

        names = nodes_by_name(plan)

        self.assertEqual(
            list(names),
            [
                "one_event_per_row",
                "collect_and_sort_window_events",
                "rebase_event_times",
                "verify_video_events",
                "merge_overlapping_windows",
                "pad_candidate_windows",
                "one_candidate_per_row",
                "transcript_candidates",
            ],
        )

    def test_baseline_plans_have_explicit_stages(self) -> None:
        video_names = list(nodes_by_name(build_video_only_query("data/example.jsonl")))
        transcript_names = list(nodes_by_name(build_transcript_only_query("data/example.jsonl")))

        self.assertEqual(
            video_names,
            ["one_event_per_row", "localize_full_video"],
        )
        self.assertEqual(
            transcript_names,
            ["one_event_per_row", "localize_from_transcript"],
        )

    def test_executes_transcript_gate_through_collection_and_sorting(self) -> None:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "Find the physical demonstration.",
            "video": {"type": "Video", "path": "/tmp/lecture.mp4"},
            "transcript": [{"start": 100, "end": 120, "text": "A demonstration begins."}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "lectures.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            plan = build_two_stage_query(str(input_path))
            nodes = nodes_by_name(plan)
            candidate_spec = nodes["transcript_candidates"].spec
            verification_spec = nodes["verify_video_events"].spec
            self.assertIsInstance(candidate_spec, PromptSpec)
            self.assertIsInstance(verification_spec, PromptSpec)

            seen_clip: dict = {}

            def candidate_handler(resolved_prompt, payload, context):
                self.assertEqual(payload["lecture_id"], "lecture-1")
                return {
                    "candidates": [
                        {"start": 100, "end": 120},
                        {"start": 115, "end": 125},
                    ]
                }

            def verification_handler(resolved_prompt, payload, context):
                seen_clip.update(payload["clip"])
                return {
                    "clip_events": [
                        {"start": 12, "end": 15},
                        {"start": 40, "end": 44},
                    ]
                }

            executor = StaticPromptExecutor(
                {
                    ("map", candidate_spec.cache_key()): candidate_handler,
                    ("map", verification_spec.cache_key()): verification_handler,
                }
            )
            result = execute(plan, prompt_executor=executor)

        self.assertEqual(
            seen_clip,
            {
                "type": "VideoView",
                "path": "/tmp/lecture.mp4",
                "start": 90.0,
                "end": 135.0,
            },
        )
        self.assertEqual(
            result,
            [
                {"lecture_id": "lecture-1", "events": {"start": 102.0, "end": 105.0}},
                {"lecture_id": "lecture-1", "events": {"start": 130.0, "end": 134.0}},
            ],
        )

    def test_video_only_query_uses_video_and_unnests_output(self) -> None:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "Find the physical demonstration.",
            "video": {"type": "Video", "path": "/tmp/lecture.mp4"},
            "transcript": [{"start": 100, "end": 120, "text": "A demonstration begins."}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "lectures.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            plan = build_video_only_query(str(input_path))
            spec = nodes_by_name(plan)["localize_full_video"].spec
            self.assertIsInstance(spec, PromptSpec)

            def handler(resolved_prompt, payload, context):
                self.assertEqual(resolved_prompt.parts[0], row["video"])
                return {"events": [{"start": 20, "end": 25}]}

            result = execute(
                plan,
                prompt_executor=StaticPromptExecutor({("map", spec.cache_key()): handler}),
            )

        self.assertEqual(
            result,
            [{**row, "events": {"start": 20, "end": 25}}],
        )

    def test_transcript_only_query_uses_transcript_and_unnests_output(self) -> None:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "Find the physical demonstration.",
            "video": {"type": "Video", "path": "/tmp/lecture.mp4"},
            "transcript": [{"start": 100, "end": 120, "text": "A demonstration begins."}],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "lectures.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            plan = build_transcript_only_query(str(input_path))
            spec = nodes_by_name(plan)["localize_from_transcript"].spec
            self.assertIsInstance(spec, PromptSpec)

            def handler(resolved_prompt, payload, context):
                self.assertEqual(resolved_prompt.parts[-1], row["transcript"])
                return {"events": [{"start": 110, "end": 116}]}

            result = execute(
                plan,
                prompt_executor=StaticPromptExecutor({("map", spec.cache_key()): handler}),
            )

        self.assertEqual(
            result,
            [{**row, "events": {"start": 110, "end": 116}}],
        )


class TranscriptGatedEdgeCaseTests(unittest.TestCase):
    """Pipeline behavior at the edges of the transcript-gated query."""

    def _row(self, **overrides) -> dict:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "Find the physical demonstration.",
            "video": {"type": "Video", "path": "/tmp/lecture.mp4"},
            "transcript": [{"start": 100, "end": 120, "text": "A demonstration begins."}],
        }
        row.update(overrides)
        return row

    def _run(self, rows, candidate_handler, verification_handler):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lectures.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            plan = build_two_stage_query(str(path))
            nodes = nodes_by_name(plan)
            executor = StaticPromptExecutor(
                {
                    ("map", nodes["transcript_candidates"].spec.cache_key()): candidate_handler,
                    ("map", nodes["verify_video_events"].spec.cache_key()): verification_handler,
                }
            )
            return execute(plan, prompt_executor=executor)

    def test_same_query_on_two_lectures_stays_separate(self) -> None:
        rows = [self._row(lecture_id="lecture-1"), self._row(lecture_id="lecture-2")]

        def candidate_handler(resolved_prompt, payload, context):
            return {"candidates": [{"start": 100, "end": 110}]}

        def verification_handler(resolved_prompt, payload, context):
            return {"clip_events": [{"start": 10, "end": 12}]}

        result = self._run(rows, candidate_handler, verification_handler)

        self.assertCountEqual(
            result,
            [
                {"lecture_id": "lecture-1", "events": {"start": 100.0, "end": 102.0}},
                {"lecture_id": "lecture-2", "events": {"start": 100.0, "end": 102.0}},
            ],
        )

    def test_disjoint_candidates_are_verified_as_separate_clips(self) -> None:
        seen_clips: list[tuple[float, float]] = []

        def candidate_handler(resolved_prompt, payload, context):
            return {"candidates": [{"start": 300, "end": 310}, {"start": 100, "end": 110}]}

        def verification_handler(resolved_prompt, payload, context):
            seen_clips.append((payload["clip"]["start"], payload["clip"]["end"]))
            return {"clip_events": [{"start": 5, "end": 6}]}

        result = self._run([self._row()], candidate_handler, verification_handler)

        self.assertCountEqual(seen_clips, [(90.0, 120.0), (290.0, 320.0)])
        self.assertEqual(
            result,
            [
                {"lecture_id": "lecture-1", "events": {"start": 95.0, "end": 96.0}},
                {"lecture_id": "lecture-1", "events": {"start": 295.0, "end": 296.0}},
            ],
        )

    def test_candidate_near_video_start_clamps_clip_to_zero(self) -> None:
        seen_clips: list[tuple[float, float]] = []

        def candidate_handler(resolved_prompt, payload, context):
            return {"candidates": [{"start": 3, "end": 8}]}

        def verification_handler(resolved_prompt, payload, context):
            seen_clips.append((payload["clip"]["start"], payload["clip"]["end"]))
            return {"clip_events": [{"start": 1, "end": 2}]}

        result = self._run([self._row()], candidate_handler, verification_handler)

        self.assertEqual(seen_clips, [(0, 18)])
        self.assertEqual(
            result,
            [{"lecture_id": "lecture-1", "events": {"start": 1, "end": 2}}],
        )

    def test_no_candidates_skips_video_stage_and_produces_no_rows(self) -> None:
        verification_calls: list[dict] = []

        def candidate_handler(resolved_prompt, payload, context):
            return {"candidates": []}

        def verification_handler(resolved_prompt, payload, context):
            verification_calls.append(payload)
            return {"clip_events": []}

        result = self._run([self._row()], candidate_handler, verification_handler)

        self.assertEqual(result, [])
        self.assertEqual(verification_calls, [])

    def test_no_verified_events_produces_no_rows(self) -> None:
        def candidate_handler(resolved_prompt, payload, context):
            return {"candidates": [{"start": 100, "end": 110}]}

        def verification_handler(resolved_prompt, payload, context):
            return {"clip_events": []}

        result = self._run([self._row()], candidate_handler, verification_handler)

        self.assertEqual(result, [])

    def test_reversed_candidate_interval_fails_explicitly(self) -> None:
        # Current behavior: candidates are not sanitized, so a model interval
        # with end <= start stops the run at Window instead of being dropped.
        def candidate_handler(resolved_prompt, payload, context):
            return {"candidates": [{"start": 120, "end": 100}]}

        def verification_handler(resolved_prompt, payload, context):
            return {"clip_events": []}

        with self.assertRaisesRegex(MMDSValidationError, "Window end must be greater than start"):
            self._run([self._row()], candidate_handler, verification_handler)

if __name__ == "__main__":
    unittest.main()
