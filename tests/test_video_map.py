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

from mmds import (  # noqa: E402
    ForEachPrompt,
    Input,
    MMDSValidationError,
    PromptSpec,
    Record,
    StaticPromptExecutor,
    VideoMap,
    VideoMapEach,
    VideoMapSpec,
    execute,
    parse_query,
    render_query,
)
from mmds.optimizers.lowering import lower_video_ops  # noqa: E402
from udfs.test_ops import annotate  # noqa: E402


def _video_prompt():
    return [
        "Inspect ",
        Record["video"],
        " and answer ",
        Record["question"],
    ]


def _video_map_each(**overrides):
    values = {
        "video_field": "video",
        "views_field": "candidate_views",
        "group_by": ["lecture_id", "question"],
        "schema": {"answer": "string"},
        "padding_time": 5,
        "clip_field": "clip",
        "name": "verify_candidates",
    }
    values.update(overrides)
    return VideoMapEach(Input("lectures.jsonl"), _video_prompt(), **values)


def _write_rows(rows: list[dict]) -> tuple[tempfile.TemporaryDirectory, str]:
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(temp_dir.name) / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return temp_dir, str(path)


class VideoMapDSLTests(unittest.TestCase):
    def test_constructs_logical_video_map_each(self) -> None:
        plan = _video_map_each()

        self.assertEqual(plan.kind, "video_map_each")
        self.assertEqual(plan.name, "verify_candidates")
        self.assertIsInstance(plan.spec, VideoMapSpec)
        self.assertEqual(plan.spec.group_by, ("lecture_id", "question"))
        self.assertEqual(plan.spec.padding_time, 5.0)

    def test_joint_video_map_requires_a_prompt(self) -> None:
        with self.assertRaisesRegex(TypeError, "prompt-backed"):
            VideoMap(
                Input("lectures.jsonl"),
                annotate,  # type: ignore[arg-type]
                video_field="video",
                views_field="candidate_views",
                group_by="lecture_id",
                schema=None,  # type: ignore[arg-type]
            )

    def test_video_map_each_accepts_a_udf(self) -> None:
        plan = VideoMapEach(
            Input("lectures.jsonl"),
            annotate,
            video_field="video",
            views_field="candidate_views",
            group_by="lecture_id",
        )

        self.assertEqual(plan.kind, "video_map_each")

    def test_referenced_context_fields_must_be_grouped(self) -> None:
        with self.assertRaisesRegex(MMDSValidationError, "question"):
            _video_map_each(group_by="lecture_id")

    def test_prompt_must_reference_the_complete_video_field(self) -> None:
        invalid_prompts = (
            ["Answer ", Record["question"]],
            ["Inspect ", Record["video"]["source"], Record["question"]],
        )
        for prompt in invalid_prompts:
            with self.subTest(prompt=prompt):
                with self.assertRaises(MMDSValidationError):
                    VideoMapEach(
                        Input("lectures.jsonl"),
                        prompt,
                        video_field="video",
                        views_field="candidate_views",
                        group_by=["lecture_id", "question"],
                        schema={"answer": "string"},
                    )

    def test_views_and_clip_fields_cannot_be_grouping_keys(self) -> None:
        for group_by in (
            ["lecture_id", "question", "candidate_views"],
            ["lecture_id", "question", "clip"],
        ):
            with self.subTest(group_by=group_by):
                with self.assertRaises(MMDSValidationError):
                    _video_map_each(group_by=group_by)

    def test_grouping_fields_must_be_non_empty_and_unique(self) -> None:
        for group_by in (
            ["lecture_id", "question", ""],
            ["lecture_id", "question", "lecture_id"],
        ):
            with self.subTest(group_by=group_by):
                with self.assertRaises(MMDSValidationError):
                    _video_map_each(group_by=group_by)

    def test_field_names_must_be_non_empty_and_distinct(self) -> None:
        cases = (
            {"video_field": ""},
            {"views_field": ""},
            {"clip_field": ""},
            {"views_field": "video"},
            {"clip_field": "video"},
            {"clip_field": "candidate_views"},
        )
        for updates in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(MMDSValidationError):
                    _video_map_each(**updates)

    def test_padding_configuration_is_validated(self) -> None:
        cases = (
            {"padding_time": -1},
            {"padding_time": float("inf")},
            {"padding_time": float("nan")},
            {"padding_time": True},
        )
        for updates in cases:
            with self.subTest(updates=updates):
                with self.assertRaises(MMDSValidationError):
                    _video_map_each(**updates)


class VideoMapRoundTripTests(unittest.TestCase):
    def test_prompt_backed_operators_round_trip(self) -> None:
        for operator in ("VideoMap", "VideoMapEach"):
            with self.subTest(operator=operator):
                program = parse_query(
                    f'''
from mmds import Input, {operator}, Record
rows = Input("lectures.jsonl")
output = {operator}(
    rows,
    ["Inspect ", Record["video"], " for ", Record["question"]],
    video_field="video",
    views_field="candidate_views",
    group_by=["lecture_id", "question"],
    schema={{"answer": "string"}},
    padding_time=5,
    clip_field="clip",
    name="verify",
)
'''
                )

                rendered = render_query(program)
                reparsed = parse_query(rendered)

                self.assertEqual(reparsed.output_expr, program.output_expr)
                self.assertIn(operator, rendered.splitlines()[0])

    def test_udf_backed_video_map_each_round_trips(self) -> None:
        program = parse_query(
            '''
from mmds import Input, VideoMapEach
from udfs.test_ops import annotate
rows = Input("lectures.jsonl")
output = VideoMapEach(
    rows,
    annotate,
    video_field="video",
    views_field="candidate_views",
    group_by="lecture_id",
)
'''
        )

        rendered = render_query(program)

        self.assertEqual(parse_query(rendered).output_expr, program.output_expr)
        self.assertIn("from udfs.test_ops import annotate", rendered)

    def test_required_keywords_and_literal_types_are_validated(self) -> None:
        invalid_queries = (
            '''
from mmds import Input, VideoMap
rows = Input("lectures.jsonl")
output = VideoMap(rows, "answer", views_field="views", group_by="id", schema={"answer": "string"})
''',
            '''
from mmds import Input, VideoMap
rows = Input("lectures.jsonl")
output = VideoMap(rows, "answer", video_field="video", views_field="views", group_by="id", schema={"answer": "string"}, padding_time="five")
''',
        )
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises(MMDSValidationError):
                    parse_query(query)


class VideoMapLoweringTests(unittest.TestCase):
    def test_per_view_lowering_builds_the_physical_pipeline(self) -> None:
        logical = _video_map_each()

        lowered = lower_video_ops(logical)
        nodes = list(lowered.walk_postorder())

        self.assertEqual(
            [node.kind for node in nodes],
            ["input", "unnest", "window", "coalesce", "map"],
        )
        self.assertEqual(
            [node.name for node in nodes[1:]],
            [
                "verify_candidates_views",
                "verify_candidates_window",
                "verify_candidates_coalesce",
                "verify_candidates",
            ],
        )
        self.assertEqual(logical.kind, "video_map_each")
        self.assertEqual(logical.source.kind, "input")
        self.assertIsInstance(lowered.spec, PromptSpec)
        self.assertIn(Record["clip"], lowered.spec.parts)
        self.assertNotIn(Record["video"], lowered.spec.parts)

    def test_joint_lowering_uses_one_reduce(self) -> None:
        logical = VideoMap(
            Input("lectures.jsonl"),
            _video_prompt(),
            video_field="video",
            views_field="candidate_views",
            group_by=["lecture_id", "question"],
            schema={"answer": "string"},
        )

        lowered = lower_video_ops(logical)

        self.assertEqual(lowered.kind, "reduce")
        self.assertEqual(lowered.group_by, ("lecture_id", "question"))
        self.assertTrue(
            any(isinstance(part, ForEachPrompt) for part in lowered.spec.parts)
        )


class VideoMapExecutionTests(unittest.TestCase):
    def test_per_view_execution_pads_coalesces_and_processes_every_view(self) -> None:
        rows = [
            {
                "lecture_id": "lecture-1",
                "question": "What happened?",
                "video": {"type": "Video", "source": "lecture.mp4"},
                "candidate_views": [
                    {"start": 10, "end": 20},
                    {"start": 18, "end": 25},
                    {"start": 50, "end": 70},
                ],
            }
        ]
        temp_dir, path = _write_rows(rows)
        self.addCleanup(temp_dir.cleanup)
        plan = VideoMapEach(
            Input(path),
            _video_prompt(),
            video_field="video",
            views_field="candidate_views",
            group_by=["lecture_id", "question"],
            schema={"answer": "string"},
            padding_time=5,
        )
        lowered = lower_video_ops(plan)
        seen: list[tuple[dict, str]] = []

        def answer(resolved, payload, context):
            seen.append((dict(resolved.parts[1]), resolved.parts[3]))
            return {"answer": "found"}

        result = execute(
            plan,
            prompt_executor=StaticPromptExecutor(
                {("map", lowered.spec.cache_key()): answer}
            ),
        )

        self.assertEqual(
            [(clip["start"], clip["end"]) for clip, _ in seen],
            [(5.0, 30.0), (45.0, 75.0)],
        )
        self.assertEqual([question for _, question in seen], ["What happened?"] * 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            set(result[0]),
            {"lecture_id", "question", "clip", "answer"},
        )

    def test_joint_execution_calls_the_prompt_once_for_all_views(self) -> None:
        rows = [
            {
                "lecture_id": "lecture-1",
                "question": "What happened?",
                "video": {"type": "Video", "source": "lecture.mp4"},
                "candidate_views": [
                    {"start": 10, "end": 20},
                    {"start": 40, "end": 50},
                ],
            }
        ]
        temp_dir, path = _write_rows(rows)
        self.addCleanup(temp_dir.cleanup)
        plan = VideoMap(
            Input(path),
            _video_prompt(),
            video_field="video",
            views_field="candidate_views",
            group_by=["lecture_id", "question"],
            schema={"answer": "string"},
        )
        lowered = lower_video_ops(plan)
        calls: list[tuple[object, ...]] = []

        def answer(resolved, payload, context):
            calls.append(resolved.parts)
            return {"answer": "done"}

        result = execute(
            plan,
            prompt_executor=StaticPromptExecutor(
                {("reduce", lowered.spec.cache_key()): answer}
            ),
        )

        views = [
            part
            for part in calls[0]
            if isinstance(part, dict) and part.get("type") == "VideoView"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [(view["start"], view["end"]) for view in views],
            [(10.0, 20.0), (40.0, 50.0)],
        )
        self.assertEqual(
            result,
            [
                {
                    "lecture_id": "lecture-1",
                    "question": "What happened?",
                    "answer": "done",
                }
            ],
        )

    def test_empty_candidate_list_produces_no_prompt_calls(self) -> None:
        temp_dir, path = _write_rows(
            [
                {
                    "lecture_id": "lecture-1",
                    "question": "What happened?",
                    "video": {"type": "Video", "source": "lecture.mp4"},
                    "candidate_views": [],
                }
            ]
        )
        self.addCleanup(temp_dir.cleanup)
        plan = VideoMapEach(
            Input(path),
            _video_prompt(),
            video_field="video",
            views_field="candidate_views",
            group_by=["lecture_id", "question"],
            schema={"answer": "string"},
        )

        self.assertEqual(execute(plan, prompt_executor=StaticPromptExecutor({})), [])

if __name__ == "__main__":
    unittest.main()
