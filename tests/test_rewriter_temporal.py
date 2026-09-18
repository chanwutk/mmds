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
    Input,
    JointTemporalPushdown,
    MMDSRewriteError,
    PerViewTemporalPushdown,
    PromptSpec,
    Record,
    StaticPromptExecutor,
    StaticRewriteAgent,
    VideoMapEach,
    execute,
    parse_query,
    render_query,
    rewrite_once,
)
from mmds.model import DatasetExpr, VideoMapSpec  # noqa: E402
from mmds.optimizers.lowering import lower_video_ops  # noqa: E402


INTERVAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
        },
        "required": ["start", "end"],
    },
}


def _program(path: str = "lectures.jsonl"):
    return parse_query(
        f'''
from mmds import Input, Map, Record
lectures = Input("{path}")
output = Map(
    lectures,
    [Record["video"], "Find ", Record["query_text"]],
    schema={{"events": {INTERVAL_SCHEMA!r}}},
)
'''
    )


def _params(**updates):
    values = {
        "video_field": "video",
        "transcript_field": "transcript",
        "query_field": "query_text",
        "candidate_prompt": "Find broad transcript-supported candidate intervals.",
    }
    values.update(updates)
    return values


def _rewrite(program, directive, **updates):
    return rewrite_once(
        program,
        agent=StaticRewriteAgent(
            directive=directive.metadata.name,
            match=f"{directive.metadata.name}:output",
            params=_params(**updates),
        ),
        directives=[directive],
    )


def _prompt_node(plan: DatasetExpr, name: str) -> DatasetExpr:
    for node in plan.walk_postorder():
        if node.name == name and isinstance(node.spec, PromptSpec):
            return node
    raise AssertionError(f"No prompt node named {name!r}")


class TemporalDirectivePlanTests(unittest.TestCase):
    def test_joint_directive_builds_logical_video_map(self) -> None:
        result = _rewrite(
            _program(),
            JointTemporalPushdown(
                identity_fields="lecture_id",
                padding_seconds=10,
                max_views=4,
                max_total_video_seconds=120,
            ),
        )
        plan = result.program.output_expr

        self.assertEqual(plan.kind, "video_map")
        self.assertIsInstance(plan.spec, VideoMapSpec)
        self.assertEqual(plan.spec.views_field, "_mmds_candidate_views")
        self.assertEqual(plan.spec.group_by, ("lecture_id", "query_text"))
        self.assertEqual(plan.source.kind, "map")
        self.assertEqual(
            set(plan.source.spec.output_schema),
            {"_mmds_candidate_views"},
        )

    def test_per_view_directive_builds_rebase_and_reconciliation(self) -> None:
        result = _rewrite(
            _program(),
            PerViewTemporalPushdown(identity_fields="lecture_id"),
        )
        kinds = [node.kind for node in result.program.output_expr.walk_postorder()]
        names = [node.name for node in result.program.output_expr.walk_postorder()]

        self.assertEqual(
            kinds,
            ["input", "map", "video_map_each", "map", "reduce"],
        )
        self.assertIn("rewrite_transcript_candidates", names)
        self.assertIn("rewrite_verify_views", names)
        self.assertIn("rewrite_rebase_events", names)

    def test_temporal_rewrite_round_trips(self) -> None:
        for directive in (JointTemporalPushdown(), PerViewTemporalPushdown()):
            with self.subTest(directive=directive.metadata.name):
                configured = type(directive)(identity_fields="lecture_id")
                result = _rewrite(_program(), configured)
                rendered = render_query(result.program)
                reparsed = parse_query(rendered)
                self.assertEqual(reparsed.output_expr, result.program.output_expr)

    def test_group_by_is_derived_from_identity_and_prompt_fields(self) -> None:
        result = _rewrite(
            _program(),
            JointTemporalPushdown(identity_fields="lecture_id"),
        )

        self.assertEqual(
            result.program.output_expr.spec.group_by,
            ("lecture_id", "query_text"),
        )

    def test_group_by_is_derived_from_ancestor_reduce(self) -> None:
        program = parse_query(
            f'''
from mmds import Input, Map, Record, Reduce
from udfs.temporal_ops import reconcile_events
rows = Input("data.jsonl")
localized = Map(
    rows,
    [Record["video"], Record["query_text"]],
    schema={{"events": {INTERVAL_SCHEMA!r}}},
)
output = Reduce(localized, "lecture_id", reconcile_events)
'''
        )
        directive = PerViewTemporalPushdown()
        result = rewrite_once(
            program,
            agent=StaticRewriteAgent(
                directive=directive.metadata.name,
                match=f"{directive.metadata.name}:output.source",
                params=_params(),
            ),
            directives=[directive],
        )
        localized = result.program.output_expr.source

        self.assertEqual(localized.group_by, ("lecture_id", "query_text"))

    def test_requires_a_derivable_grouping_field(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("data.jsonl")
output = Map(rows, [Record["video"]], schema={"answer": "string"})
'''
        )
        directive = JointTemporalPushdown()

        with self.assertRaisesRegex(MMDSRewriteError, "identity_fields"):
            rewrite_once(
                program,
                agent=StaticRewriteAgent(
                    directive=directive.metadata.name,
                    match=f"{directive.metadata.name}:output",
                    params={
                        "video_field": "video",
                        "transcript_field": "transcript",
                        "query_field": "query_text",
                        "candidate_prompt": "Find candidates.",
                    },
                ),
                directives=[directive],
            )

    def test_directive_configuration_is_validated(self) -> None:
        invalid_settings = (
            {"identity_fields": ["lecture_id", "lecture_id"]},
            {"padding_seconds": float("inf")},
            {"max_views": 0},
            {"max_total_video_seconds": 0},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises((TypeError, ValueError)):
                    JointTemporalPushdown(**settings)

    def test_per_view_requires_events_output(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("data.jsonl")
output = Map(rows, [Record["video"], Record["query_text"]], schema={"answer": "string"})
'''
        )

        with self.assertRaisesRegex(MMDSRewriteError, "events"):
            _rewrite(
                program,
                PerViewTemporalPushdown(identity_fields="lecture_id"),
            )


class TemporalDirectiveExecutionTests(unittest.TestCase):
    def test_joint_rewrite_calls_model_once_with_all_selected_views(self) -> None:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "What happened?",
            "video": {"type": "Video", "source": "lecture.mp4"},
            "transcript": [{"start": 10, "end": 60, "text": "Relevant"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lectures.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = _rewrite(
                _program(str(path)),
                JointTemporalPushdown(identity_fields="lecture_id"),
            )
            lowered = lower_video_ops(result.program.output_expr)
            candidate = _prompt_node(lowered, "rewrite_transcript_candidates")
            self.assertEqual(lowered.kind, "reduce")
            self.assertIsInstance(lowered.spec, PromptSpec)
            calls: list[list[dict]] = []

            def answer(resolved, payload, context):
                views = [
                    part
                    for part in resolved.parts
                    if isinstance(part, dict) and part.get("type") == "VideoView"
                ]
                calls.append(views)
                return {"events": [{"start": 10, "end": 12}]}

            executor = StaticPromptExecutor(
                {
                    ("map", candidate.spec.cache_key()): {
                        "_mmds_candidate_views": [
                            {"start": 10, "end": 20},
                            {"start": 40, "end": 50},
                        ]
                    },
                    ("reduce", lowered.spec.cache_key()): answer,
                }
            )
            rows = execute(result.program, prompt_executor=executor)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [(view["start"], view["end"]) for view in calls[0]],
            [(10.0, 20.0), (40.0, 50.0)],
        )
        self.assertEqual(
            rows,
            [
                {
                    "lecture_id": "lecture-1",
                    "query_text": "What happened?",
                    "events": [{"start": 10, "end": 12}],
                }
            ],
        )

    def test_per_view_rewrite_executes_with_absolute_event_times(self) -> None:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "Find the demonstration.",
            "video": {"type": "Video", "source": "lecture.mp4"},
            "transcript": [{"start": 100, "end": 120, "text": "Demo"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lectures.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = _rewrite(
                _program(str(path)),
                PerViewTemporalPushdown(
                    identity_fields="lecture_id",
                    padding_seconds=10,
                ),
            )
            lowered = lower_video_ops(result.program.output_expr)
            candidate = _prompt_node(lowered, "rewrite_transcript_candidates")
            verifier = _prompt_node(lowered, "rewrite_verify_views")
            seen_clips: list[dict] = []

            def verify(resolved, payload, context):
                seen_clips.append(dict(payload["clip"]))
                return {"clip_events": [{"start": 12, "end": 15}]}

            executor = StaticPromptExecutor(
                {
                    ("map", candidate.spec.cache_key()): {
                        "_mmds_candidate_views": [
                            {"start": 100, "end": 120},
                            {"start": 115, "end": 125},
                        ]
                    },
                    ("map", verifier.spec.cache_key()): verify,
                }
            )
            rows = execute(result.program, prompt_executor=executor)

        self.assertEqual(
            seen_clips,
            [
                {
                    "type": "VideoView",
                    "source": "lecture.mp4",
                    "start": 90.0,
                    "end": 135.0,
                }
            ],
        )
        self.assertEqual(
            rows,
            [
                {
                    "lecture_id": "lecture-1",
                    "query_text": "Find the demonstration.",
                    "events": [{"start": 102.0, "end": 105.0}],
                }
            ],
        )

    def test_video_map_each_enforces_view_count_and_duration_budgets(self) -> None:
        row = {
            "lecture_id": "lecture-1",
            "query_text": "Find events.",
            "video": {"type": "Video", "source": "lecture.mp4"},
            "views": [
                {"start": 0, "end": 10},
                {"start": 20, "end": 40},
                {"start": 50, "end": 60},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            plan = VideoMapEach(
                Input(str(path)),
                [Record["video"]],
                video_field="video",
                views_field="views",
                group_by=["lecture_id", "query_text"],
                schema={"answer": "string"},
                max_views=2,
                max_total_video_seconds=15,
            )
            lowered = lower_video_ops(plan)
            prompt_node = next(
                node
                for node in lowered.walk_postorder()
                if node.kind == "map" and isinstance(node.spec, PromptSpec)
            )
            clips: list[dict] = []

            def handler(resolved, payload, context):
                clips.append(dict(payload["clip"]))
                return {"answer": "found"}

            rows = execute(
                plan,
                prompt_executor=StaticPromptExecutor(
                    {("map", prompt_node.spec.cache_key()): handler}
                ),
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [(clip["start"], clip["end"]) for clip in clips],
            [(0.0, 10.0), (20.0, 25.0)],
        )


if __name__ == "__main__":
    unittest.main()
