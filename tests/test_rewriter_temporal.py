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
    PromptSpec,
    StaticPromptExecutor,
    UdfSpec,
    VideoMapSpec,
    execute,
    parse_query,
)
from mmds.optimizers.lowering import lower_video_ops  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    JointTemporalPushdown,
    MMDSRewriteError,
    PerViewTemporalPushdown,
    PlanIndex,
    apply_rewrite,
)


EVENTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "start": {"type": "number"},
            "end": {"type": "number"},
        },
        "required": ["type", "start", "end"],
    },
}

PARAMS = {
    "video_field": "video",
    "transcript_field": "transcript",
    "query_field": "query",
    "candidate_prompt": "Find high-recall source-time intervals relevant to the query.",
    "video_prompt": (
        "Inspect this selected video view for every event matching the query. "
        "Return clip-relative event times."
    ),
}


def _event_program(path: str = "lectures.jsonl"):
    return parse_query(
        f'''
from mmds import Input, Map, Record
rows = Input({path!r})
output = Map(
    rows,
    ["Find every event matching ", Record["query"], " in ", Record["video"]],
    schema={{"events": {EVENTS_SCHEMA!r}}},
    name="localize_events",
)
'''
    )


def _answer_program(path: str):
    return parse_query(
        f'''
from mmds import Input, Map, Record
rows = Input({path!r})
output = Map(
    rows,
    ["Answer ", Record["query"], " using ", Record["video"]],
    schema={{"answer": "string"}},
    name="answer_question",
)
'''
    )


def _rewrite(program, directive):
    matches = directive.find_matches(PlanIndex.build(program.output_expr))
    return apply_rewrite(
        program,
        directive=directive,
        match=matches[0],
        params=PARAMS,
    )


def _write_rows(rows: list[dict]):
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(temp_dir.name) / "lectures.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return temp_dir, str(path)


class TemporalDirectivePlanTests(unittest.TestCase):
    def test_joint_pushdown_builds_candidate_map_then_logical_video_map(self) -> None:
        original = _event_program()
        rewritten = _rewrite(
            original,
            JointTemporalPushdown(
                identity_fields="lecture_id",
                padding_seconds=5,
            ),
        )

        video_map = rewritten.output_expr
        candidates = video_map.source
        self.assertEqual(video_map.kind, "video_map")
        self.assertIsInstance(video_map.spec, VideoMapSpec)
        self.assertEqual(video_map.spec.video_field, "video")
        self.assertEqual(video_map.spec.views_field, "_mmds_candidate_views")
        self.assertEqual(video_map.spec.group_by, ("lecture_id", "query"))
        self.assertEqual(video_map.spec.padding_time, 5.0)
        self.assertEqual(
            video_map.spec.map_spec.output_schema,
            original.output_expr.spec.output_schema,
        )
        self.assertEqual(
            video_map.spec.map_spec.parts[0],
            PARAMS["video_prompt"],
        )
        self.assertEqual(candidates.kind, "map")
        self.assertIsInstance(candidates.spec, PromptSpec)
        self.assertEqual(
            set(candidates.spec.output_schema),
            {"_mmds_candidate_views"},
        )

    def test_per_view_pushdown_rebases_then_collects_without_merging(self) -> None:
        rewritten = _rewrite(
            _event_program(),
            PerViewTemporalPushdown(identity_fields="lecture_id"),
        )

        reconcile = rewritten.output_expr
        rebase = reconcile.source
        localized = rebase.source
        candidates = localized.source
        self.assertEqual(reconcile.kind, "reduce")
        self.assertEqual(
            reconcile.spec,
            UdfSpec(module="udfs.temporal_ops", name="reconcile_events"),
        )
        self.assertEqual(rebase.kind, "map")
        self.assertEqual(
            rebase.spec,
            UdfSpec(module="udfs.temporal_ops", name="rebase_clip_events"),
        )
        self.assertEqual(localized.kind, "video_map_each")
        self.assertEqual(candidates.kind, "map")
        self.assertEqual(reconcile.group_by, ("lecture_id", "query"))
        self.assertEqual(
            set(localized.spec.map_spec.output_schema),
            {"clip_events"},
        )

    def test_downstream_grouping_keys_are_preserved(self) -> None:
        program = parse_query(
            f'''
from mmds import ForEach, Input, Map, Record, Reduce
rows = Input("lectures.jsonl")
events = Map(
    rows,
    ["Find ", Record["query"], " in ", Record["video"]],
    schema={{"events": {EVENTS_SCHEMA!r}}},
)
output = Reduce(
    events,
    "course_id",
    ["Summarize ", ForEach([Record["events"]])],
    schema={{"summary": "string"}},
)
'''
        )
        directive = JointTemporalPushdown(identity_fields="lecture_id")
        matches = directive.find_matches(PlanIndex.build(program.output_expr))

        rewritten = apply_rewrite(
            program,
            directive=directive,
            match=matches[0],
            params=PARAMS,
        )

        inserted = rewritten.output_expr.source
        self.assertEqual(inserted.kind, "video_map")
        self.assertEqual(
            inserted.spec.group_by,
            ("lecture_id", "course_id", "query"),
        )

    def test_configuration_requires_safe_identity_and_padding(self) -> None:
        invalid_constructors = (
            lambda: JointTemporalPushdown(identity_fields=[]),
            lambda: JointTemporalPushdown(identity_fields=["lecture_id", "lecture_id"]),
            lambda: JointTemporalPushdown(identity_fields="_mmds_candidate_views"),
            lambda: JointTemporalPushdown(identity_fields="clip"),
            lambda: JointTemporalPushdown(identity_fields="lecture_id", padding_seconds=-1),
            lambda: JointTemporalPushdown(identity_fields="lecture_id", padding_seconds=True),
            lambda: JointTemporalPushdown(
                identity_fields="lecture_id",
                padding_seconds=float("inf"),
            ),
        )

        for construct in invalid_constructors:
            with self.subTest(construct=construct):
                with self.assertRaises((TypeError, ValueError)):
                    construct()

    def test_apply_rejects_unsafe_field_choices_and_prompt_shapes(self) -> None:
        nested = parse_query(
            f'''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(
    rows,
    [Record["video"], Record["query"], Record["metadata"]["term"]],
    schema={{"events": {EVENTS_SCHEMA!r}}},
)
'''
        )
        cases = (
            (
                _event_program(),
                JointTemporalPushdown(identity_fields="video"),
                PARAMS,
                "video field",
            ),
            (
                _event_program(),
                JointTemporalPushdown(identity_fields="lecture_id"),
                {**PARAMS, "query_field": "missing"},
                "query field",
            ),
            (
                nested,
                JointTemporalPushdown(identity_fields="lecture_id"),
                PARAMS,
                "top-level",
            ),
            (
                _event_program(),
                JointTemporalPushdown(identity_fields="lecture_id"),
                {**PARAMS, "transcript_field": "video"},
                "Invalid parameters",
            ),
        )

        for program, directive, params, error in cases:
            with self.subTest(error=error):
                match = directive.find_matches(
                    PlanIndex.build(program.output_expr)
                )[0]
                with self.assertRaisesRegex(MMDSRewriteError, error):
                    apply_rewrite(
                        program,
                        directive=directive,
                        match=match,
                        params=params,
                    )

    def test_per_view_requires_exactly_one_events_output(self) -> None:
        program = _answer_program("lectures.jsonl")
        directive = PerViewTemporalPushdown(identity_fields="lecture_id")
        match = directive.find_matches(PlanIndex.build(program.output_expr))[0]

        with self.assertRaisesRegex(MMDSRewriteError, "exactly one field"):
            apply_rewrite(
                program,
                directive=directive,
                match=match,
                params=PARAMS,
            )


class TemporalDirectiveExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = [
            {
                "lecture_id": "lecture-1",
                "query": "When is sorting explained?",
                "transcript": "00:10 sorting starts; 00:20 partition example",
                "video": {"type": "Video", "source": "lecture.mp4"},
            }
        ]
        self.temp_dir, self.path = _write_rows(rows)
        self.addCleanup(self.temp_dir.cleanup)

    def test_joint_rewrite_calls_candidate_and_final_models_once(self) -> None:
        rewritten = _rewrite(
            _answer_program(self.path),
            JointTemporalPushdown(identity_fields="lecture_id"),
        )
        video_map = rewritten.output_expr
        lowered = lower_video_ops(video_map)
        calls: list[tuple[object, ...]] = []

        def answer(resolved, payload, context):
            calls.append(resolved.parts)
            return {"answer": "A partition example is shown."}

        executor = StaticPromptExecutor(
            {
                ("map", video_map.source.spec.cache_key()): {
                    "_mmds_candidate_views": [
                        {"start": 10, "end": 15},
                        {"start": 20, "end": 25},
                    ]
                },
                ("reduce", lowered.spec.cache_key()): answer,
            }
        )

        result = execute(rewritten, prompt_executor=executor)

        views = [
            part
            for part in calls[0]
            if isinstance(part, dict) and part.get("type") == "VideoView"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [(view["start"], view["end"]) for view in views],
            [(10.0, 15.0), (20.0, 25.0)],
        )
        self.assertEqual(
            result,
            [
                {
                    "lecture_id": "lecture-1",
                    "query": "When is sorting explained?",
                    "answer": "A partition example is shown.",
                }
            ],
        )

    def test_per_view_rewrite_coalesces_rebases_and_collects(self) -> None:
        rewritten = _rewrite(
            _event_program(self.path),
            PerViewTemporalPushdown(
                identity_fields="lecture_id",
                padding_seconds=5,
            ),
        )
        reconcile = rewritten.output_expr
        localized = reconcile.source.source
        candidates = localized.source
        lowered_localized = lower_video_ops(localized)
        seen_views: list[dict] = []

        def localize(resolved, payload, context):
            view = next(
                part
                for part in resolved.parts
                if isinstance(part, dict) and part.get("type") == "VideoView"
            )
            seen_views.append(dict(view))
            return {
                "clip_events": [
                    {"type": "sorting", "start": 2, "end": 4}
                ]
            }

        executor = StaticPromptExecutor(
            {
                ("map", candidates.spec.cache_key()): {
                    "_mmds_candidate_views": [
                        {"start": 10, "end": 20},
                        {"start": 18, "end": 25},
                    ]
                },
                ("map", lowered_localized.spec.cache_key()): localize,
            }
        )

        result = execute(rewritten, prompt_executor=executor)

        self.assertEqual(
            [(view["start"], view["end"]) for view in seen_views],
            [(5.0, 30.0)],
        )
        self.assertEqual(
            result,
            [
                {
                    "lecture_id": "lecture-1",
                    "query": "When is sorting explained?",
                    "events": [
                        {"type": "sorting", "start": 7.0, "end": 9.0}
                    ],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
