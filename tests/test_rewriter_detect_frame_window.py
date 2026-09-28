from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    DetectSpec,
    PromptSpec,
    UdfSpec,
    VideoMapSpec,
    parse_query,
)
from mmds.optimizers.rewriter import (  # noqa: E402
    DetectedFrameWindowBeforeMap,
    MMDSRewriteError,
    PlanIndex,
    apply_rewrite,
)
from udfs.detection_ops import detections_to_candidate_views  # noqa: E402


def _video_map_program():
    """Naive VLM Map over long clips that asks about bears in the full video."""
    return parse_query(
        '''
from mmds import Input, Map, Record
clips = Input("clips.jsonl")
output = Map(
    clips,
    [
        "Describe any bears visible in ",
        Record["video"],
        " for query ",
        Record["query"],
    ],
    schema={"summary": "string"},
    name="describe_bears",
)
'''
    )


class DetectionsToCandidateViewsTests(unittest.TestCase):
    # One-frame windows from absolute frame_idx / fps; empty detections → [].
    def test_converts_frame_indices_using_row_fps(self) -> None:
        row = {
            "video": {"type": "Video", "source": "clip.mp4"},
            "_mmds_video_fps": 10.0,
            "detections": [
                {
                    "type": "bear",
                    "bboxes": [
                        {"frame_idx": 0, "bbox": [0, 0, 1, 1], "confidence": 0.9},
                        {"frame_idx": 5, "bbox": [0, 0, 1, 1], "confidence": 0.8},
                        {"frame_idx": 5, "bbox": [1, 1, 2, 2], "confidence": 0.7},
                    ],
                }
            ],
        }
        self.assertEqual(
            detections_to_candidate_views(row),
            {
                "_mmds_candidate_views": [
                    {"start": 0.0, "end": 0.1},
                    {"start": 0.5, "end": 0.6},
                ]
            },
        )
        self.assertEqual(
            detections_to_candidate_views({"detections": []}),
            {"_mmds_candidate_views": []},
        )


class DetectedFrameWindowBeforeMapTests(unittest.TestCase):
    # Happy path: Detect → interval Map → joint VideoMap over coalesced windows,
    # preserving the original Map schema/name. Coalesce runs inside VideoMap lowering.
    def test_builds_detect_interval_map_then_joint_video_map(self) -> None:
        original = _video_map_program()
        rewritten = apply_rewrite(
            original,
            directive=DetectedFrameWindowBeforeMap(identity_fields="clip_id"),
            match=DetectedFrameWindowBeforeMap(
                identity_fields="clip_id"
            ).find_matches(PlanIndex.build(original.output_expr))[0],
            params={
                "video_field": "video",
                "classes": ["bear"],
                "padding_seconds": 1.5,
                "min_confidence": 0.25,
            },
        )

        video_map = rewritten.output_expr
        intervals = video_map.source
        detected = intervals.source
        self.assertEqual(video_map.kind, "video_map")
        self.assertEqual(video_map.name, "describe_bears")
        self.assertIsInstance(video_map.spec, VideoMapSpec)
        assert isinstance(video_map.spec, VideoMapSpec)
        self.assertEqual(video_map.spec.video_field, "video")
        self.assertEqual(video_map.spec.views_field, "_mmds_candidate_views")
        self.assertEqual(video_map.spec.group_by, ("clip_id", "query"))
        self.assertEqual(video_map.spec.padding_time, 1.5)
        self.assertIsInstance(video_map.spec.map_spec, PromptSpec)
        self.assertEqual(
            video_map.spec.map_spec.output_schema,
            original.output_expr.spec.output_schema,
        )
        self.assertEqual(video_map.spec.map_spec.parts, original.output_expr.spec.parts)

        self.assertEqual(intervals.kind, "map")
        self.assertEqual(
            intervals.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="detections_to_candidate_views",
            ),
        )
        self.assertEqual(detected.kind, "detect")
        self.assertIsInstance(detected.spec, DetectSpec)
        assert isinstance(detected.spec, DetectSpec)
        self.assertEqual(detected.spec.classes, ("bear",))
        self.assertEqual(detected.spec.conf, 0.25)
        self.assertEqual(detected.source, original.output_expr.source)

    def test_rejects_unsafe_matches_and_params(self) -> None:
        nested = parse_query(
            '''
from mmds import Input, Map, Record
clips = Input("clips.jsonl")
output = Map(
    clips,
    [Record["video"], Record["metadata"]["term"]],
    schema={"summary": "string"},
)
'''
        )
        directive = DetectedFrameWindowBeforeMap(identity_fields="clip_id")
        cases = (
            (
                _video_map_program(),
                {"video_field": "missing", "classes": ["bear"]},
                "directly reference",
            ),
            (
                nested,
                {"video_field": "video", "classes": ["bear"]},
                "top-level",
            ),
            (
                _video_map_program(),
                {
                    "video_field": "video",
                    "classes": ["bear"],
                    "min_confidence": 1.5,
                },
                "Invalid parameters",
            ),
            (
                _video_map_program(),
                {
                    "video_field": "video",
                    "classes": ["bear"],
                    "padding_seconds": -1,
                },
                "Invalid parameters",
            ),
        )
        for program, params, error in cases:
            with self.subTest(error=error, params=params):
                match = directive.find_matches(PlanIndex.build(program.output_expr))[0]
                with self.assertRaisesRegex(MMDSRewriteError, error):
                    apply_rewrite(
                        program,
                        directive=directive,
                        match=match,
                        params=params,
                    )

    def test_identity_fields_configuration(self) -> None:
        invalid = (
            lambda: DetectedFrameWindowBeforeMap(identity_fields=[]),
            lambda: DetectedFrameWindowBeforeMap(identity_fields=["a", "a"]),
            lambda: DetectedFrameWindowBeforeMap(identity_fields="_mmds_candidate_views"),
            lambda: DetectedFrameWindowBeforeMap(identity_fields="clip"),
        )
        for construct in invalid:
            with self.subTest(construct=construct):
                with self.assertRaises((TypeError, ValueError)):
                    construct()


class DetectedFrameWindowDownstreamFieldTests(unittest.TestCase):
    """Same guard as temporal pushdown: the rewrite emits only group_by + outputs."""

    PARAMS = {"video_field": "video", "classes": ["dog"], "padding_seconds": 1.0}

    def _program(self, downstream: str, *, schema: str = '{"has_dog": "boolean"}'):
        return parse_query(
            f"""
from mmds import ForEach, Input, Map, Record, Reduce
from udfs.test_ops import annotate
clips = Input("clips.jsonl")
answers = Map(clips, ["Is there a dog in ", Record["video"], "?"], schema={schema})
{downstream}
"""
        )

    def _offered(self, program, identity_fields="clip_id"):
        directive = DetectedFrameWindowBeforeMap(identity_fields=identity_fields)
        return [str(m.path) for m in directive.find_matches(PlanIndex.build(program.output_expr))]

    def test_not_offered_when_downstream_reads_a_dropped_field(self) -> None:
        program = self._program(
            'output = Map(answers, ["Title ", Record["title"], ": ", Record["has_dog"]], '
            'schema={"line": "string"})'
        )

        self.assertNotIn("output.source", self._offered(program))
        self.assertIn("output.source", self._offered(program, ["clip_id", "title"]))

    def test_not_offered_when_a_downstream_udf_reads_the_rows(self) -> None:
        program = self._program("output = Map(answers, annotate)")

        self.assertNotIn("output.source", self._offered(program))

    def test_group_by_excludes_fields_the_map_itself_produces(self) -> None:
        program = self._program(
            'output = Reduce(answers, "category", ["Count ", ForEach([Record["has_dog"]])], '
            'schema={"count": "integer"})',
            schema='{"has_dog": "boolean", "category": "string"}',
        )
        directive = DetectedFrameWindowBeforeMap(identity_fields="clip_id")
        match = directive.find_matches(PlanIndex.build(program.output_expr))[0]

        rewritten = apply_rewrite(program, directive=directive, match=match, params=self.PARAMS)

        self.assertEqual(rewritten.output_expr.source.spec.group_by, ("clip_id",))

    def test_apply_rejects_a_downstream_read_of_the_chosen_video_field(self) -> None:
        program = self._program(
            'output = Map(answers, ["Replay ", Record["video"], ": ", Record["has_dog"]], '
            'schema={"line": "string"})'
        )
        directive = DetectedFrameWindowBeforeMap(identity_fields="clip_id")
        match = next(
            m
            for m in directive.find_matches(PlanIndex.build(program.output_expr))
            if str(m.path) == "output.source"
        )

        with self.assertRaisesRegex(MMDSRewriteError, r"drop fields read downstream: \['video'\]"):
            apply_rewrite(program, directive=directive, match=match, params=self.PARAMS)


if __name__ == "__main__":
    unittest.main()
