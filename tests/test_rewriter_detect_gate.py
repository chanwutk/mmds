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

from mmds import DetectSpec, PromptSpec, UdfSpec, parse_query, render_query  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    DetectGateBeforeMap,
    MMDSRewriteError,
    PlanIndex,
    apply_rewrite,
)


def _video_map_program():
    """Naive VLM Map over clips: describe bears from video + title (no Detect gate)."""
    return parse_query(
        '''
from mmds import Input, Map, Record
clips = Input("clips.jsonl")
output = Map(
    clips,
    [
        "Describe any bears visible in ",
        Record["video"],
        " with title ",
        Record["title"],
    ],
    schema={"summary": "string"},
    name="describe_bears",
)
'''
    )


class DetectRoundTripTests(unittest.TestCase):
    # Detect must serialize to normalized Python and parse back unchanged so
    # rewrite directives that insert Detect pass validate_rewrite_structure.
    def test_detect_parse_render_round_trip(self) -> None:
        program = parse_query(
            '''
from mmds import Detect, Filter, Input
from udfs.detection_ops import keep_rows_with_detections
clips = Input("clips.jsonl")
detected = Detect(clips, "video", ["bear", "deer"], name="gate")
output = Filter(detected, keep_rows_with_detections)
'''
        )
        detect = program.output_expr.source
        self.assertEqual(detect.kind, "detect")
        self.assertIsInstance(detect.spec, DetectSpec)
        assert isinstance(detect.spec, DetectSpec)
        self.assertEqual(detect.spec.video_field, "video")
        self.assertEqual(detect.spec.classes, ("bear", "deer"))
        self.assertEqual(detect.spec.output_field, "detections")

        rendered = render_query(program)
        self.assertIn('Detect(clips, "video", ["bear", "deer"]', rendered)
        reparsed = parse_query(rendered)
        self.assertEqual(reparsed.output_expr, program.output_expr)


class DetectGateBeforeMapTests(unittest.TestCase):
    # Happy path: Map(video) becomes Detect → Filter(keep_rows_with_detections)
    # → Map, preserving the original Map prompt, schema, and name.
    def test_inserts_detect_and_keep_filter_before_map(self) -> None:
        original = _video_map_program()
        rewritten = apply_rewrite(
            original,
            directive=DetectGateBeforeMap(),
            match=DetectGateBeforeMap().find_matches(
                PlanIndex.build(original.output_expr)
            )[0],
            params={"video_field": "video", "classes": ["bear"]},
        )

        mapped = rewritten.output_expr
        gated = mapped.source
        detected = gated.source
        self.assertEqual(mapped.kind, "map")
        self.assertEqual(mapped.name, "describe_bears")
        self.assertIsInstance(mapped.spec, PromptSpec)
        self.assertEqual(
            mapped.spec.output_schema,
            original.output_expr.spec.output_schema,
        )
        self.assertEqual(mapped.spec.parts, original.output_expr.spec.parts)
        self.assertEqual(gated.kind, "filter")
        self.assertEqual(gated.name, "rewrite_keep_detections")
        self.assertEqual(
            gated.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="keep_rows_with_detections",
            ),
        )
        self.assertEqual(detected.kind, "detect")
        self.assertEqual(detected.name, "rewrite_detect_gate")
        self.assertIsInstance(detected.spec, DetectSpec)
        assert isinstance(detected.spec, DetectSpec)
        self.assertEqual(detected.spec.classes, ("bear",))
        self.assertEqual(detected.spec.video_field, "video")
        self.assertEqual(detected.source, original.output_expr.source)

    # Apply-time guards: video_field must be a direct top-level Record ref,
    # nested Record paths are unsupported, and do not double-gate an existing
    # Detect → keep Filter stack.
    def test_rejects_missing_video_nested_or_existing_gate(self) -> None:
        already_gated = parse_query(
            '''
from mmds import Detect, Filter, Input, Map, Record
from udfs.detection_ops import keep_rows_with_detections
clips = Input("clips.jsonl")
detected = Detect(clips, "video", ["bear"])
gated = Filter(detected, keep_rows_with_detections)
output = Map(
    gated,
    ["Describe ", Record["video"]],
    schema={"summary": "string"},
)
'''
        )
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
                already_gated,
                {"video_field": "video", "classes": ["bear"]},
                "already sits behind a Detect gate",
            ),
        )
        for program, params, error in cases:
            with self.subTest(error=error):
                match = DetectGateBeforeMap().find_matches(
                    PlanIndex.build(program.output_expr)
                )[0]
                with self.assertRaisesRegex(MMDSRewriteError, error):
                    apply_rewrite(
                        program,
                        directive=DetectGateBeforeMap(),
                        match=match,
                        params=params,
                    )

    # Pydantic param contract: non-empty unique classes, non-blank video_field
    # and model.
    def test_parameters_are_strict(self) -> None:
        program = _video_map_program()
        match = DetectGateBeforeMap().find_matches(
            PlanIndex.build(program.output_expr)
        )[0]
        invalid = (
            {"video_field": "video", "classes": []},
            {"video_field": "video", "classes": ["bear", "bear"]},
            {"video_field": "video", "classes": ["  "]},
            {"video_field": "   ", "classes": ["bear"]},
            {"video_field": "video", "classes": ["bear"], "model": ""},
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
                    apply_rewrite(
                        program,
                        directive=DetectGateBeforeMap(),
                        match=match,
                        params=params,
                    )


if __name__ == "__main__":
    unittest.main()
