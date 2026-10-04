from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import DetectSpec, UdfSpec, parse_query, render_query  # noqa: E402
from mmds.execution.ops.map import _apply_map  # noqa: E402
from mmds.model import DatasetExpr  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    DetectPresenceMap,
    MMDSRewriteError,
    PlanIndex,
    apply_rewrite,
)
from udfs.detection_ops import map_detection_presence  # noqa: E402


def _presence_program():
    return parse_query(
        '''
from mmds import Input, Map, Record
clips = Input("clips.jsonl")
output = Map(
    clips,
    [
        "Watch this video.\\n",
        Record["video"],
        "\\nIs a bear clearly visible anywhere in this video? "
        "Answer true only when a bear is visible, otherwise false.",
    ],
    schema={"bear_present": "boolean"},
    name="bear_present",
)
'''
    )


class DetectPresenceMapTests(unittest.TestCase):
    def test_replaces_boolean_map_with_detect_filter_and_code_map(self) -> None:
        original = _presence_program()
        rewritten = apply_rewrite(
            original,
            directive=DetectPresenceMap(),
            match=DetectPresenceMap().find_matches(
                PlanIndex.build(original.output_expr)
            )[0],
            params={
                "video_field": "video",
                "classes": ["bear"],
                "flag_field": "bear_present",
            },
        )

        mapped = rewritten.output_expr
        gated = mapped.source
        detected = gated.source
        self.assertEqual(mapped.kind, "map")
        self.assertEqual(mapped.name, "bear_present")
        self.assertEqual(
            mapped.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="map_detection_presence",
                args=("bear_present",),
            ),
        )
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
        self.assertEqual(detected.name, "rewrite_detect_presence")
        self.assertIsInstance(detected.spec, DetectSpec)
        assert isinstance(detected.spec, DetectSpec)
        self.assertEqual(detected.spec.classes, ("bear",))
        self.assertEqual(detected.spec.video_field, "video")
        self.assertEqual(detected.spec.stop_after_n, 1)
        self.assertEqual(detected.source, original.output_expr.source)

    def test_does_not_offer_non_boolean_maps(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
clips = Input("clips.jsonl")
output = Map(
    clips,
    ["Describe ", Record["video"]],
    schema={"summary": "string"},
)
'''
        )
        matches = DetectPresenceMap().find_matches(
            PlanIndex.build(program.output_expr)
        )
        self.assertEqual(matches, ())

    def test_rejects_wrong_flag_or_missing_video(self) -> None:
        program = _presence_program()
        match = DetectPresenceMap().find_matches(
            PlanIndex.build(program.output_expr)
        )[0]
        cases = (
            (
                {
                    "video_field": "video",
                    "classes": ["bear"],
                    "flag_field": "other",
                },
                "schema is exactly",
            ),
            (
                {
                    "video_field": "title",
                    "classes": ["bear"],
                    "flag_field": "bear_present",
                },
                "directly reference",
            ),
        )
        for params, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(MMDSRewriteError, error):
                    apply_rewrite(
                        program,
                        directive=DetectPresenceMap(),
                        match=match,
                        params=params,
                    )

    def test_parameters_are_strict(self) -> None:
        program = _presence_program()
        match = DetectPresenceMap().find_matches(
            PlanIndex.build(program.output_expr)
        )[0]
        invalid = (
            {"video_field": "video", "classes": [], "flag_field": "bear_present"},
            {
                "video_field": "video",
                "classes": ["bear", "bear"],
                "flag_field": "bear_present",
            },
            {"video_field": "  ", "classes": ["bear"], "flag_field": "bear_present"},
            {"video_field": "video", "classes": ["bear"], "flag_field": "  "},
            {
                "video_field": "video",
                "classes": ["bear"],
                "flag_field": "bear_present",
                "min_tracks": 0,
            },
            {
                "video_field": "video",
                "classes": ["bear"],
                "flag_field": "bear_present",
                "min_tracks": True,
            },
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
                    apply_rewrite(
                        program,
                        directive=DetectPresenceMap(),
                        match=match,
                        params=params,
                    )

    def test_min_tracks_counts_tracks_and_stops_at_that_count(self) -> None:
        original = _presence_program()
        rewritten = apply_rewrite(
            original,
            directive=DetectPresenceMap(),
            match=DetectPresenceMap().find_matches(
                PlanIndex.build(original.output_expr)
            )[0],
            params={
                "video_field": "video",
                "classes": ["bear"],
                "flag_field": "bear_present",
                "min_tracks": 5,
            },
        )
        mapped = rewritten.output_expr
        gated = mapped.source
        detected = gated.source
        self.assertEqual(detected.spec.stop_after_n, 5)
        self.assertEqual(
            gated.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="keep_rows_with_at_least_n_tracks",
                args=("5",),
            ),
        )
        self.assertEqual(
            mapped.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="map_at_least_n_tracks",
                args=("bear_present", "5"),
            ),
        )

    def test_track_threshold_ignores_repeated_boxes_of_one_track(self) -> None:
        from udfs.detection_ops import (  # noqa: E402
            keep_rows_with_at_least_n_tracks,
            map_at_least_n_tracks,
        )

        one_track = {
            "detections": [
                {
                    "type": "bear",
                    "bboxes": [
                        {"frame_idx": index, "track_id": 1, "bbox": [0, 0, 1, 1], "confidence": 0.9}
                        for index in range(8)
                    ],
                }
            ]
        }
        five_tracks = {
            "detections": [
                {
                    "type": "bear",
                    "bboxes": [
                        {
                            "frame_idx": index,
                            "track_id": index + 1,
                            "bbox": [0, 0, 1, 1],
                            "confidence": 0.9,
                        }
                        for index in range(5)
                    ],
                }
            ]
        }
        self.assertFalse(keep_rows_with_at_least_n_tracks(one_track, "5"))
        self.assertEqual(
            map_at_least_n_tracks(one_track, "at_least_five_bears", "5"),
            {"at_least_five_bears": False},
        )
        self.assertTrue(keep_rows_with_at_least_n_tracks(five_tracks, "5"))
        self.assertEqual(
            map_at_least_n_tracks(five_tracks, "at_least_five_bears", "5"),
            {"at_least_five_bears": True},
        )

    def test_five_bears_example_imports_as_python(self) -> None:
        path = ROOT / "examples" / "animals_five_bears_detect_presence.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.detected.spec.stop_after_n, 5)
        self.assertEqual(module.output.spec.name, "map_at_least_five_bears")

    def test_bound_udf_sets_the_named_boolean(self) -> None:
        node = DatasetExpr(
            kind="map",
            source=DatasetExpr(kind="input", input_path="rows.jsonl"),
            spec=UdfSpec(
                module="udfs.detection_ops",
                name="map_detection_presence",
                args=("bear_present",),
            ),
        )
        present = _apply_map(
            node,
            {
                "video": "clip.mp4",
                "detections": [{"type": "bear", "bboxes": [{"frame_idx": 0}]}],
            },
            None,
        )
        absent = _apply_map(node, {"video": "clip.mp4", "detections": []}, None)

        self.assertTrue(present["bear_present"])
        self.assertEqual(present["video"], "clip.mp4")
        self.assertFalse(absent["bear_present"])
        self.assertEqual(
            map_detection_presence({"detections": []}, "bear_present"),
            {"bear_present": False},
        )

    def test_udf_call_round_trips(self) -> None:
        program = parse_query(
            '''
from mmds import Detect, Filter, Input, Map
from udfs.detection_ops import keep_rows_with_detections, map_detection_presence
clips = Input("clips.jsonl")
detected = Detect(clips, "video", ["bear"], stop_after_n=1)
kept = Filter(detected, keep_rows_with_detections)
output = Map(kept, map_detection_presence("bear_present"), name="bear_present")
'''
        )
        mapped = program.output_expr
        self.assertEqual(
            mapped.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="map_detection_presence",
                args=("bear_present",),
            ),
        )
        rendered = render_query(program)
        self.assertIn('map_detection_presence("bear_present")', rendered)
        self.assertEqual(parse_query(rendered).output_expr, program.output_expr)

    def test_example_imports_as_python(self) -> None:
        path = ROOT / "examples" / "animals_bear_detect_presence.py"
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.output.kind, "map")
        self.assertEqual(
            module.output.spec,
            UdfSpec(module="udfs.detection_ops", name="map_bear_present"),
        )
        program = parse_query(path.read_text(encoding="utf-8"))
        self.assertEqual(program.output_expr.spec, module.output.spec)

    def test_udf_call_rejects_non_strings(self) -> None:
        with self.assertRaises(Exception):
            parse_query(
                '''
from mmds import Input, Map
from udfs.detection_ops import map_detection_presence
clips = Input("clips.jsonl")
output = Map(clips, map_detection_presence(1))
'''
            )


if __name__ == "__main__":
    unittest.main()
