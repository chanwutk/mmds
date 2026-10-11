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
from mmds.model import MAX_TRACK_FRAME_GAP, DatasetExpr  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    DetectCountMap,
    DetectPresenceMap,
    MMDSRewriteError,
    PlanIndex,
    apply_rewrite,
)
from udfs.detection_ops import (  # noqa: E402
    count_detection_tracks,
    keep_rows_with_at_least_n_tracks,
    map_bear_count,
    map_detection_track_count,
    map_vehicle_count,
)

_PARAMS = {"video_field": "video", "classes": ["bear"], "count_field": "bear_count"}


def _count_program(schema: str = '{"bear_count": "integer"}'):
    return parse_query(
        f'''
from mmds import Input, Map, Record
videos = Input("videos.jsonl")
output = Map(
    videos,
    [
        "Watch this video.\\n",
        Record["video"],
        "\\nHow many bears are there?",
    ],
    schema={schema},
    name="bear_count",
)
'''
    )


def _match(program):
    return DetectCountMap().find_matches(PlanIndex.build(program.output_expr))[0]


def _bboxes(*track_ids):
    return [
        {"frame_idx": index, "track_id": track_id, "bbox": [0, 0, 1, 1], "confidence": 0.9}
        for index, track_id in enumerate(track_ids)
    ]


class DetectCountMapRewriteTests(unittest.TestCase):
    def test_replaces_integer_map_with_full_scan_detect_and_count_map(self) -> None:
        original = _count_program()
        rewritten = apply_rewrite(
            original,
            directive=DetectCountMap(),
            match=_match(original),
            params={**_PARAMS, "frame_stride": 10},
        )

        mapped = rewritten.output_expr
        detected = mapped.source
        self.assertEqual(mapped.kind, "map")
        self.assertEqual(mapped.name, "bear_count")
        self.assertEqual(
            mapped.spec,
            UdfSpec(
                module="udfs.detection_ops",
                name="map_detection_track_count",
                args=("bear_count",),
            ),
        )
        self.assertEqual(detected.kind, "detect")
        self.assertEqual(detected.name, "rewrite_detect_count")
        assert isinstance(detected.spec, DetectSpec)
        self.assertEqual(detected.spec.classes, ("bear",))
        self.assertEqual(detected.spec.video_field, "video")
        self.assertEqual(detected.spec.output_field, "detections")
        self.assertEqual(detected.spec.frame_stride, 10)
        self.assertIsNone(detected.spec.stop_after_n)
        self.assertEqual(detected.source, original.output_expr.source)

    def test_rewritten_plan_round_trips_through_render(self) -> None:
        original = _count_program()
        rewritten = apply_rewrite(
            original, directive=DetectCountMap(), match=_match(original), params=_PARAMS
        )
        rendered = render_query(rewritten)
        self.assertIn('map_detection_track_count("bear_count")', rendered)
        self.assertNotIn("stop_after_n", rendered)
        self.assertEqual(parse_query(rendered).output_expr, rewritten.output_expr)

    def test_accepts_expanded_integer_schema(self) -> None:
        program = _count_program('{"bear_count": {"type": "integer"}}')
        self.assertEqual(len(DetectCountMap().find_matches(PlanIndex.build(program.output_expr))), 1)

    def test_does_not_offer_non_integer_or_multi_field_maps(self) -> None:
        for schema in (
            '{"bear_present": "boolean"}',
            '{"bear_count": "number"}',
            '{"bear_count": "integer", "notes": "string"}',
        ):
            with self.subTest(schema=schema):
                program = _count_program(schema)
                self.assertEqual(
                    DetectCountMap().find_matches(PlanIndex.build(program.output_expr)),
                    (),
                )

    def test_presence_directive_does_not_offer_integer_maps(self) -> None:
        program = _count_program()
        self.assertEqual(
            DetectPresenceMap().find_matches(PlanIndex.build(program.output_expr)), ()
        )

    def test_rejects_wrong_count_field_or_missing_video(self) -> None:
        program = _count_program()
        cases = (
            ({**_PARAMS, "count_field": "other"}, "schema is exactly"),
            ({**_PARAMS, "video_field": "title"}, "directly reference"),
        )
        for params, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(MMDSRewriteError, error):
                    apply_rewrite(
                        program, directive=DetectCountMap(), match=_match(program), params=params
                    )

    def test_parameters_are_strict(self) -> None:
        program = _count_program()
        invalid = (
            {**_PARAMS, "classes": []},
            {**_PARAMS, "classes": ["bear", " bear "]},
            {**_PARAMS, "classes": ["  "]},
            {**_PARAMS, "video_field": "  "},
            {**_PARAMS, "count_field": "  "},
            {**_PARAMS, "model": "  "},
            {**_PARAMS, "frame_stride": 0},
            {**_PARAMS, "frame_stride": MAX_TRACK_FRAME_GAP + 1},
            {**_PARAMS, "frame_stride": True},
            {**_PARAMS, "frame_stride": 2.0},
            {**_PARAMS, "stop_after_n": 5},
            {**_PARAMS, "min_track_boxes": 0},
            {**_PARAMS, "min_track_boxes": True},
            {**_PARAMS, "min_track_boxes": "3"},
            {**_PARAMS, "min_track_boxes": 3.0},
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
                    apply_rewrite(
                        program, directive=DetectCountMap(), match=_match(program), params=params
                    )

    def test_frame_stride_at_the_track_gap_is_allowed(self) -> None:
        program = _count_program()
        rewritten = apply_rewrite(
            program,
            directive=DetectCountMap(),
            match=_match(program),
            params={**_PARAMS, "frame_stride": MAX_TRACK_FRAME_GAP},
        )
        self.assertEqual(rewritten.output_expr.source.spec.frame_stride, MAX_TRACK_FRAME_GAP)

    def test_min_track_boxes_above_one_is_bound_as_a_second_argument(self) -> None:
        program = _count_program()
        rewritten = apply_rewrite(
            program,
            directive=DetectCountMap(),
            match=_match(program),
            params={**_PARAMS, "min_track_boxes": 3},
        )
        self.assertEqual(rewritten.output_expr.spec.args, ("bear_count", "3"))
        rendered = render_query(rewritten)
        self.assertIn('map_detection_track_count("bear_count", "3")', rendered)
        self.assertEqual(parse_query(rendered).output_expr, rewritten.output_expr)

    def test_min_track_boxes_of_one_keeps_the_single_argument_form(self) -> None:
        program = _count_program()
        rewritten = apply_rewrite(
            program,
            directive=DetectCountMap(),
            match=_match(program),
            params={**_PARAMS, "min_track_boxes": 1},
        )
        self.assertEqual(rewritten.output_expr.spec.args, ("bear_count",))


class TrackCountUdfTests(unittest.TestCase):
    def test_counts_distinct_tracks_not_boxes(self) -> None:
        row = {"detections": [{"type": "bear", "bboxes": _bboxes(1, 1, 1, 2, 2, 3)}]}
        self.assertEqual(map_detection_track_count(row, "bear_count"), {"bear_count": 3})
        self.assertEqual(map_bear_count(row), {"bear_count": 3})

    def test_counts_tracks_across_class_groups(self) -> None:
        row = {
            "detections": [
                {"type": "bear", "bboxes": _bboxes(1, 2)},
                {"type": "black bear", "bboxes": _bboxes(3)},
            ]
        }
        self.assertEqual(count_detection_tracks(row), 3)

    def test_empty_or_malformed_detections_count_zero(self) -> None:
        for row in (
            {},
            {"detections": None},
            {"detections": []},
            {"detections": ["bear"]},
            {"detections": [{"type": "bear", "bboxes": None}]},
            {"detections": [{"type": "bear", "bboxes": ["box"]}]},
            {"detections": [{"type": "bear", "bboxes": [{"frame_idx": 0}]}]},
            {"detections": [{"type": "bear", "bboxes": [{"track_id": True}]}]},
            {"detections": [{"type": "bear", "bboxes": [{"track_id": "1"}]}]},
        ):
            with self.subTest(row=row):
                self.assertEqual(map_bear_count(row), {"bear_count": 0})

    def test_min_boxes_drops_short_tracks(self) -> None:
        row = {"detections": [{"type": "vehicle", "bboxes": _bboxes(1, 1, 1, 2, 2, 3)}]}
        self.assertEqual(count_detection_tracks(row, 1), 3)
        self.assertEqual(count_detection_tracks(row, 2), 2)
        self.assertEqual(count_detection_tracks(row, 3), 1)
        self.assertEqual(count_detection_tracks(row, 4), 0)
        self.assertEqual(
            map_detection_track_count(row, "vehicle_count", "2"), {"vehicle_count": 2}
        )

    def test_min_boxes_counts_boxes_across_class_groups(self) -> None:
        row = {
            "detections": [
                {"type": "car", "bboxes": _bboxes(1, 1)},
                {"type": "truck", "bboxes": _bboxes(1)},
            ]
        }
        self.assertEqual(count_detection_tracks(row, 3), 1)

    def test_vehicle_wrapper_requires_three_boxes(self) -> None:
        row = {"detections": [{"type": "vehicle", "bboxes": _bboxes(1, 1, 1, 2, 2, 3)}]}
        self.assertEqual(map_vehicle_count(row), {"vehicle_count": 1})
        self.assertEqual(map_vehicle_count({}), {"vehicle_count": 0})

    def test_invalid_min_boxes_raise(self) -> None:
        for value in (0, -1, True, 1.0, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    count_detection_tracks({"detections": []}, value)
        for value in ("0", "-1", "03", "x", "", "1.5", 3, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "min_track_boxes"):
                    map_detection_track_count({"detections": []}, "n", value)

    def test_threshold_udf_error_names_its_argument(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_tracks"):
            keep_rows_with_at_least_n_tracks({"detections": []}, "0")

    def test_rejects_blank_count_field(self) -> None:
        for field in ("", None, 3):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    map_detection_track_count({"detections": []}, field)

    def test_bound_udf_keeps_the_row_and_sets_the_count(self) -> None:
        node = DatasetExpr(
            kind="map",
            source=DatasetExpr(kind="input", input_path="rows.jsonl"),
            spec=UdfSpec(
                module="udfs.detection_ops",
                name="map_detection_track_count",
                args=("bear_count",),
            ),
        )
        row = _apply_map(
            node,
            {"video": "clip.mp4", "detections": [{"type": "bear", "bboxes": _bboxes(4, 7)}]},
            None,
        )
        self.assertEqual(row["bear_count"], 2)
        self.assertEqual(row["video"], "clip.mp4")


class BearCountExampleTests(unittest.TestCase):
    def _load(self, name: str):
        path = ROOT / "examples" / name
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return path, module

    def test_detect_example_scans_the_full_video(self) -> None:
        path, module = self._load("animals_bear_count.py")
        self.assertEqual(module.videos.input_path, "data/swan_valley_full.jsonl")
        self.assertEqual(module.detected.spec.classes, ("bear",))
        self.assertIsNone(module.detected.spec.stop_after_n)
        self.assertLessEqual(module.detected.spec.frame_stride, MAX_TRACK_FRAME_GAP)
        self.assertEqual(
            module.output.spec, UdfSpec(module="udfs.detection_ops", name="map_bear_count")
        )
        self.assertEqual(parse_query(path.read_text(encoding="utf-8")).output_expr, module.output)

    def test_semantic_example_is_the_matching_integer_map(self) -> None:
        path, module = self._load("animals_bear_count_map.py")
        self.assertEqual(module.videos.input_path, "data/swan_valley_full.jsonl")
        self.assertEqual(module.output.spec.output_schema, {"bear_count": "integer"})
        program = parse_query(path.read_text(encoding="utf-8"))
        self.assertEqual(len(DetectCountMap().find_matches(PlanIndex.build(program.output_expr))), 1)

    def test_rewriting_the_semantic_example_matches_the_detect_example(self) -> None:
        _, semantic = self._load("animals_bear_count_map.py")
        _, detect = self._load("animals_bear_count.py")
        program = parse_query((ROOT / "examples" / "animals_bear_count_map.py").read_text(encoding="utf-8"))
        rewritten = apply_rewrite(
            program,
            directive=DetectCountMap(),
            match=_match(program),
            params={**_PARAMS, "frame_stride": detect.detected.spec.frame_stride},
        )
        detected = rewritten.output_expr.source
        self.assertEqual(detected.spec.classes, detect.detected.spec.classes)
        self.assertEqual(detected.spec.frame_stride, detect.detected.spec.frame_stride)
        self.assertEqual(detected.spec.output_field, detect.detected.spec.output_field)
        self.assertEqual(detected.source, semantic.output.source)


class VehicleCountExampleTests(unittest.TestCase):
    def _load(self, name: str):
        path = ROOT / "examples" / name
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return path, module

    def test_detect_example_tracks_every_frame_with_one_class(self) -> None:
        path, module = self._load("vehicle_count.py")
        self.assertEqual(module.clips.input_path, "data/i24v_highway2_5s.jsonl")
        self.assertEqual(module.detected.spec.classes, ("vehicle",))
        self.assertEqual(module.detected.spec.frame_stride, 1)
        self.assertIsNone(module.detected.spec.stop_after_n)
        self.assertEqual(
            module.output.spec, UdfSpec(module="udfs.detection_ops", name="map_vehicle_count")
        )
        self.assertEqual(parse_query(path.read_text(encoding="utf-8")).output_expr, module.output)

    def test_rewriting_the_semantic_example_matches_the_detect_example(self) -> None:
        path, semantic = self._load("vehicle_count_map.py")
        _, detect = self._load("vehicle_count.py")
        self.assertEqual(semantic.output.spec.output_schema, {"vehicle_count": "integer"})
        program = parse_query(path.read_text(encoding="utf-8"))
        rewritten = apply_rewrite(
            program,
            directive=DetectCountMap(),
            match=_match(program),
            params={
                "video_field": "video",
                "classes": ["vehicle"],
                "count_field": "vehicle_count",
                "frame_stride": 1,
                "min_track_boxes": 3,
            },
        )
        detected = rewritten.output_expr.source
        self.assertEqual(rewritten.output_expr.spec.args, ("vehicle_count", "3"))
        self.assertEqual(detected.spec.classes, detect.detected.spec.classes)
        self.assertEqual(detected.spec.frame_stride, detect.detected.spec.frame_stride)
        self.assertEqual(detected.source, semantic.output.source)
        row = {"detections": [{"type": "vehicle", "bboxes": _bboxes(1, 1, 1, 2, 2, 3)}]}
        self.assertEqual(
            map_detection_track_count(row, *rewritten.output_expr.spec.args),
            map_vehicle_count(row),
        )


if __name__ == "__main__":
    unittest.main()
