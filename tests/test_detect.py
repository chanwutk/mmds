from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from mmds import Detect, Input, parse_query, render_query  # noqa: E402
from mmds.model import DatasetExpr, DetectSpec, MMDSValidationError  # noqa: E402
from mmds.execution.ops.detect import (  # noqa: E402
    _apply_detect,
    _detect_in_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(num_frames: int = 3, width: int = 64, height: int = 48, frames=None):
    """Return a mock Video that yields ``frames`` (blank ones by default)."""
    from mmds.utilities.video import Video

    if frames is None:
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
    mock = MagicMock(spec=Video)
    mock.__iter__ = MagicMock(side_effect=lambda: iter(frames))
    mock.sampled = MagicMock(
        side_effect=lambda stride: ((i, f) for i, f in enumerate(frames) if i % stride == 0)
    )
    mock.num_frames = len(frames)
    mock.width = width
    mock.height = height
    mock.fps = 30.0
    return mock


def _make_yoloe_result(class_id: int, class_name: str, bbox, conf: float):
    """Build a minimal fake ultralytics result object."""
    box = MagicMock()
    box.xyxy.__getitem__ = lambda self, i: MagicMock(tolist=lambda: bbox)
    box.conf.__getitem__ = lambda self, i: conf
    box.cls.__getitem__ = lambda self, i: class_id

    boxes = MagicMock()
    boxes.__len__ = lambda self: 1
    boxes.xyxy = MagicMock()
    boxes.xyxy.__getitem__ = lambda self, i: MagicMock(tolist=lambda: bbox)
    boxes.conf = MagicMock()
    boxes.conf.__getitem__ = lambda self, i: conf
    boxes.cls = MagicMock()
    boxes.cls.__getitem__ = lambda self, i: class_id

    result = MagicMock()
    result.boxes = boxes
    result.names = {class_id: class_name}
    return result


def _make_detect_node(
    video_field: str = "video",
    classes: tuple[str, ...] = ("dog",),
    model: str = "yoloe-11s-seg.pt",
    output_field: str = "detections",
) -> DatasetExpr:
    source = DatasetExpr(kind="input", input_path="dummy.jsonl")
    return DatasetExpr(
        kind="detect",
        source=source,
        spec=DetectSpec(
            video_field=video_field,
            classes=classes,
            model=model,
            output_field=output_field,
        ),
    )


# ---------------------------------------------------------------------------
# DetectSpec construction
# ---------------------------------------------------------------------------


class DetectSpecTests(unittest.TestCase):
    def test_valid_spec(self) -> None:
        spec = DetectSpec(video_field="v", classes=("dog", "cat"))
        self.assertEqual(spec.video_field, "v")
        self.assertEqual(spec.classes, ("dog", "cat"))
        self.assertEqual(spec.model, "yoloe-11s-seg.pt")
        self.assertEqual(spec.output_field, "detections")

    def test_custom_model_and_output_field(self) -> None:
        spec = DetectSpec(
            video_field="v",
            classes=("person",),
            model="yoloe-s.pt",
            output_field="hits",
        )
        self.assertEqual(spec.model, "yoloe-s.pt")
        self.assertEqual(spec.output_field, "hits")

    def test_empty_video_field_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="", classes=("dog",))

    def test_empty_classes_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=())

    def test_blank_class_name_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog", ""))

    def test_empty_model_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), model="")

    def test_empty_output_field_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), output_field="")


# ---------------------------------------------------------------------------
# Detect() DSL function
# ---------------------------------------------------------------------------


class DetectDSLTests(unittest.TestCase):
    def _source(self) -> DatasetExpr:
        return DatasetExpr(kind="input", input_path="data.jsonl")

    def test_returns_detect_dataset_expr(self) -> None:
        node = Detect(self._source(), "clip", ["dog"])
        self.assertIsInstance(node, DatasetExpr)
        self.assertEqual(node.kind, "detect")

    def test_spec_fields_are_set(self) -> None:
        node = Detect(
            self._source(), "clip", ["dog", "cat"], model="m.pt", output_field="hits"
        )
        spec = node.spec
        self.assertIsInstance(spec, DetectSpec)
        assert isinstance(spec, DetectSpec)
        self.assertEqual(spec.video_field, "clip")
        self.assertEqual(spec.classes, ("dog", "cat"))
        self.assertEqual(spec.model, "m.pt")
        self.assertEqual(spec.output_field, "hits")

    def test_name_is_forwarded(self) -> None:
        node = Detect(self._source(), "clip", ["dog"], name="step_detect")
        self.assertEqual(node.name, "step_detect")

    def test_empty_video_field_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "", ["dog"])

    def test_empty_classes_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", [])

    def test_blank_class_in_list_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog", ""])

    def test_non_string_source_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect("not-a-node", "clip", ["dog"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _detect_in_video
# ---------------------------------------------------------------------------


class DetectInVideoTests(unittest.TestCase):
    def _run(self, frames, results_per_frame, classes=("dog",)):
        video = _make_video(frames=frames)

        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.side_effect = [
            [r] if r is not None else [] for r in results_per_frame
        ]

        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            return _detect_in_video(video, list(classes), "yoloe-11s-seg.pt")

    def test_single_detection(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = _make_yoloe_result(0, "dog", [10.0, 20.0, 50.0, 60.0], 0.9)
        detections = self._run([frame], [result])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["type"], "dog")
        self.assertEqual(len(detections[0]["bboxes"]), 1)
        bbox_entry = detections[0]["bboxes"][0]
        self.assertEqual(bbox_entry["frame_idx"], 0)
        self.assertAlmostEqual(bbox_entry["confidence"], 0.9, places=4)

    def test_no_detections_returns_empty_list(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        empty_result = MagicMock()
        empty_result.boxes = None
        detections = self._run([frame], [empty_result])
        self.assertEqual(detections, [])

    def test_same_class_across_frames_grouped(self) -> None:
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
        r0 = _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.8)
        r1 = _make_yoloe_result(0, "dog", [5, 5, 15, 15], 0.7)
        detections = self._run(frames, [r0, r1])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["type"], "dog")
        self.assertEqual(len(detections[0]["bboxes"]), 2)
        self.assertEqual(detections[0]["bboxes"][0]["frame_idx"], 0)
        self.assertEqual(detections[0]["bboxes"][1]["frame_idx"], 1)

    def test_multiple_classes_separate_entries(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        dog_result = _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.9)
        cat_result = _make_yoloe_result(1, "cat", [20, 20, 40, 40], 0.8)

        # Two results in one frame
        video = _make_video(frames=[frame])
        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.return_value = [dog_result, cat_result]

        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            detections = _detect_in_video(video, ["dog", "cat"], "yoloe-11s-seg.pt")

        types = {d["type"] for d in detections}
        self.assertEqual(types, {"dog", "cat"})

    def test_videoview_reports_absolute_frame_indices(self) -> None:
        from mmds.utilities.video import Video, VideoView

        base_video = MagicMock(spec=Video)
        base_video.fps = 10.0
        base_video.num_frames = 100
        base_video.path = Path("/tmp/test.mp4")

        view = VideoView(base_video, start=2.0, end=4.0)

        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(2)]
        iter_cap = MagicMock()
        iter_cap.read.side_effect = [(True, frame) for frame in frames] + [
            (False, None)
        ]

        r0 = _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.8)
        r1 = _make_yoloe_result(0, "dog", [5, 5, 15, 15], 0.7)

        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.side_effect = [[r0], [r1]]

        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=iter_cap):
            with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
                detections = _detect_in_video(view, ["dog"], "yoloe-11s-seg.pt")

        self.assertEqual(detections[0]["bboxes"][0]["frame_idx"], 20)
        self.assertEqual(detections[0]["bboxes"][1]["frame_idx"], 21)


# ---------------------------------------------------------------------------
# _apply_detect (full row integration)
# ---------------------------------------------------------------------------


class ApplyDetectTests(unittest.TestCase):
    def test_merges_detections_into_row(self) -> None:
        node = _make_detect_node()
        row = {"video": "/data/clip.mp4", "title": "test"}
        video = _make_video()
        mock_detections = [
            {
                "type": "dog",
                "bboxes": [{"frame_idx": 0, "bbox": [0, 0, 10, 10], "confidence": 0.9}],
            }
        ]

        with patch("mmds.execution.ops.detect.open_video", return_value=video):
            with patch(
                "mmds.execution.ops.detect._detect_in_video",
                return_value=mock_detections,
            ):
                result = _apply_detect(node, row)

        self.assertEqual(result["title"], "test")
        self.assertEqual(result["video"], "/data/clip.mp4")
        self.assertEqual(result["detections"], mock_detections)

    def test_custom_output_field(self) -> None:
        node = _make_detect_node(output_field="hits")
        row = {"video": "clip.mp4"}
        video = _make_video()

        with patch("mmds.execution.ops.detect.open_video", return_value=video):
            with patch("mmds.execution.ops.detect._detect_in_video", return_value=[]):
                result = _apply_detect(node, row)

        self.assertIn("hits", result)
        self.assertNotIn("detections", result)

    def test_missing_video_field_raises(self) -> None:
        node = _make_detect_node(video_field="clip")
        with self.assertRaises(MMDSValidationError):
            _apply_detect(node, {"other": "value"})

    def test_dict_video_field_is_resolved(self) -> None:
        node = _make_detect_node()
        row = {"video": {"source": "/data/clip.mp4"}}
        video = _make_video()

        with patch(
            "mmds.execution.ops.detect.open_video", return_value=video
        ) as mock_open:
            with patch("mmds.execution.ops.detect._detect_in_video", return_value=[]):
                _apply_detect(node, row)

        mock_open.assert_called_once_with("/data/clip.mp4")

    def test_directory_result_raises(self) -> None:
        node = _make_detect_node()
        row = {"video": "/data/videos/"}

        with patch(
            "mmds.execution.ops.detect.open_video",
            return_value=[MagicMock(), MagicMock()],
        ):
            with self.assertRaises(MMDSValidationError):
                _apply_detect(node, row)

    def test_original_row_is_not_mutated(self) -> None:
        node = _make_detect_node()
        row = {"video": "clip.mp4", "x": 1}
        original = dict(row)
        video = _make_video()

        with patch("mmds.execution.ops.detect.open_video", return_value=video):
            with patch("mmds.execution.ops.detect._detect_in_video", return_value=[]):
                _apply_detect(node, row)

        self.assertEqual(row, original)

    def test_videoview_dict_creates_videoview(self) -> None:
        """A dict with start/end should wrap the Video in a VideoView."""
        from mmds.utilities.video import VideoView

        node = _make_detect_node()
        row = {
            "video": {
                "type": "VideoView",
                "source": "/data/clip.mp4",
                "start": 10,
                "end": 20,
            }
        }
        video = _make_video(num_frames=900)
        video.fps = 30.0

        with patch("mmds.execution.ops.detect.open_video", return_value=video):
            with patch(
                "mmds.execution.ops.detect._detect_in_video", return_value=[]
            ) as mock_detect:
                _apply_detect(node, row)

        # The first argument to _detect_in_video should be a VideoView.
        called_video = mock_detect.call_args[0][0]
        self.assertIsInstance(called_video, VideoView)
        self.assertEqual(called_video.start_frame, 300)  # 10s * 30fps
        self.assertEqual(called_video.end_frame, 600)  # 20s * 30fps

    def test_plain_string_does_not_wrap_in_videoview(self) -> None:
        """A plain string video field should pass the Video directly."""
        from mmds.utilities.video import Video, VideoView

        node = _make_detect_node()
        row = {"video": "/data/clip.mp4"}
        video = _make_video()

        with patch("mmds.execution.ops.detect.open_video", return_value=video):
            with patch(
                "mmds.execution.ops.detect._detect_in_video", return_value=[]
            ) as mock_detect:
                _apply_detect(node, row)

        called_video = mock_detect.call_args[0][0]
        self.assertNotIsInstance(called_video, VideoView)

    def test_dict_without_start_end_does_not_wrap(self) -> None:
        """A dict without start/end should pass the Video directly."""
        from mmds.utilities.video import VideoView

        node = _make_detect_node()
        row = {"video": {"source": "/data/clip.mp4"}}
        video = _make_video()

        with patch("mmds.execution.ops.detect.open_video", return_value=video):
            with patch(
                "mmds.execution.ops.detect._detect_in_video", return_value=[]
            ) as mock_detect:
                _apply_detect(node, row)

        called_video = mock_detect.call_args[0][0]
        self.assertNotIsInstance(called_video, VideoView)


class DetectionOpsTests(unittest.TestCase):
    def test_keep_rows_with_high_confidence_bear(self) -> None:
        from udfs.detection_ops import (
            HIGH_CONFIDENCE_THRESHOLD,
            keep_rows_with_class,
            keep_rows_with_high_confidence_class,
            prune_detections,
            prune_class_detections_to_high_confidence,
        )

        low_only = {
            "detections": [
                {
                    "type": "bear",
                    "bboxes": [{"frame_idx": 1, "bbox": [0, 0, 1, 1], "confidence": 0.3}],
                }
            ]
        }
        high = {
            "detections": [
                {
                    "type": "bear",
                    "bboxes": [
                        {
                            "frame_idx": 2,
                            "bbox": [0, 0, 1, 1],
                            "confidence": HIGH_CONFIDENCE_THRESHOLD,
                        }
                    ],
                }
            ]
        }

        self.assertTrue(keep_rows_with_class(low_only, "bear"))
        self.assertFalse(keep_rows_with_high_confidence_class(low_only, "bear"))
        self.assertTrue(keep_rows_with_high_confidence_class(high, "bear"))

        pruned_low = prune_class_detections_to_high_confidence(low_only, "bear")
        self.assertEqual(pruned_low["detections"], [])
        self.assertFalse(keep_rows_with_high_confidence_class(pruned_low, "bear"))

        pruned_high = prune_class_detections_to_high_confidence(high, "bear")
        self.assertTrue(keep_rows_with_high_confidence_class(pruned_high, "bear"))
        self.assertEqual(pruned_high["detections"][0]["type"], "bear")
        self.assertEqual(len(pruned_high["detections"][0]["bboxes"]), 1)
        self.assertGreaterEqual(
            pruned_high["detections"][0]["bboxes"][0]["confidence"],
            HIGH_CONFIDENCE_THRESHOLD,
        )

        pruned_high_generic = prune_detections(
            high,
            min_confidence=HIGH_CONFIDENCE_THRESHOLD,
            only_classes={"bear"},
            keep_other_classes=True,
        )
        self.assertEqual(pruned_high_generic, pruned_high)


def _query(body: str, imports: str) -> str:
    return f'from mmds import {imports}\n\nrows = Input("rows.jsonl")\n{body}\n'


class DetectParseRenderTests(unittest.TestCase):
    def _parse(self, body: str):
        return parse_query(_query(body, "Input, Detect"))

    def test_round_trips_all_keyword_arguments(self) -> None:
        program = self._parse(
            'output = Detect(rows, "video", ["dog", "cat"], model="m.pt", '
            'output_field="hits", frame_stride=5, conf=0.4, imgsz=640, '
            'stop_after_n=1, name="det")'
        )
        spec = program.output_expr.spec

        self.assertEqual(spec.classes, ("dog", "cat"))
        self.assertEqual(spec.model, "m.pt")
        self.assertEqual(spec.output_field, "hits")
        self.assertEqual(spec.frame_stride, 5)
        self.assertEqual(spec.conf, 0.4)
        self.assertEqual(spec.imgsz, 640)
        self.assertEqual(spec.stop_after_n, 1)
        self.assertEqual(program.output_expr.name, "det")
        rendered = render_query(program)
        self.assertEqual(parse_query(rendered).output_expr, program.output_expr)
        self.assertEqual(render_query(parse_query(rendered)), rendered)

    def test_defaults_are_omitted_when_rendering(self) -> None:
        program = self._parse('output = Detect(rows, "video", ["dog"])')
        spec = program.output_expr.spec

        self.assertEqual(
            (spec.frame_stride, spec.conf, spec.imgsz, spec.stop_after_n),
            (1, None, None, None),
        )
        self.assertIn('output = Detect(rows, "video", ["dog"])', render_query(program))

    def test_explicit_none_conf_imgsz_and_stop_after_n_are_accepted(self) -> None:
        spec = self._parse(
            'output = Detect(rows, "video", ["dog"], conf=None, imgsz=None, '
            "stop_after_n=None)"
        ).output_expr.spec

        self.assertEqual((spec.conf, spec.imgsz, spec.stop_after_n), (None, None, None))

    def test_stop_after_n_is_rendered_only_when_set(self) -> None:
        program = self._parse('output = Detect(rows, "video", ["dog"], stop_after_n=1)')

        self.assertIn(
            'output = Detect(rows, "video", ["dog"], stop_after_n=1)',
            render_query(program),
        )

    def test_rejects_invalid_arguments(self) -> None:
        cases = {
            "frame_stride zero": 'Detect(rows, "video", ["dog"], frame_stride=0)',
            "frame_stride float": 'Detect(rows, "video", ["dog"], frame_stride=1.5)',
            "frame_stride bool": 'Detect(rows, "video", ["dog"], frame_stride=True)',
            "conf above one": 'Detect(rows, "video", ["dog"], conf=1.5)',
            "conf negative": 'Detect(rows, "video", ["dog"], conf=-0.1)',
            "conf bool": 'Detect(rows, "video", ["dog"], conf=True)',
            "conf string": 'Detect(rows, "video", ["dog"], conf="0.5")',
            "imgsz zero": 'Detect(rows, "video", ["dog"], imgsz=0)',
            "imgsz float": 'Detect(rows, "video", ["dog"], imgsz=640.0)',
            "stop_after_n zero": 'Detect(rows, "video", ["dog"], stop_after_n=0)',
            "stop_after_n two": 'Detect(rows, "video", ["dog"], stop_after_n=2)',
            "stop_after_n float": 'Detect(rows, "video", ["dog"], stop_after_n=1.0)',
            "stop_after_n bool": 'Detect(rows, "video", ["dog"], stop_after_n=True)',
            "empty classes": 'Detect(rows, "video", [])',
            "non-literal classes": 'Detect(rows, "video", classes)',
            "unknown keyword": 'Detect(rows, "video", ["dog"], fps=3)',
            "missing classes": 'Detect(rows, "video")',
        }
        for label, call in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(MMDSValidationError):
                    self._parse(f"output = {call}")

    def test_requires_detect_import(self) -> None:
        with self.assertRaisesRegex(MMDSValidationError, "Detect must be imported"):
            parse_query(_query('output = Detect(rows, "video", ["dog"])', "Input"))


class DetectParamValidationTests(unittest.TestCase):
    def test_dsl_rejects_invalid_params(self) -> None:
        cases = {
            "frame_stride zero": {"frame_stride": 0},
            "frame_stride bool": {"frame_stride": True},
            "conf above one": {"conf": 1.01},
            "conf nan": {"conf": float("nan")},
            "conf bool": {"conf": False},
            "imgsz zero": {"imgsz": 0},
            "imgsz float": {"imgsz": 32.0},
            "stop_after_n zero": {"stop_after_n": 0},
            "stop_after_n two": {"stop_after_n": 2},
            "stop_after_n float": {"stop_after_n": 1.0},
            "stop_after_n bool": {"stop_after_n": True},
        }
        for label, kwargs in cases.items():
            with self.subTest(case=label):
                with self.assertRaises((TypeError, MMDSValidationError)):
                    Detect(Input("clips.jsonl"), "video", ["dog"], **kwargs)

    def test_spec_accepts_boundary_values(self) -> None:
        for conf in (0.0, 1.0):
            with self.subTest(conf=conf):
                spec = DetectSpec(video_field="video", classes=("dog",), conf=conf)
                self.assertEqual(spec.conf, conf)
        spec = DetectSpec(video_field="video", classes=("dog",), frame_stride=1, imgsz=1)
        self.assertEqual((spec.frame_stride, spec.imgsz), (1, 1))

    def test_spec_stop_after_n_accepts_only_none_and_one(self) -> None:
        for value in (None, 1):
            with self.subTest(value=value):
                spec = DetectSpec(video_field="video", classes=("dog",), stop_after_n=value)
                self.assertEqual(spec.stop_after_n, value)
        for value in (0, 2, -1, True, 1.0, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(
                MMDSValidationError, "stop_after_n must be 1 or None"
            ):
                DetectSpec(video_field="video", classes=("dog",), stop_after_n=value)

    def test_dsl_passes_stop_after_n_to_the_spec(self) -> None:
        plan = Detect(Input("clips.jsonl"), "video", ["dog"], stop_after_n=1)

        self.assertEqual(plan.spec.stop_after_n, 1)


class DetectParamExecutionTests(unittest.TestCase):
    def _run(self, num_frames: int, **kwargs):
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(num_frames)]
        video = _make_video(frames=frames)
        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.side_effect = lambda frame, **_: [
            _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.9)
        ]
        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            detections = _detect_in_video(video, ["dog"], "yoloe-11s-seg.pt", **kwargs)
        return detections, mock_model

    def test_frame_stride_runs_every_nth_frame_with_absolute_indices(self) -> None:
        detections, model = self._run(7, frame_stride=3)

        self.assertEqual(model.predict.call_count, 3)
        self.assertEqual(
            [bbox["frame_idx"] for bbox in detections[0]["bboxes"]],
            [0, 3, 6],
        )

    def test_frame_stride_larger_than_video_runs_first_frame_only(self) -> None:
        detections, model = self._run(3, frame_stride=10)

        self.assertEqual(model.predict.call_count, 1)
        self.assertEqual([bbox["frame_idx"] for bbox in detections[0]["bboxes"]], [0])

    def test_conf_and_imgsz_are_forwarded_to_predict(self) -> None:
        _, model = self._run(1, conf=0.25, imgsz=320)

        kwargs = model.predict.call_args.kwargs
        self.assertEqual(kwargs["conf"], 0.25)
        self.assertEqual(kwargs["imgsz"], 320)

    def test_unset_conf_and_imgsz_use_model_defaults(self) -> None:
        _, model = self._run(1)

        kwargs = model.predict.call_args.kwargs
        self.assertNotIn("conf", kwargs)
        self.assertNotIn("imgsz", kwargs)

    def test_invalid_frame_stride_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            self._run(1, frame_stride=0)

    def test_apply_detect_forwards_spec_params(self) -> None:
        source = DatasetExpr(kind="input", input_path="dummy.jsonl")
        node = DatasetExpr(
            kind="detect",
            source=source,
            spec=DetectSpec(
                video_field="video",
                classes=("dog",),
                frame_stride=4,
                conf=0.5,
                imgsz=480,
            ),
        )

        with patch("mmds.execution.ops.detect.open_video", return_value=_make_video()):
            with patch(
                "mmds.execution.ops.detect._detect_in_video", return_value=[]
            ) as detect:
                _apply_detect(node, {"video": "clip.mp4"})

        self.assertEqual(
            detect.call_args.kwargs,
            {"frame_stride": 4, "conf": 0.5, "imgsz": 480, "stop_after_n": None},
        )

    def test_apply_detect_forwards_stop_after_n(self) -> None:
        source = DatasetExpr(kind="input", input_path="dummy.jsonl")
        node = DatasetExpr(
            kind="detect",
            source=source,
            spec=DetectSpec(video_field="video", classes=("dog",), stop_after_n=1),
        )

        with patch("mmds.execution.ops.detect.open_video", return_value=_make_video()):
            with patch(
                "mmds.execution.ops.detect._detect_in_video", return_value=[]
            ) as detect:
                _apply_detect(node, {"video": "clip.mp4"})

        self.assertEqual(detect.call_args.kwargs["stop_after_n"], 1)


class SelectDeviceTests(unittest.TestCase):
    """``_select_device`` prefers CUDA, then MPS, and needs the probe to run."""

    @staticmethod
    def _fake_torch(*, cuda: bool, mps: bool, broken: tuple[str, ...] = ()):
        from types import SimpleNamespace

        class Conv:
            def __init__(self, *args) -> None:
                self.device = "cpu"

            def to(self, device: str) -> "Conv":
                self.device = device
                return self

            def __call__(self, tensor):
                if self.device in broken:
                    raise RuntimeError(f"no kernel image for {self.device}")
                return tensor

        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
            nn=SimpleNamespace(Conv2d=Conv),
            zeros=lambda *shape, device=None: object(),
        )

    def _select(self, **kwargs) -> str:
        from mmds.execution.ops.detect import _select_device

        with patch.dict(sys.modules, {"torch": self._fake_torch(**kwargs)}):
            return _select_device()

    def test_prefers_usable_cuda(self) -> None:
        self.assertEqual(self._select(cuda=True, mps=True), "cuda")

    def test_uses_mps_when_cuda_is_missing(self) -> None:
        self.assertEqual(self._select(cuda=False, mps=True), "mps")

    def test_uses_mps_when_cuda_cannot_run_the_probe(self) -> None:
        self.assertEqual(self._select(cuda=True, mps=True, broken=("cuda",)), "mps")

    def test_falls_back_to_cpu_when_no_accelerator_runs_the_probe(self) -> None:
        self.assertEqual(self._select(cuda=True, mps=True, broken=("cuda", "mps")), "cpu")
        self.assertEqual(self._select(cuda=False, mps=False), "cpu")

    def test_falls_back_to_cpu_without_torch(self) -> None:
        from mmds.execution.ops.detect import _select_device

        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(_select_device(), "cpu")

    def _class_setup_calls(self, device: str) -> list[str]:
        """Run one frame and return the model calls made while setting classes."""
        video = _make_video(num_frames=1)
        model = MagicMock()
        model.predict.return_value = []
        with patch("mmds.execution.ops.detect._get_model", return_value=model), patch(
            "mmds.execution.ops.detect._get_device", return_value=device
        ):
            _detect_in_video(video, ["dog"], "yoloe-11s-seg.pt")
        names = [c[0] for c in model.mock_calls]
        calls = [
            f"{name}({c[1][0]})" if name == "to" else name
            for name, c in zip(names, model.mock_calls)
            if name in ("to", "get_text_pe", "set_classes")
        ]
        return calls

    def test_mps_encodes_class_names_on_the_cpu(self) -> None:
        self.assertEqual(
            self._class_setup_calls("mps"),
            ["to(cpu)", "get_text_pe", "set_classes", "to(mps)"],
        )

    def test_cuda_and_cpu_encode_class_names_in_place(self) -> None:
        for device in ("cuda", "cpu"):
            with self.subTest(device=device):
                self.assertEqual(self._class_setup_calls(device), ["get_text_pe", "set_classes"])


class DetectEarlyStopTests(unittest.TestCase):
    """``stop_after_n=1`` must give the same yes/no answer as a full scan."""

    def _run(self, hits: list[int], **kwargs):
        """Scan frames where ``hits[i]`` is the number of dog boxes in frame ``i``."""
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in hits]
        video = _make_video(frames=frames)
        per_call = iter(hits[:: kwargs.get("frame_stride", 1)])

        def predict(frame, **_):
            count = next(per_call)
            return [
                _make_yoloe_result(0, "dog", [k, k, k + 10, k + 10], 0.9)
                for k in range(count)
            ]

        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.side_effect = predict
        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            detections = _detect_in_video(video, ["dog"], "yoloe-11s-seg.pt", **kwargs)
        return detections, mock_model

    @staticmethod
    def _frames(detections) -> list[int]:
        return [box["frame_idx"] for entry in detections for box in entry["bboxes"]]

    def test_stops_after_the_first_frame_with_a_box(self) -> None:
        detections, model = self._run([0, 0, 1, 1, 0], stop_after_n=1)

        self.assertEqual(model.predict.call_count, 3)
        self.assertEqual(self._frames(detections), [2])

    def test_keeps_every_box_in_the_stopping_frame(self) -> None:
        detections, _ = self._run([0, 3, 1], stop_after_n=1)

        self.assertEqual(self._frames(detections), [1, 1, 1])

    def test_video_without_boxes_is_scanned_to_the_end(self) -> None:
        detections, model = self._run([0, 0, 0, 0], stop_after_n=1)

        self.assertEqual(model.predict.call_count, 4)
        self.assertEqual(detections, [])

    def test_box_in_the_first_frame_stops_immediately(self) -> None:
        detections, model = self._run([1, 1, 1], stop_after_n=1)

        self.assertEqual(model.predict.call_count, 1)
        self.assertEqual(self._frames(detections), [0])

    def test_box_only_in_the_last_frame_is_found(self) -> None:
        detections, model = self._run([0, 0, 0, 2], stop_after_n=1)

        self.assertEqual(model.predict.call_count, 4)
        self.assertEqual(self._frames(detections), [3, 3])

    def test_matches_full_scan_for_every_hit_pattern(self) -> None:
        for mask in range(2**5):
            hits = [(mask >> i) & 1 for i in range(5)]
            with self.subTest(hits=hits):
                early, early_model = self._run(hits, stop_after_n=1)
                full, _ = self._run(hits)
                full_frames = self._frames(full)

                self.assertEqual(bool(early), bool(full))
                if full_frames:
                    first = full_frames[0]
                    self.assertEqual(
                        self._frames(early), [f for f in full_frames if f == first]
                    )
                    self.assertEqual(early_model.predict.call_count, first + 1)
                else:
                    self.assertEqual(early_model.predict.call_count, len(hits))

    def test_stride_skips_unsampled_frames_before_stopping(self) -> None:
        # Frame 1 has a box but is not sampled at stride 2; frame 4 is.
        detections, model = self._run([0, 1, 0, 0, 1, 0], frame_stride=2, stop_after_n=1)

        self.assertEqual(model.predict.call_count, 3)
        self.assertEqual(self._frames(detections), [4])

    def test_videoview_stop_reports_absolute_frame_indices(self) -> None:
        from mmds.utilities.video import Video, VideoView

        base_video = MagicMock(spec=Video)
        base_video.fps = 10.0
        base_video.num_frames = 100
        base_video.path = Path("/tmp/test.mp4")
        view = VideoView(base_video, start=2.0, end=2.5)

        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
        capture = MagicMock()
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        empty = MagicMock()
        empty.boxes = None
        hit = _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.8)
        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.side_effect = [[empty], [hit], [hit]]

        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=capture):
            with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
                detections = _detect_in_video(view, ["dog"], "yoloe-11s-seg.pt", stop_after_n=1)

        self.assertEqual(mock_model.predict.call_count, 2)
        self.assertEqual(self._frames(detections), [21])

    def test_invalid_stop_after_n_raises(self) -> None:
        for value in (0, 2):
            with self.subTest(value=value), self.assertRaises(MMDSValidationError):
                self._run([1], stop_after_n=value)


if __name__ == "__main__":
    unittest.main()
