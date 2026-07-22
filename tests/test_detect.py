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

from mmds import Detect, Input  # noqa: E402
from mmds.model import DatasetExpr, DetectSpec, MMDSValidationError  # noqa: E402
from mmds.execution.ops.detect import (  # noqa: E402
    _apply_detect,
    _detect_in_video,
    _resolve_video_source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(num_frames: int = 3, width: int = 64, height: int = 48):
    """Return a mock Video object that yields simple numpy frames."""
    from mmds.utilities.video import Video

    frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
    mock = MagicMock(spec=Video)
    mock.__iter__ = MagicMock(return_value=iter(frames))
    mock.num_frames = num_frames
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

    def test_default_frame_stride_is_one(self) -> None:
        spec = DetectSpec(video_field="v", classes=("dog",))
        self.assertEqual(spec.frame_stride, 1)

    def test_custom_frame_stride(self) -> None:
        spec = DetectSpec(video_field="v", classes=("dog",), frame_stride=3)
        self.assertEqual(spec.frame_stride, 3)

    def test_zero_frame_stride_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), frame_stride=0)

    def test_negative_frame_stride_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), frame_stride=-1)

    def test_non_int_frame_stride_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), frame_stride=1.5)  # type: ignore[arg-type]

    def test_bool_frame_stride_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), frame_stride=True)  # type: ignore[arg-type]

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

    def test_conf_and_imgsz_default_to_none(self) -> None:
        spec = DetectSpec(video_field="v", classes=("dog",))
        self.assertIsNone(spec.conf)
        self.assertIsNone(spec.imgsz)

    def test_custom_conf_and_imgsz(self) -> None:
        spec = DetectSpec(video_field="v", classes=("dog",), conf=0.1, imgsz=1280)
        self.assertEqual(spec.conf, 0.1)
        self.assertEqual(spec.imgsz, 1280)

    def test_conf_bounds_are_inclusive(self) -> None:
        self.assertEqual(DetectSpec(video_field="v", classes=("d",), conf=0.0).conf, 0.0)
        self.assertEqual(DetectSpec(video_field="v", classes=("d",), conf=1.0).conf, 1.0)

    def test_conf_below_zero_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), conf=-0.1)

    def test_conf_above_one_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), conf=1.1)

    def test_conf_bool_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), conf=True)  # type: ignore[arg-type]

    def test_conf_non_number_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), conf="0.1")  # type: ignore[arg-type]

    def test_imgsz_zero_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), imgsz=0)

    def test_imgsz_negative_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), imgsz=-1)

    def test_imgsz_bool_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), imgsz=True)  # type: ignore[arg-type]

    def test_imgsz_float_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DetectSpec(video_field="v", classes=("dog",), imgsz=1280.0)  # type: ignore[arg-type]


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

    def test_default_frame_stride_is_one(self) -> None:
        node = Detect(self._source(), "clip", ["dog"])
        spec = node.spec
        assert isinstance(spec, DetectSpec)
        self.assertEqual(spec.frame_stride, 1)

    def test_custom_frame_stride_is_forwarded(self) -> None:
        node = Detect(self._source(), "clip", ["dog"], frame_stride=4)
        spec = node.spec
        assert isinstance(spec, DetectSpec)
        self.assertEqual(spec.frame_stride, 4)

    def test_zero_frame_stride_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], frame_stride=0)

    def test_negative_frame_stride_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], frame_stride=-2)

    def test_bool_frame_stride_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], frame_stride=True)  # type: ignore[arg-type]

    def test_conf_and_imgsz_default_to_none(self) -> None:
        node = Detect(self._source(), "clip", ["dog"])
        spec = node.spec
        assert isinstance(spec, DetectSpec)
        self.assertIsNone(spec.conf)
        self.assertIsNone(spec.imgsz)

    def test_conf_and_imgsz_are_forwarded(self) -> None:
        node = Detect(self._source(), "clip", ["dog"], conf=0.1, imgsz=1280)
        spec = node.spec
        assert isinstance(spec, DetectSpec)
        self.assertEqual(spec.conf, 0.1)
        self.assertEqual(spec.imgsz, 1280)

    def test_out_of_range_conf_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], conf=1.5)

    def test_bool_conf_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], conf=True)  # type: ignore[arg-type]

    def test_non_positive_imgsz_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], imgsz=0)

    def test_bool_imgsz_raises(self) -> None:
        with self.assertRaises(TypeError):
            Detect(self._source(), "clip", ["dog"], imgsz=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_video_source
# ---------------------------------------------------------------------------


class ResolveVideoSourceTests(unittest.TestCase):
    def test_plain_string(self) -> None:
        self.assertEqual(_resolve_video_source("/data/clip.mp4"), "/data/clip.mp4")

    def test_dict_with_source_key(self) -> None:
        self.assertEqual(
            _resolve_video_source({"source": "s3://bucket/v.mp4"}), "s3://bucket/v.mp4"
        )

    def test_dict_with_path_key(self) -> None:
        self.assertEqual(
            _resolve_video_source({"path": "/local/v.mp4"}), "/local/v.mp4"
        )

    def test_dict_with_uri_key(self) -> None:
        self.assertEqual(
            _resolve_video_source({"uri": "https://yt.com/v"}), "https://yt.com/v"
        )

    def test_dict_prefers_source_over_path(self) -> None:
        self.assertEqual(
            _resolve_video_source({"source": "s", "path": "p"}),
            "s",
        )

    def test_dict_without_known_keys_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            _resolve_video_source({"url": "https://example.com/v.mp4"})

    def test_int_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            _resolve_video_source(42)

    def test_none_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            _resolve_video_source(None)

    def test_videoview_dict_extracts_source(self) -> None:
        """VideoView dicts with start/end still resolve to the source string."""
        self.assertEqual(
            _resolve_video_source(
                {
                    "type": "VideoView",
                    "source": "https://www.youtube.com/watch?v=abc",
                    "start": 0,
                    "end": 19,
                }
            ),
            "https://www.youtube.com/watch?v=abc",
        )


# ---------------------------------------------------------------------------
# _detect_in_video
# ---------------------------------------------------------------------------


class DetectInVideoTests(unittest.TestCase):
    def _run(self, frames, results_per_frame, classes=("dog",)):
        video = _make_video(num_frames=len(frames))
        video.__iter__ = MagicMock(return_value=iter(frames))

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
        video = _make_video(num_frames=1)
        video.__iter__ = MagicMock(return_value=iter([frame]))
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

    def test_frame_stride_skips_predict_on_non_sampled_frames(self) -> None:
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)]
        video = _make_video(num_frames=len(frames))
        video.__iter__ = MagicMock(return_value=iter(frames))

        results = [
            _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.9),
            _make_yoloe_result(0, "dog", [1, 1, 11, 11], 0.8),
        ]
        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.side_effect = [[results[0]], [results[1]]]

        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            detections = _detect_in_video(video, ["dog"], "yoloe-11s-seg.pt", frame_stride=2)

        # Only frames 0 and 2 (relative) should be sent through the model.
        self.assertEqual(mock_model.predict.call_count, 2)
        bboxes = detections[0]["bboxes"]
        self.assertEqual([b["frame_idx"] for b in bboxes], [0, 2])

    def test_frame_stride_one_matches_default_behavior(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = _make_yoloe_result(0, "dog", [10.0, 20.0, 50.0, 60.0], 0.9)
        detections = self._run([frame], [result])
        strided = self._run([frame], [result])
        self.assertEqual(detections, strided)

    def test_zero_frame_stride_raises(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        video = _make_video(num_frames=1)
        video.__iter__ = MagicMock(return_value=iter([frame]))
        mock_model = MagicMock()
        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            with self.assertRaises(MMDSValidationError):
                _detect_in_video(video, ["dog"], "yoloe-11s-seg.pt", frame_stride=0)

    def _predict_kwargs(self, **detect_kwargs) -> dict:
        """Run one frame through the model and return the kwargs predict saw."""
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        video = _make_video(num_frames=1)
        video.__iter__ = MagicMock(return_value=iter([frame]))
        result = _make_yoloe_result(0, "dog", [0, 0, 10, 10], 0.9)
        mock_model = MagicMock()
        mock_model.get_text_pe.return_value = MagicMock()
        mock_model.predict.return_value = [result]

        with patch("mmds.execution.ops.detect._get_model", return_value=mock_model):
            _detect_in_video(video, ["dog"], "yoloe-11s-seg.pt", **detect_kwargs)
        return mock_model.predict.call_args.kwargs

    def test_conf_and_imgsz_forwarded_to_predict(self) -> None:
        kwargs = self._predict_kwargs(conf=0.1, imgsz=1280)
        self.assertEqual(kwargs["conf"], 0.1)
        self.assertEqual(kwargs["imgsz"], 1280)

    def test_unset_conf_and_imgsz_not_forwarded(self) -> None:
        # When unset, the model's own defaults must apply — we must NOT pin
        # conf/imgsz to None, which Ultralytics would reject.
        kwargs = self._predict_kwargs()
        self.assertNotIn("conf", kwargs)
        self.assertNotIn("imgsz", kwargs)

    def test_only_conf_forwarded_when_imgsz_unset(self) -> None:
        kwargs = self._predict_kwargs(conf=0.25)
        self.assertEqual(kwargs["conf"], 0.25)
        self.assertNotIn("imgsz", kwargs)


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

    def test_nms_vehicle_detections_keeps_highest_confidence_overlap(self) -> None:
        from udfs.detection_ops import nms_vehicle_detections

        row = {
            "detections": [
                {
                    "type": "suv",
                    "bboxes": [
                        {
                            "frame_idx": 10,
                            "bbox": [0.0, 0.0, 100.0, 100.0],
                            "confidence": 0.9,
                        },
                        {
                            "frame_idx": 10,
                            "bbox": [5.0, 5.0, 95.0, 95.0],
                            "confidence": 0.4,
                        },
                    ],
                }
            ]
        }
        nmsed = nms_vehicle_detections(row)
        boxes = nmsed["detections"][0]["bboxes"]
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0]["confidence"], 0.9)

    def test_nms_vehicle_detections_suppresses_across_classes(self) -> None:
        """Overlapping boxes of different classes (YOLOE label flicker) collapse
        to the single highest-confidence box, keeping its class."""
        from udfs.detection_ops import nms_vehicle_detections

        row = {
            "detections": [
                {
                    "type": "sedan",
                    "bboxes": [
                        {"frame_idx": 7, "bbox": [0.0, 0.0, 100.0, 100.0], "confidence": 0.55},
                    ],
                },
                {
                    "type": "suv",
                    "bboxes": [
                        {"frame_idx": 7, "bbox": [4.0, 4.0, 104.0, 104.0], "confidence": 0.80},
                    ],
                },
            ]
        }
        nmsed = nms_vehicle_detections(row)
        # Only one box should survive, and it should be the suv (higher conf).
        total = sum(len(g["bboxes"]) for g in nmsed["detections"])
        self.assertEqual(total, 1)
        self.assertEqual(nmsed["detections"][0]["type"], "suv")
        self.assertAlmostEqual(nmsed["detections"][0]["bboxes"][0]["confidence"], 0.80)
        # The transient class-carrying key must not leak into output boxes.
        self.assertNotIn("_nms_vehicle_class", nmsed["detections"][0]["bboxes"][0])

    def test_nms_vehicle_detections_keeps_non_overlapping_across_classes(self) -> None:
        """Different-class boxes that do NOT overlap are both kept."""
        from udfs.detection_ops import nms_vehicle_detections

        row = {
            "detections": [
                {
                    "type": "sedan",
                    "bboxes": [
                        {"frame_idx": 7, "bbox": [0.0, 0.0, 50.0, 50.0], "confidence": 0.6},
                    ],
                },
                {
                    "type": "truck",
                    "bboxes": [
                        {"frame_idx": 7, "bbox": [500.0, 500.0, 600.0, 600.0], "confidence": 0.7},
                    ],
                },
            ]
        }
        nmsed = nms_vehicle_detections(row)
        types = {g["type"] for g in nmsed["detections"]}
        self.assertEqual(types, {"sedan", "truck"})

    def test_classify_vehicle_color_neutrals_by_brightness(self) -> None:
        from udfs.detection_ops import classify_vehicle_color

        self.assertEqual(classify_vehicle_color((10.0, 10.0, 10.0)), "black")
        self.assertEqual(classify_vehicle_color((110.0, 110.0, 110.0)), "gray")
        self.assertEqual(classify_vehicle_color((175.0, 175.0, 175.0)), "silver")
        self.assertEqual(classify_vehicle_color((245.0, 245.0, 245.0)), "white")

    def test_classify_vehicle_color_chromatic_by_hue(self) -> None:
        from udfs.detection_ops import classify_vehicle_color

        self.assertEqual(classify_vehicle_color((200.0, 20.0, 20.0)), "red")
        self.assertEqual(classify_vehicle_color((230.0, 220.0, 30.0)), "yellow")
        self.assertEqual(classify_vehicle_color((30.0, 160.0, 60.0)), "green")
        self.assertEqual(classify_vehicle_color((30.0, 60.0, 200.0)), "blue")

    def test_classify_vehicle_color_dark_saturated_is_black(self) -> None:
        from udfs.detection_ops import classify_vehicle_color

        # Low brightness overrides a noisy hue.
        self.assertEqual(classify_vehicle_color((20.0, 8.0, 8.0)), "black")

    def test_classify_vehicle_color_outputs_are_in_vocab(self) -> None:
        from udfs.detection_ops import VEHICLE_COLOR_VOCAB, classify_vehicle_color

        for rgb in [
            (0, 0, 0), (128, 128, 128), (255, 255, 255), (200, 20, 20),
            (230, 220, 30), (30, 160, 60), (30, 60, 200), (120, 80, 40),
        ]:
            self.assertIn(classify_vehicle_color(rgb), VEHICLE_COLOR_VOCAB)

    def test_dominant_rgb_uses_center_region(self) -> None:
        from udfs.detection_ops import _dominant_rgb_from_crop

        # Red border, blue center: the center-region median should be blue.
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        crop[:, :] = (0, 0, 200)  # BGR red everywhere
        crop[12:38, 15:35] = (200, 0, 0)  # BGR blue in the central body region
        r, g, b = _dominant_rgb_from_crop(crop)
        self.assertGreater(b, r)  # blue dominates the sampled center

    def test_dominant_rgb_none_on_empty(self) -> None:
        from udfs.detection_ops import _dominant_rgb_from_crop

        self.assertIsNone(_dominant_rgb_from_crop(None))
        self.assertIsNone(_dominant_rgb_from_crop(np.zeros((0, 0, 3), dtype=np.uint8)))

    def test_build_vehicle_frame_detections_shape(self) -> None:
        from udfs.detection_ops import build_vehicle_frame_detections

        row = {
            "camera_id": "cam-i24v-highway2",
            "detections": [
                {
                    "type": "sedan",
                    "bboxes": [
                        {
                            "frame_idx": 3,
                            "bbox": [10.0, 20.0, 110.0, 80.0],
                            "confidence": 0.82,
                        }
                    ],
                }
            ],
        }
        result = build_vehicle_frame_detections(row)
        self.assertEqual(len(result["frame_detections"]), 1)
        detection = result["frame_detections"][0]
        self.assertEqual(detection["frame_id"], 3)
        self.assertEqual(detection["camera_id"], "cam-i24v-highway2")
        self.assertEqual(detection["vehicle_class"], "sedan")
        self.assertEqual(detection["bbox"], [10.0, 20.0, 110.0, 80.0])
        self.assertAlmostEqual(detection["confidence"], 0.82)
        self.assertIn(
            detection["color"],
            {
                "white",
                "silver",
                "gray",
                "black",
                "beige",
                "yellow",
                "red",
                "blue",
                "green",
                "brown",
            },
        )
        self.assertIn(
            detection["subtype"],
            {"hatchback", "pickup", "sedan", "coupe", "suv"},
        )


if __name__ == "__main__":
    unittest.main()
