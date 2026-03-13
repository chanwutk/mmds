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


if __name__ == "__main__":
    unittest.main()
