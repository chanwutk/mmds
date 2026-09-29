"""Tests for udfs.detection_ops and mmds.utilities.media.resolve_video_source.

Carried over from the original cross-camera join PR (#4).
"""

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

import numpy as np  # noqa: E402

from mmds.model import MMDSValidationError  # noqa: E402
from mmds.utilities.media import resolve_video_source  # noqa: E402


class ResolveVideoSourceTests(unittest.TestCase):
    def test_plain_string(self) -> None:
        self.assertEqual(resolve_video_source("/data/clip.mp4"), "/data/clip.mp4")

    def test_dict_with_source_key(self) -> None:
        self.assertEqual(
            resolve_video_source({"source": "s3://bucket/v.mp4"}), "s3://bucket/v.mp4"
        )

    def test_dict_with_path_key(self) -> None:
        self.assertEqual(
            resolve_video_source({"path": "/local/v.mp4"}), "/local/v.mp4"
        )

    def test_dict_with_uri_key(self) -> None:
        self.assertEqual(
            resolve_video_source({"uri": "https://yt.com/v"}), "https://yt.com/v"
        )

    def test_dict_prefers_source_over_path(self) -> None:
        self.assertEqual(
            resolve_video_source({"source": "s", "path": "p"}),
            "s",
        )

    def test_dict_without_known_keys_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            resolve_video_source({"url": "https://example.com/v.mp4"})

    def test_int_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            resolve_video_source(42)

    def test_none_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            resolve_video_source(None)

    def test_videoview_dict_extracts_source(self) -> None:
        """VideoView dicts with start/end still resolve to the source string."""
        self.assertEqual(
            resolve_video_source(
                {
                    "type": "VideoView",
                    "source": "https://www.youtube.com/watch?v=abc",
                    "start": 0,
                    "end": 19,
                }
            ),
            "https://www.youtube.com/watch?v=abc",
        )


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
        from unittest.mock import patch

        import numpy as np

        from udfs.detection_ops import build_vehicle_frame_detections

        row = {
            "camera_id": "cam-i24v-highway2",
            "video": "/tmp/clip.mp4",
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
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        with (
            patch(
                "udfs.detection_ops._video_path_from_row",
                return_value="/tmp/clip.mp4",
            ),
            patch(
                "mmds.utilities.video.read_frames_at_indices",
                return_value={3: frame},
            ),
        ):
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

    def test_build_vehicle_frame_detections_requires_openable_video(self) -> None:
        from udfs.detection_ops import build_vehicle_frame_detections

        row = {
            "camera_id": "cam",
            "detections": [
                {
                    "type": "sedan",
                    "bboxes": [
                        {
                            "frame_idx": 0,
                            "bbox": [0.0, 0.0, 10.0, 10.0],
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(MMDSValidationError):
            build_vehicle_frame_detections(row)


class DetectionFpsFallbackTests(unittest.TestCase):
    def _fps(self, row):
        from udfs.detection_ops import _resolve_detection_fps

        return _resolve_detection_fps(row)

    def test_prefers_the_fps_written_by_detect(self) -> None:
        row = {"_mmds_video_fps": 25.0, "video": {"path": "a.mp4", "fps": 10}}
        self.assertEqual(self._fps(row), 25.0)

    def test_falls_back_to_a_video_shaped_field(self) -> None:
        for key in ("source", "path", "uri"):
            with self.subTest(key=key):
                self.assertEqual(self._fps({"video": {key: "a.mp4", "fps": 12}}), 12.0)

    def test_ignores_non_video_dicts_with_an_fps_key(self) -> None:
        row = {"config": {"fps": 1}, "video": {"path": "a.mp4", "fps": 24}}
        self.assertEqual(self._fps(row), 24.0)
        self.assertEqual(self._fps({"config": {"fps": 1}}), 30.0)

    def test_invalid_values_fall_back_to_default(self) -> None:
        for row in ({"_mmds_video_fps": 0}, {"_mmds_video_fps": True}, {"video": {"path": "a", "fps": -1}}):
            with self.subTest(row=row):
                self.assertEqual(self._fps(row), 30.0)


if __name__ == "__main__":
    unittest.main()
