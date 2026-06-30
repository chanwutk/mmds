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

from udfs.tracking_ops import (  # noqa: E402
    StrongSortTracker,
    project_track_summary_row,
    promote_track_summary_row,
    strongsort_track_frame_detections,
)


class StrongSortTrackerTests(unittest.TestCase):
    def test_assigns_stable_track_id_across_frames(self) -> None:
        detections = [
            {
                "frame_id": 1,
                "camera_id": "cam-a",
                "bbox": [10.0, 10.0, 50.0, 50.0],
                "confidence": 0.9,
                "vehicle_class": "suv",
                "color": "black",
                "subtype": "suv",
            },
            {
                "frame_id": 2,
                "camera_id": "cam-a",
                "bbox": [12.0, 12.0, 52.0, 52.0],
                "confidence": 0.85,
                "vehicle_class": "suv",
                "color": "black",
                "subtype": "suv",
            },
        ]
        tracker = StrongSortTracker(track_prefix="suv")
        tracked = tracker.update(detections)
        self.assertEqual(len(tracked), 2)
        self.assertEqual(tracked[0]["track_id"], tracked[1]["track_id"])

    def test_closes_track_after_frame_gap(self) -> None:
        detections = [
            {
                "frame_id": 1,
                "bbox": [10.0, 10.0, 50.0, 50.0],
                "confidence": 0.9,
                "vehicle_class": "sedan",
                "color": "white",
                "subtype": "sedan",
            },
            {
                "frame_id": 100,
                "bbox": [12.0, 12.0, 52.0, 52.0],
                "confidence": 0.8,
                "vehicle_class": "sedan",
                "color": "white",
                "subtype": "sedan",
            },
        ]
        tracker = StrongSortTracker(track_prefix="sedan", max_frame_gap=30)
        tracked = tracker.update(detections)
        self.assertNotEqual(tracked[0]["track_id"], tracked[1]["track_id"])


class TrackSummaryTests(unittest.TestCase):
    def test_track_summary_shape(self) -> None:
        row = {
            "camera_id": "cam-i24v-highway2",
            "fps": 30.0,
            "recorded_at": "2023-08-10T12:00:00Z",
            "video": {"start": 0, "end": 5},
            "frame_detections": [
                {
                    "frame_id": 10,
                    "camera_id": "cam-i24v-highway2",
                    "bbox": [100.0, 200.0, 200.0, 300.0],
                    "confidence": 0.9,
                    "vehicle_class": "suv",
                    "color": "black",
                    "subtype": "suv",
                },
                {
                    "frame_id": 11,
                    "camera_id": "cam-i24v-highway2",
                    "bbox": [110.0, 205.0, 210.0, 305.0],
                    "confidence": 0.88,
                    "vehicle_class": "suv",
                    "color": "black",
                    "subtype": "suv",
                },
            ],
        }
        result = strongsort_track_frame_detections(row)
        summaries = result["track_summaries"]
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["track_id"], "suv-1")
        self.assertEqual(summary["camera_id"], "cam-i24v-highway2")
        self.assertEqual(summary["vehicle_class"], "suv")
        self.assertEqual(summary["color"], "black")
        self.assertEqual(summary["subtype"], "suv")
        self.assertIn("start_time", summary)
        self.assertIn("end_time", summary)
        self.assertGreater(summary["avg_speed"], 0.0)
        self.assertIn(summary["entry_direction"], {"E", "NE", "N", "NW", "W", "SW", "S", "SE", "stationary"})
        self.assertEqual(len(summary["centroid_path"]), 2)
        self.assertAlmostEqual(summary["confidence"], 0.89, places=2)

    def test_promote_track_summary_row(self) -> None:
        row = {
            "camera_id": "cam-i24v-highway2",
            "video": {"start": 0, "end": 5},
            "track_summaries": {
                "track_id": "suv-1",
                "camera_id": "cam-i24v-highway2",
                "vehicle_class": "suv",
                "color": "black",
                "subtype": "suv",
                "confidence": 0.89,
            },
        }
        promoted = promote_track_summary_row(row)
        self.assertEqual(promoted["track_id"], "suv-1")
        self.assertEqual(promoted["vehicle_class"], "suv")
        self.assertNotIn("track_summaries", promoted)

    def test_project_track_summary_row(self) -> None:
        row = {
            "track_id": "suv-1",
            "camera_id": "cam-i24v-highway2",
            "start_time": "1970-01-01T00:00:00.367033Z",
            "end_time": "1970-01-01T00:00:00.433767Z",
            "vehicle_class": "suv",
            "color": "gray",
            "subtype": "suv",
            "avg_speed": 12.5,
            "entry_direction": "NE",
            "exit_direction": "NE",
            "confidence": 0.89,
            "detections": [{"type": "suv", "bboxes": []}],
            "frame_detections": [{"frame_id": 1, "bbox": [0, 0, 1, 1]}],
            "video": {"start": 0, "end": 5},
        }
        projected = project_track_summary_row(row)
        self.assertEqual(projected["track_id"], "suv-1")
        self.assertEqual(projected["camera_id"], "cam-i24v-highway2")
        self.assertNotIn("detections", projected)
        self.assertNotIn("frame_detections", projected)
        self.assertNotIn("video", projected)


if __name__ == "__main__":
    unittest.main()
