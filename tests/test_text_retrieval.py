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

from udfs.retrieval_ops import (  # noqa: E402
    format_gt_caption_clip,
    group_gt_clips_by_video,
    tag_gallery_video,
)
from udfs.retrieval_ops import (  # noqa: E402
    format_gt_caption_clip as format_gt_udf,
)
from udfs.retrieval_ops import (
    group_gt_clips_by_video as group_udf,
)
from udfs.retrieval_ops import (
    tag_gallery_video as tag_gallery_udf,
)


class TagGalleryTests(unittest.TestCase):
    def test_tag_gallery_video(self) -> None:
        self.assertEqual(
            tag_gallery_video({"video_id": "Abuse001_x264"}),
            {"source_video_id": "Abuse001_x264"},
        )
        self.assertEqual(tag_gallery_video({}), {})


class FormatGtCaptionClipTests(unittest.TestCase):
    def test_format_missing_sides(self) -> None:
        self.assertEqual(format_gt_caption_clip({}), {})
        self.assertEqual(format_gt_caption_clip({"left": {}}), {})

    def test_format_gt_caption_clip(self) -> None:
        hit = format_gt_caption_clip(
            {
                "left": {
                    "caption_id": "Abuse001_x264-004",
                    "video_id": "Abuse001_x264",
                    "caption": "he punched the woman in the head",
                    "start_sec": 8.9,
                    "end_sec": 11.2,
                },
                "right": {
                    "video_id": "Abuse001_x264",
                    "title": "UCA Abuse001",
                    "duration_sec": 91.0,
                    "video": {"type": "Video", "path": "data/uca/videos/Abuse001_x264.mp4"},
                },
            }
        )
        self.assertEqual(hit["caption_id"], "Abuse001_x264-004")
        self.assertEqual(hit["timestamps"], [8.9, 11.2])
        self.assertEqual(hit["video"]["type"], "VideoView")
        self.assertEqual(hit["video"]["path"], "data/uca/videos/Abuse001_x264.mp4")
        self.assertEqual(hit["video"]["start"], 8.9)
        self.assertEqual(hit["video"]["end"], 11.2)

    def test_format_gt_caption_clip_rejects_bad_bounds(self) -> None:
        self.assertEqual(
            format_gt_caption_clip(
                {
                    "left": {"start_sec": 5.0, "end_sec": 1.0, "caption": "x"},
                    "right": {"video": {"path": "a.mp4"}},
                }
            ),
            {},
        )


class GroupByVideoTests(unittest.TestCase):
    def test_group_gt_clips_by_video(self) -> None:
        aggregate = group_gt_clips_by_video(
            [
                {
                    "caption_id": "Abuse001_x264-001",
                    "video_id": "Abuse001_x264",
                    "caption": "second",
                    "start_sec": 7.0,
                    "end_sec": 8.5,
                    "title": "UCA Abuse001",
                    "duration_sec": 91.0,
                    "video": {
                        "type": "VideoView",
                        "path": "data/uca/videos/Abuse001_x264.mp4",
                        "start": 7.0,
                        "end": 8.5,
                    },
                },
                {
                    "caption_id": "Abuse001_x264-000",
                    "video_id": "Abuse001_x264",
                    "caption": "first",
                    "start_sec": 0.0,
                    "end_sec": 5.3,
                    "title": "UCA Abuse001",
                    "duration_sec": 91.0,
                    "video": {
                        "type": "VideoView",
                        "path": "data/uca/videos/Abuse001_x264.mp4",
                        "start": 0.0,
                        "end": 5.3,
                    },
                },
            ]
        )
        self.assertEqual(aggregate["duration"], 91.0)
        self.assertEqual(aggregate["timestamps"], [[0.0, 5.3], [7.0, 8.5]])
        self.assertEqual(aggregate["sentences"], ["first", "second"])
        self.assertEqual(aggregate["title"], "UCA Abuse001")
        self.assertEqual(
            aggregate["video"],
            {"type": "Video", "path": "data/uca/videos/Abuse001_x264.mp4"},
        )

    def test_group_empty(self) -> None:
        self.assertEqual(group_gt_clips_by_video([]), {})
        self.assertEqual(group_gt_clips_by_video("bad"), {})  # type: ignore[arg-type]


class UdfWrapperTests(unittest.TestCase):
    def test_udfs_delegate(self) -> None:
        self.assertEqual(
            tag_gallery_udf({"video_id": "Abuse002_x264"}),
            {"source_video_id": "Abuse002_x264"},
        )
        gt_row = {
            "left": {
                "caption_id": "Abuse001_x264-000",
                "video_id": "Abuse001_x264",
                "caption": "woman reading",
                "start_sec": 0.0,
                "end_sec": 5.3,
            },
            "right": {
                "video_id": "Abuse001_x264",
                "video": {"type": "Video", "path": "data/uca/videos/Abuse001_x264.mp4"},
            },
        }
        self.assertEqual(format_gt_udf(gt_row), format_gt_caption_clip(gt_row))
        clips = [
            {
                "caption_id": "a",
                "caption": "x",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "duration_sec": 10.0,
                "video": {"type": "VideoView", "path": "a.mp4", "start": 0.0, "end": 1.0},
            }
        ]
        self.assertEqual(group_udf(clips), group_gt_clips_by_video(clips))


if __name__ == "__main__":
    unittest.main()
