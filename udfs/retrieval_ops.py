"""UDF wrappers for the UCA ground-truth caption↔video example."""

from __future__ import annotations

from typing import Any

from mmds.join.text_retrieval import (
    format_gt_caption_clip as format_gt_caption_clip_impl,
    group_gt_clips_by_video as group_gt_clips_by_video_impl,
    tag_gallery_video as tag_gallery_video_impl,
)


def tag_gallery_video(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: add ``source_video_id`` on gallery rows."""
    return tag_gallery_video_impl(row)


def format_gt_caption_clip(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: materialize a ground-truth caption + VideoView clip."""
    return format_gt_caption_clip_impl(row)


def group_gt_clips_by_video(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce UDF: group caption clips into an annotation_excerpt-style video row."""
    return group_gt_clips_by_video_impl(rows)
