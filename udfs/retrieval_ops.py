"""UDF wrappers for the UCA ground-truth caption-to-video example."""

from __future__ import annotations

from typing import Any

from mmds.case_studies import text_retrieval as _impl


def tag_gallery_video(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: add ``source_video_id`` on gallery rows."""
    return _impl.tag_gallery_video(row)


def format_gt_caption_clip(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: materialize a ground-truth caption + VideoView clip."""
    return _impl.format_gt_caption_clip(row)


def group_gt_clips_by_video(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce UDF: group caption clips into an annotation_excerpt-style video row."""
    return _impl.group_gt_clips_by_video(rows)
