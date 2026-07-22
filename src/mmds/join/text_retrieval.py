"""Helpers for UCA ground-truth caption↔video joins."""

from __future__ import annotations

from typing import Any


def tag_gallery_video(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF fields: stable ``source_video_id`` for a gallery row."""
    if not isinstance(row, dict):
        return {}
    video_id = row.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return {}
    return {"source_video_id": video_id}


def _as_nonneg_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _video_source_ref(raw_video: Any) -> dict[str, Any] | None:
    """Copy path/uri/source keys from a gallery Video / VideoView dict."""
    if not isinstance(raw_video, dict):
        return None
    ref: dict[str, Any] = {}
    for key in ("path", "uri", "source"):
        value = raw_video.get(key)
        if isinstance(value, str) and value:
            ref[key] = value
    return ref or None


def format_gt_caption_clip(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten Join ``{left=GT caption, right=gallery}`` into an oracle clip row.

    Video bounds come from the caption's ``start_sec`` / ``end_sec`` (ground
    truth from ``annotation_excerpt.json``).
    """
    if not isinstance(row, dict):
        return {}
    left = row.get("left")
    right = row.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {}

    start = _as_nonneg_float(left.get("start_sec"))
    end = _as_nonneg_float(left.get("end_sec"))
    if start is None or end is None or end < start:
        return {}

    media_ref = _video_source_ref(right.get("video"))
    if media_ref is None:
        return {}

    video_view: dict[str, Any] = {
        "type": "VideoView",
        **media_ref,
        "start": start,
        "end": end,
    }

    video_id = left.get("video_id")
    if not isinstance(video_id, str):
        video_id = right.get("video_id")

    return {
        "caption_id": left.get("caption_id"),
        "video_id": video_id,
        "caption": left.get("caption"),
        "start_sec": start,
        "end_sec": end,
        "timestamps": [start, end],
        "video": video_view,
        "title": right.get("title"),
        "duration_sec": right.get("duration_sec"),
    }


def group_gt_clips_by_video(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce UDF: fold per-caption clips into one annotation_excerpt-style video row.

    Output mirrors ``annotation_excerpt.json`` values::

        {
          "duration": <float>,
          "timestamps": [[start, end], ...],
          "sentences": ["...", ...],
          "title": "...",          # from gallery when present
          "video": {"type": "Video", "path"|"uri"|"source": ...},
        }

    ``video_id`` is added by ``Reduce(..., group_by="video_id")``. Captions are
    sorted by ``(start_sec, end_sec, caption_id)`` for stable ordering.
    """
    if not isinstance(rows, list):
        return {}

    clips: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _as_nonneg_float(row.get("start_sec"))
        end = _as_nonneg_float(row.get("end_sec"))
        caption = row.get("caption")
        if start is None or end is None or end < start or not isinstance(caption, str):
            continue
        clips.append(row)

    if not clips:
        return {}

    clips.sort(
        key=lambda row: (
            float(row["start_sec"]),
            float(row["end_sec"]),
            str(row.get("caption_id") or ""),
        )
    )

    first = clips[0]
    duration = _as_nonneg_float(first.get("duration_sec"))
    media_ref = _video_source_ref(first.get("video"))

    aggregate: dict[str, Any] = {
        "duration": duration if duration is not None else 0.0,
        "timestamps": [
            [float(row["start_sec"]), float(row["end_sec"])] for row in clips
        ],
        "sentences": [row["caption"] for row in clips],
    }
    title = first.get("title")
    if isinstance(title, str) and title:
        aggregate["title"] = title
    if media_ref is not None:
        aggregate["video"] = {"type": "Video", **media_ref}
    return aggregate
