"""Appearance re-identification UDFs for the cross-camera vehicle join.

- :func:`attach_track_embedding` — Map UDF that attaches a per-track appearance
  embedding (from the track's representative crop) under ``embedding``.
- :func:`appearance_match_score` — Join score UDF: cosine similarity of two
  tracks' embeddings, so the cross-camera join is driven by *appearance* rather
  than exact categorical keys. Falls back to the mean-confidence score when an
  embedding is missing, so the join still runs without the re-ID model.
"""

from __future__ import annotations

from typing import Any

from udfs.detection_ops import (
    _crop_from_frame,
    _video_path_from_row,
    crop_from_track_row,
)
from udfs.join_ops import vehicle_match_score
from udfs.vehicle_reid_model import cosine_similarity, embed_crop, embed_crops


def attach_track_embedding(
    row: dict[str, Any],
    *,
    video_field: str = "video",
    output_field: str = "embedding",
) -> dict[str, Any]:
    """Map UDF: attach an appearance embedding for a track row.

    Extracts the track's representative crop (``rep_frame_id`` / ``rep_bbox``)
    and embeds it. On success ``output_field`` holds a list of floats; when the
    crop or model is unavailable it holds ``[]`` so the field is always present
    for the downstream projection and join (which then falls back to confidence).
    """
    result = dict(row)
    crop = crop_from_track_row(row, video_field=video_field)
    embedding = embed_crop(crop) if crop is not None else None
    result[output_field] = embedding if embedding is not None else []
    return result


def attach_track_summary_embeddings(
    row: dict[str, Any],
    *,
    video_field: str = "video",
    summaries_field: str = "track_summaries",
    output_field: str = "embedding",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Embed all track summaries in one camera row before ``Unnest``.

    Representative frames are deduplicated and read through one capture, then
    all resulting crops are embedded in bounded model batches.
    """
    result = dict(row)
    summaries = row.get(summaries_field)
    if not isinstance(summaries, list):
        result[summaries_field] = []
        return result

    copied_summaries = [
        dict(summary) if isinstance(summary, dict) else summary
        for summary in summaries
    ]
    valid_summaries = [
        summary for summary in copied_summaries if isinstance(summary, dict)
    ]
    frame_ids = [
        summary["rep_frame_id"]
        for summary in valid_summaries
        if isinstance(summary.get("rep_frame_id"), int)
    ]

    frames: dict[int, Any] = {}
    video_path = _video_path_from_row(row, video_field=video_field)
    if video_path and frame_ids:
        from mmds.utilities.video import read_frames_at_indices

        frames = read_frames_at_indices(video_path, frame_ids)

    crops: list[Any | None] = []
    for summary in valid_summaries:
        frame_id = summary.get("rep_frame_id")
        bbox = summary.get("rep_bbox")
        frame = frames.get(frame_id) if isinstance(frame_id, int) else None
        crop = (
            _crop_from_frame(frame, bbox)
            if frame is not None and isinstance(bbox, list) and len(bbox) == 4
            else None
        )
        crops.append(crop)

    embeddings = embed_crops(crops, batch_size=batch_size)
    for summary, embedding in zip(valid_summaries, embeddings):
        summary[output_field] = embedding if embedding is not None else []

    result[summaries_field] = copied_summaries
    return result


def appearance_match_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Join score UDF: cosine similarity of two tracks' appearance embeddings.

    Returns the embedding cosine similarity in ``[0, 1]`` when both tracks carry
    an embedding; otherwise falls back to :func:`vehicle_match_score` (mean track
    confidence) so the join degrades gracefully without the re-ID model.
    """
    left_embedding = left.get("embedding")
    right_embedding = right.get("embedding")
    if (
        isinstance(left_embedding, (list, tuple))
        and isinstance(right_embedding, (list, tuple))
        and left_embedding
        and right_embedding
    ):
        return cosine_similarity(left_embedding, right_embedding)
    return vehicle_match_score(left, right)
