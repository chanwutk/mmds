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

from udfs.detection_ops import crop_from_track_row
from udfs.join_ops import vehicle_match_score
from udfs.vehicle_reid_model import cosine_similarity, embed_crop


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
