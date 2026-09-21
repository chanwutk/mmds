"""Build a cross-camera vehicle join over an I24V feed manifest."""

from __future__ import annotations

from mmds import Detect, Input, Join, Map, Unnest
from mmds.model import DatasetExpr
from udfs.detection_ops import build_vehicle_frame_detections, nms_vehicle_detections
from udfs.join_ops import same_vehicle
from udfs.reid_ops import appearance_match_score, attach_track_summary_embeddings
from udfs.tracking_ops import (
    project_track_summary_row,
    promote_track_summary_row,
    strongsort_track_frame_detections,
)
from udfs.trajectory_ops import join_match_to_trajectory

DEFAULT_MANIFEST = "data/i24v_traffic_highway2_highway3_5s.jsonl"

"""Run: ./run examples/join_cross_camera_vehicle.py after importing MP4s"""

def build_query(
    feeds_jsonl: str = DEFAULT_MANIFEST,
    *,
    min_score: float = 0.4,
) -> DatasetExpr:
    """Build the UDF-backed vehicle trajectory pipeline."""
    feeds = Input(feeds_jsonl)
    detected = Detect(
        feeds,
        "video",
        ["sedan", "suv", "truck"],
        output_field="detections",
        frame_stride=1,
        conf=0.1,
        imgsz=1280,
        name="detect_vehicles",
    )
    nmsed = Map(
        detected,
        nms_vehicle_detections,
        name="suppress_duplicate_detections",
    )
    framed = Map(
        nmsed,
        build_vehicle_frame_detections,
        name="build_frame_detections",
    )
    tracked = Map(
        framed,
        strongsort_track_frame_detections,
        name="track_vehicles",
    )
    embedded = Map(
        tracked,
        attach_track_summary_embeddings,
        replace=True,
        name="embed_track_summaries",
    )
    unnested = Unnest(
        embedded,
        "track_summaries",
        name="one_track_per_row",
    )
    promoted = Map(
        unnested,
        promote_track_summary_row,
        replace=True,
        name="promote_track_summary",
    )
    track_rows = Map(
        promoted,
        project_track_summary_row,
        replace=True,
        name="project_track_fields",
    )
    matches = Join(
        track_rows,
        track_rows,
        same_vehicle,
        one_to_one=True,
        score=appearance_match_score,
        min_score=min_score,
        left_key=("camera_id", "track_id"),
        right_key=("camera_id", "track_id"),
        name="match_cross_camera_tracks",
    )
    return Map(
        matches,
        join_match_to_trajectory,
        replace=True,
        name="build_vehicle_trajectories",
    )


output = build_query()
