from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

_VEHICLE_CLASSES = frozenset({"sedan", "suv", "truck"})
_DEFAULT_FPS = 30.0 # Default fps for tracking timestamps

# IoU association threshold for StrongSORT-shaped linking across frames.
_TRACK_IOU_THRESHOLD = 0.3
# Close a track when no matching box appears within this many frames.
_MAX_TRACK_FRAME_GAP = 30
# Margin (fraction of frame width/height) for edge exit detection.
_EDGE_MARGIN = 0.05


def _bbox_iou(left: list[float], right: list[float]) -> float:
    """Intersection Over Union (IoU) between two bounding boxes. Returns a value between 0 and 1."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _bbox_centroid(bbox: list[float]) -> tuple[float, float]:
    """Calculate the center point of a bounding box. Used for path, speed, and direction."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _row_fps(row: dict[str, Any]) -> float:
    """Get the fps from the row. Used for tracking timestamps."""
    fps = row.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)
    return _DEFAULT_FPS


def _clip_start_seconds(row: dict[str, Any]) -> float:
    """Get the start time from the video. Used for tracking timestamps."""
    video = row.get("video")
    if isinstance(video, dict):
        start = video.get("start")
        if isinstance(start, (int, float)):
            return float(start)
    return 0.0


def _frame_to_timestamp(row: dict[str, Any], frame_id: int) -> str:
    """
    Map an absolute ``frame_id`` to an ISO-8601 UTC timestamp string.
    
    If record_at is provided, timestamp is the recorded_at + frame_id / fps.
    Else, timestamp is the epoch + frame_id / fps.
    """
    fps = _row_fps(row)
    offset_sec = _clip_start_seconds(row) + (frame_id / fps)
    recorded_at = row.get("recorded_at")
    if isinstance(recorded_at, str) and recorded_at:
        timestamp = recorded_at
        if timestamp.endswith("Z"):
            timestamp = timestamp[:-1] + "+00:00"
        try:
            base = datetime.fromisoformat(timestamp)
        except ValueError:
            base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        moment = base + timedelta(seconds=offset_sec - _clip_start_seconds(row))
        return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_sec)
    return epoch.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _compass_direction(dx: float, dy: float) -> str:
    """Map an image-plane delta to a compass label (E, NE, N, ...)."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "stationary"
    angle = math.degrees(math.atan2(-dy, dx)) % 360.0
    labels = [
        "E",
        "NE",
        "N",
        "NW",
        "W",
        "SW",
        "S",
        "SE",
    ]
    index = int((angle + 22.5) // 45) % 8
    return labels[index]


def _direction_from_points(
    start: tuple[float, float],
    end: tuple[float, float],
) -> str:
    """Map a start and end point to a compass label."""
    return _compass_direction(end[0] - start[0], end[1] - start[1])


def _frame_dimensions(row: dict[str, Any]) -> tuple[float, float]:
    """Get the width and height from the row. Used for tracking."""
    width = row.get("width")
    height = row.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return float(width), float(height)
    return 1920.0, 1080.0


def _is_near_frame_edge(
    centroid: tuple[float, float],
    *,
    width: float,
    height: float,
) -> bool:
    """Check if the centroid is near 5% of frame edge. Use to infer entry/exit direction."""
    x, y = centroid
    margin_x = width * _EDGE_MARGIN
    margin_y = height * _EDGE_MARGIN
    return (
        x <= margin_x
        or y <= margin_y
        or x >= width - margin_x
        or y >= height - margin_y
    )


def _assign_track_ids(
    detections: list[dict[str, Any]],
    *,
    track_prefix: str,
    max_frame_gap: int = _MAX_TRACK_FRAME_GAP,
) -> list[dict[str, Any]]:
    """
    Greedy IoU tracker: assign stable ``track_id`` values across frames.
    
    Args:
        detections: List of detection dicts, each with a 'frame_id', 'bbox', and 'track_id'.
        track_prefix: Prefix for new track IDs.
        max_frame_gap: Maximum frame gap to consider for track association.

    Returns:
        List of detection dicts with assigned 'track_id'.
    
    For each detection, find the best matching track (highest IoU) across all active tracks.
    If no match is found, create a new track (suv-1, sedan-2, ...).
    Tag each detection with the best track ID.
    """
    ordered = sorted(
        detections,
        key=lambda item: (item.get("frame_id", 0), item.get("bbox", [0])[0]),
    )
    active_tracks: dict[str, dict[str, Any]] = {}
    next_track = 1
    tracked: list[dict[str, Any]] = []

    for detection in ordered:
        frame_id = detection.get("frame_id")
        bbox = detection.get("bbox")
        if not isinstance(frame_id, int) or not isinstance(bbox, list) or len(bbox) != 4:
            continue

        best_track_id: str | None = None
        best_iou = _TRACK_IOU_THRESHOLD
        for track_id, last_detection in active_tracks.items():
            last_frame = last_detection.get("frame_id")
            last_bbox = last_detection.get("bbox")
            if not isinstance(last_frame, int) or not isinstance(last_bbox, list):
                continue
            gap = frame_id - last_frame
            if gap <= 0 or gap > max_frame_gap:
                continue
            iou = _bbox_iou([float(v) for v in bbox], [float(v) for v in last_bbox])
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = f"{track_prefix}-{next_track}"
            next_track += 1

        tagged = dict(detection)
        tagged["track_id"] = best_track_id
        tracked.append(tagged)
        active_tracks[best_track_id] = tagged

    return tracked


class StrongSortTracker:
    """StrongSORT-shaped multi-object tracker (v1 stub).

    v1 uses greedy IoU association across frames. The class is an explicit
    stub for a real StrongSORT backend (e.g. ``boxmot``): replace
    :meth:`update` without changing query UDFs.
    """

    def __init__(
        self,
        *,
        track_prefix: str = "track",
        max_frame_gap: int = _MAX_TRACK_FRAME_GAP,
    ) -> None:
        self._track_prefix = track_prefix
        self._max_frame_gap = max_frame_gap

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign track IDs to detections. Helper function for strongsort_track_frame_detections."""
        return _assign_track_ids(
            detections,
            track_prefix=self._track_prefix,
            max_frame_gap=self._max_frame_gap,
        )


def _centroid_path(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate and list the {frame_id, x, y} centroids for each detection in temporal order."""
    path: list[dict[str, Any]] = []
    for detection in detections:
        bbox = detection.get("bbox")
        frame_id = detection.get("frame_id")
        if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(frame_id, int):
            continue
        cx, cy = _bbox_centroid(bbox)
        path.append({"frame_id": frame_id, "x": cx, "y": cy})
    return path


def _average_speed_px_per_sec(
    detections: list[dict[str, Any]],
    *,
    fps: float,
) -> float:
    """Calculate the average speed in pixels per second (centroid distance / (current frame - previous frame) * fps) for a list of detections."""
    if len(detections) < 2:
        return 0.0
    ordered = sorted(detections, key=lambda item: item.get("frame_id", 0))
    speeds: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        prev_frame = previous.get("frame_id")
        curr_frame = current.get("frame_id")
        prev_bbox = previous.get("bbox")
        curr_bbox = current.get("bbox")
        if (
            not isinstance(prev_frame, int)
            or not isinstance(curr_frame, int)
            or not isinstance(prev_bbox, list)
            or not isinstance(curr_bbox, list)
        ):
            continue
        frame_gap = curr_frame - prev_frame
        if frame_gap <= 0:
            continue
        p_centroid = _bbox_centroid(prev_bbox)
        c_centroid = _bbox_centroid(curr_bbox)
        distance = math.hypot(c_centroid[0] - p_centroid[0], c_centroid[1] - p_centroid[1])
        elapsed = frame_gap / fps
        speeds.append(distance / elapsed)
    if not speeds:
        return 0.0
    return sum(speeds) / len(speeds)


def _majority_label(detections: list[dict[str, Any]], field: str, default: str) -> str:
    counts: dict[str, int] = {}
    """
    Count the occurrences of each value in the vehicle_class, color, or subtype column.
    Return the most frequent value, or the default if no values are present.
    """
    for detection in detections:
        value = detection.get(field)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(counts, key=counts.get)


def _summarize_track(
    track_id: str,
    detections: list[dict[str, Any]],
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Summarize each track by:
    - Extracting camera_id, fps, width, height.
    - Calculating centroid path, entry/exit points, and directions.
    - Closing tracks that exit through the frame edge.
    - Calculating confidence as the mean of detection confidences.
    """
    ordered = sorted(detections, key=lambda item: item.get("frame_id", 0))
    if not ordered:
        return {}

    camera_id = row.get("camera_id")
    if not isinstance(camera_id, str):
        camera_id = ordered[0].get("camera_id", "")

    fps = _row_fps(row)
    width, height = _frame_dimensions(row)
    path = _centroid_path(ordered)
    first_centroid = (path[0]["x"], path[0]["y"])
    last_centroid = (path[-1]["x"], path[-1]["y"])
    entry_point = (path[0]["x"], path[0]["y"])
    exit_point = (path[-1]["x"], path[-1]["y"])
    if len(path) >= 2:
        entry_point = (path[0]["x"], path[0]["y"])
        entry_next = (path[1]["x"], path[1]["y"])
        exit_prev = (path[-2]["x"], path[-2]["y"])
        exit_point = (path[-1]["x"], path[-1]["y"])
        entry_direction = _direction_from_points(entry_point, entry_next)
        exit_direction = _direction_from_points(exit_prev, exit_point)
    else:
        entry_direction = "stationary"
        exit_direction = "stationary"

    # Close tracks that exit through the frame edge.
    if _is_near_frame_edge((last_centroid[0], last_centroid[1]), width=width, height=height):
        exit_direction = _direction_from_points(
            (path[-2]["x"], path[-2]["y"]) if len(path) >= 2 else first_centroid,
            last_centroid,
        )

    confidences = [
        float(detection["confidence"])
        for detection in ordered
        if isinstance(detection.get("confidence"), (int, float))
    ]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0

    first_frame = ordered[0].get("frame_id")
    last_frame = ordered[-1].get("frame_id")
    if not isinstance(first_frame, int) or not isinstance(last_frame, int):
        return {}

    return {
        "track_id": track_id,
        "camera_id": camera_id,
        "start_time": _frame_to_timestamp(row, first_frame),
        "end_time": _frame_to_timestamp(row, last_frame),
        "vehicle_class": _majority_label(ordered, "vehicle_class", "sedan"),
        "color": _majority_label(ordered, "color", "gray"),
        "subtype": _majority_label(ordered, "subtype", "sedan"),
        "avg_speed": _average_speed_px_per_sec(ordered, fps=fps),
        "entry_direction": entry_direction,
        "exit_direction": exit_direction,
        "centroid_path": path,
        "confidence": confidence,
    }


def strongsort_track_frame_detections(
    row: dict[str, Any],
    *,
    input_field: str = "frame_detections",
    summaries_field: str = "track_summaries",
) -> dict[str, Any]:
    """Map UDF: StrongSORT-shaped tracking over ``frame_detections``.

    Feeds per-frame detections into :class:`StrongSortTracker`, closes tracks
    when association breaks (frame gap) or the feed ends, and emits completed
    track summaries as JSON-serializable dicts plus a JSON string field.

    Example output:
    {
    "camera_id": "cam-i24v-highway2",
    "video": {...},
    "frame_detections": [...],
    "track_summaries": [
        {
        "track_id": "suv-1",
        "camera_id": "cam-i24v-highway2",
        "start_time": "2023-08-10T12:00:00.333333Z",
        "end_time": "2023-08-10T12:00:01.100000Z",
        "vehicle_class": "suv",
        "color": "black",
        "subtype": "suv",
        "avg_speed": 42.5,
        "entry_direction": "E",
        "exit_direction": "NE",
        "centroid_path": [{"frame_id": 10, "x": 150.0, "y": 250.0}, ...],
        "confidence": 0.89
        }
    ]
    }
    """
    frame_detections = row.get(input_field)
    if not isinstance(frame_detections, list) or not frame_detections:
        return {summaries_field: []}

    by_class: dict[str, list[dict[str, Any]]] = {}
    for detection in frame_detections:
        if not isinstance(detection, dict):
            continue
        vehicle_class = detection.get("vehicle_class")
        if vehicle_class not in _VEHICLE_CLASSES:
            continue
        by_class.setdefault(str(vehicle_class), []).append(detection)

    summaries: list[dict[str, Any]] = []
    for vehicle_class, class_detections in sorted(by_class.items()):
        tracker = StrongSortTracker(track_prefix=vehicle_class)
        tracked = tracker.update(class_detections)

        by_track: dict[str, list[dict[str, Any]]] = {}
        for detection in tracked:
            track_id = detection.get("track_id")
            if not isinstance(track_id, str):
                continue
            by_track.setdefault(track_id, []).append(detection)

        for track_id, track_detections in sorted(by_track.items()):
            summary = _summarize_track(track_id, track_detections, row)
            if summary:
                summaries.append(summary)

    return {summaries_field: summaries}


def promote_track_summary_row(
    row: dict[str, Any],
    *,
    summary_field: str = "track_summaries",
) -> dict[str, Any]:
    """Map UDF: lift one unnested track summary to top-level join fields."""
    summary = row.get(summary_field)
    if not isinstance(summary, dict):
        return {}
    promoted = {key: value for key, value in row.items() if key != summary_field}
    promoted.update(summary)
    return promoted


_TRACK_JOIN_FIELDS: tuple[str, ...] = (
    "track_id",
    "camera_id",
    "start_time",
    "end_time",
    "vehicle_class",
    "color",
    "subtype",
    "avg_speed",
    "entry_direction",
    "exit_direction",
    "confidence",
)


def project_track_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: keep only fields needed for cross-camera join and trajectory export."""
    return {key: row[key] for key in _TRACK_JOIN_FIELDS if key in row}
