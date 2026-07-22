from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

_VEHICLE_CLASSES = frozenset({"sedan", "suv", "truck"})
_DEFAULT_FPS = 30.0 # Default fps for tracking timestamps

# IoU association threshold for linking a detection to a track.
_TRACK_IOU_THRESHOLD = 0.3
# Close a track when no matching box appears within this many frames. Kept
# generous because the low-confidence detector produces intermittent boxes;
# a short tolerance fragments a vehicle across its own detection dropouts.
_MAX_TRACK_FRAME_GAP = 30
# EMA weight for the smoothed per-frame center velocity (0..1). Higher = more
# responsive to the latest motion; lower = smoother/steadier prediction.
_VELOCITY_SMOOTHING = 0.5
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


def _predicted_bbox(
    last_bbox: list[float],
    center: tuple[float, float],
    velocity: tuple[float, float],
    gap: int,
) -> tuple[list[float], tuple[float, float]]:
    """Constant-velocity forecast of a track's box ``gap`` frames ahead.

    Extrapolates the (smoothed) center velocity and re-centers the last box's
    dimensions on the predicted center. Returns ``(predicted_bbox, predicted_center)``.
    """
    pred_cx = center[0] + velocity[0] * gap
    pred_cy = center[1] + velocity[1] * gap
    width = last_bbox[2] - last_bbox[0]
    height = last_bbox[3] - last_bbox[1]
    predicted = [
        pred_cx - width / 2.0,
        pred_cy - height / 2.0,
        pred_cx + width / 2.0,
        pred_cy + height / 2.0,
    ]
    return predicted, (pred_cx, pred_cy)


def _assign_track_ids(
    detections: list[dict[str, Any]],
    *,
    track_prefix: str,
    max_frame_gap: int = _MAX_TRACK_FRAME_GAP,
    iou_threshold: float = _TRACK_IOU_THRESHOLD,
    velocity_smoothing: float = _VELOCITY_SMOOTHING,
) -> list[dict[str, Any]]:
    """Motion-aware greedy tracker: assign stable ``track_id`` across frames.

    For each detection (processed in frame order), every active track is also
    forecast forward with a **constant-velocity** model using a **smoothed**
    (EMA) center velocity, and association uses the **better of the predicted-box
    and last-box IoU**. The prediction only *adds* reach — it bridges moderate
    motion and short detection gaps that a pure last-box-IoU tracker would drop —
    and can never lose a match the last box would have made, so it never
    increases fragmentation relative to the plain IoU tracker.

    A track is an eligible match when the frame gap is in ``[1, max_frame_gap]``
    and ``max(predicted_iou, last_iou) >= iou_threshold``. The best eligible
    track (highest such IoU) wins; unmatched detections start a new track
    (``veh-1``, ``veh-2``, ...).

    (A center-distance fallback gate and a class-consistency gate were evaluated
    and both *increased* fragmentation on the I24V clips — the distance gate via
    greedy mis-assignment churn, the class gate by splitting a vehicle whenever
    YOLOE's label flickered to ``truck`` — so neither is used. Robustly fixing
    those would need global (Hungarian) assignment + a Kalman filter, i.e. a real
    StrongSORT backend.)

    Args:
        detections: Detection dicts, each with ``frame_id`` and ``bbox``.
        track_prefix: Prefix for new track IDs.
        max_frame_gap: Maximum frame gap to bridge when associating.
        iou_threshold: Minimum ``max(predicted, last-box)`` IoU to associate.
        velocity_smoothing: EMA weight for the smoothed center velocity.

    Returns:
        The detections tagged with an assigned ``track_id``.
    """
    ordered = sorted(
        detections,
        key=lambda item: (item.get("frame_id", 0), item.get("bbox", [0])[0]),
    )
    # Track state: last_frame/bbox/center, smoothed velocity, and how many
    # detections it has (to seed velocity on the first observed motion).
    active_tracks: dict[str, dict[str, Any]] = {}
    next_track = 1
    tracked: list[dict[str, Any]] = []

    for detection in ordered:
        frame_id = detection.get("frame_id")
        bbox = detection.get("bbox")
        if not isinstance(frame_id, int) or not isinstance(bbox, list) or len(bbox) != 4:
            continue

        det_bbox = [float(v) for v in bbox]
        det_center = _bbox_centroid(det_bbox)

        best_track_id: str | None = None
        best_iou = iou_threshold
        for track_id, state in active_tracks.items():
            gap = frame_id - state["last_frame"]
            if gap < 1 or gap > max_frame_gap:
                continue
            pred_bbox, _ = _predicted_bbox(
                state["last_bbox"], state["last_center"], state["velocity"], gap
            )
            # Better of the predicted box and the last box: prediction adds reach
            # for moving objects without ever losing a last-box overlap match.
            assoc_iou = max(
                _bbox_iou(det_bbox, pred_bbox), _bbox_iou(det_bbox, state["last_bbox"])
            )
            if assoc_iou > best_iou:
                best_iou = assoc_iou
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = f"{track_prefix}-{next_track}"
            next_track += 1
            active_tracks[best_track_id] = {
                "last_frame": frame_id,
                "last_bbox": det_bbox,
                "last_center": det_center,
                "velocity": (0.0, 0.0),
                "n": 0,
            }

        state = active_tracks[best_track_id]
        gap = max(1, frame_id - state["last_frame"])
        instant_v = (
            (det_center[0] - state["last_center"][0]) / gap,
            (det_center[1] - state["last_center"][1]) / gap,
        )
        if state["n"] == 0:
            velocity = (0.0, 0.0)  # first detection: no motion sample yet
        elif state["n"] == 1:
            velocity = instant_v  # first motion sample seeds the velocity
        else:
            old_vx, old_vy = state["velocity"]
            velocity = (
                velocity_smoothing * instant_v[0] + (1.0 - velocity_smoothing) * old_vx,
                velocity_smoothing * instant_v[1] + (1.0 - velocity_smoothing) * old_vy,
            )
        state["velocity"] = velocity
        state["last_frame"] = frame_id
        state["last_bbox"] = det_bbox
        state["last_center"] = det_center
        state["n"] += 1

        tagged = dict(detection)
        tagged["track_id"] = best_track_id
        tracked.append(tagged)

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

    # Representative frame/box: the highest-confidence detection in the track.
    # Used downstream to extract a single crop for appearance labeling.
    rep_detection = max(
        ordered,
        key=lambda detection: float(detection["confidence"])
        if isinstance(detection.get("confidence"), (int, float))
        else -1.0,
    )
    rep_frame_id = rep_detection.get("frame_id")
    rep_bbox = rep_detection.get("bbox")
    if not isinstance(rep_frame_id, int):
        rep_frame_id = first_frame
    if isinstance(rep_bbox, list) and len(rep_bbox) == 4:
        rep_bbox = [float(value) for value in rep_bbox]
    else:
        rep_bbox = None

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
        "rep_frame_id": rep_frame_id,
        "rep_bbox": rep_bbox,
    }


def strongsort_track_frame_detections(
    row: dict[str, Any],
    *,
    input_field: str = "frame_detections",
    summaries_field: str = "track_summaries",
    min_track_frames: int | None = None,
    min_track_confidence: float | None = None,
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

    # Track class-agnostically: associate boxes across frames by IoU regardless of
    # the per-frame YOLOE class. YOLOE's class label flickers (sedan<->suv) for the
    # same physical vehicle; tracking per class would split one vehicle into several
    # tracks. Each track's class/color/subtype is resolved afterwards by majority
    # vote in :func:`_summarize_track`.
    vehicle_detections = [
        detection
        for detection in frame_detections
        if isinstance(detection, dict) and detection.get("vehicle_class") in _VEHICLE_CLASSES
    ]
    if not vehicle_detections:
        return {summaries_field: []}

    tracker = StrongSortTracker(track_prefix="veh")
    tracked = tracker.update(vehicle_detections)

    by_track: dict[str, list[dict[str, Any]]] = {}
    for detection in tracked:
        track_id = detection.get("track_id")
        if not isinstance(track_id, str):
            continue
        by_track.setdefault(track_id, []).append(detection)

    frames_threshold = (
        _MIN_TRACK_FRAMES if min_track_frames is None else min_track_frames
    )
    confidence_threshold = (
        _MIN_TRACK_CONFIDENCE
        if min_track_confidence is None
        else min_track_confidence
    )

    summaries: list[dict[str, Any]] = []
    for track_id, track_detections in sorted(by_track.items()):
        summary = _summarize_track(track_id, track_detections, row)
        # Drop short / low-confidence fragments so the greedy IoU tracker's
        # split tracks don't inflate the distinct-vehicle count.
        if summary and _track_is_substantial(
            summary,
            min_frames=frames_threshold,
            min_confidence=confidence_threshold,
        ):
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
    "embedding",
)


def project_track_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: keep only fields needed for cross-camera join and trajectory export."""
    return {key: row[key] for key in _TRACK_JOIN_FIELDS if key in row}


# Minimum observed detections and mean confidence for a track to be treated as
# a real vehicle rather than a short tracker fragment. The greedy IoU tracker
# fragments fast-moving highway vehicles into many one-/two-frame stubs, which
# otherwise inflate the vehicle count and flood the cross-camera join with
# spurious pairs. Confidence is kept low because the detector runs at a low
# ``conf`` floor, so a real track's mean confidence is legitimately modest;
# track *length* is the more reliable fragment signal.
_MIN_TRACK_FRAMES = 5
_MIN_TRACK_CONFIDENCE = 0.15


def _track_is_substantial(
    summary: dict[str, Any],
    *,
    min_frames: int = _MIN_TRACK_FRAMES,
    min_confidence: float = _MIN_TRACK_CONFIDENCE,
) -> bool:
    """Return whether a track summary is a real vehicle vs. a short fragment.

    Shared by :func:`is_substantial_track` (promoted-row Filter predicate) and
    :func:`strongsort_track_frame_detections` (nested-summary filtering) so both
    apply identical thresholds. Uses ``centroid_path`` length as the frame count
    and mean ``confidence``.
    """
    path = summary.get("centroid_path")
    frame_count = len(path) if isinstance(path, list) else 0
    confidence = summary.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    return frame_count >= min_frames and confidence >= min_confidence


def is_substantial_track(row: dict[str, Any]) -> bool:
    """Filter predicate: drop short / low-confidence tracker fragments.

    Expects a promoted track-summary row (``centroid_path`` and ``confidence``
    at the top level). Keeps tracks with at least ``_MIN_TRACK_FRAMES`` observed
    detections and mean ``confidence`` >= ``_MIN_TRACK_CONFIDENCE``.
    """
    return _track_is_substantial(row)
