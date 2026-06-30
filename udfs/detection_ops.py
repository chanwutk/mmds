from __future__ import annotations

from typing import Any


# YOLOE conf scores are model-internal (0–1); Ultralytics defaults drop boxes below ~0.25.
HIGH_CONFIDENCE_THRESHOLD = 0.5


def prune_detections(
    row: dict[str, Any],
    *,
    min_confidence: float,
    only_classes: set[str] | None = None,
    keep_other_classes: bool = True,
    detection_field: str = "detections"
) -> dict[str, Any]:
    """Return mapping updates that prune a Detect-style 'detections' field.

    Expected input shape (as produced by MMDS 'Detect'):

        [{"type": <class_name>, "bboxes": [{"confidence": float, ...}, ...]}, ...]

    Rules:
    - drops non-dict items and non-list bbox containers
    - for classes in 'only_classes' (or all classes if None), keeps only bboxes with
      numeric 'confidence' >= 'min_confidence'
    - drops a class entry if it has no remaining bboxes
    - if 'keep_other_classes' is True, classes not in 'only_classes' are preserved
      unchanged (including their low-confidence bboxes)

    Returns '{detection_field: <pruned_list>}' so it can be used as a 'Map' UDF.
    """
    if not isinstance(min_confidence, (int, float)):
        raise TypeError("min_confidence must be numeric.")

    detections = row.get(detection_field)
    if not isinstance(detections, list):
        return {detection_field: []}

    pruned: list[dict[str, Any]] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        cls = item.get("type")
        bboxes = item.get("bboxes")
        if not isinstance(cls, str) or not isinstance(bboxes, list):
            continue

        should_prune = only_classes is None or cls in only_classes
        if not should_prune:
            if keep_other_classes:
                pruned.append({"type": cls, "bboxes": bboxes})
            continue

        kept_bboxes: list[dict[str, Any]] = []
        for bbox in bboxes:
            if not isinstance(bbox, dict):
                continue
            conf = bbox.get("confidence")
            if isinstance(conf, (int, float)) and conf >= min_confidence:
                kept_bboxes.append(bbox)
        if kept_bboxes:
            pruned.append({"type": cls, "bboxes": kept_bboxes})

    return {detection_field: pruned}


def _has_class_with_boxes(
    row: dict[str, Any],
    class_name: str,
    *,
    min_confidence: float | None = None,
) -> bool:
    detections = row.get("detections")
    if not isinstance(detections, list):
        return False
    for item in detections:
        if not isinstance(item, dict):
            continue
        if item.get("type") != class_name:
            continue
        bboxes = item.get("bboxes")
        if not isinstance(bboxes, list):
            continue
        for bbox in bboxes:
            if not isinstance(bbox, dict):
                continue
            if min_confidence is None:
                return True
            confidence = bbox.get("confidence")
            if isinstance(confidence, (int, float)) and confidence >= min_confidence:
                return True
    return False


def keep_rows_with_class(
    row: dict[str, Any],
    class_name: str = "bear",
    *,
    min_confidence: float | None = None,
) -> bool:
    """Keep rows where ``Detect`` found at least one box for ``class_name``."""
    return _has_class_with_boxes(row, class_name, min_confidence=min_confidence)


def keep_rows_with_high_confidence_class(
    row: dict[str, Any],
    class_name: str = "bear",
    *,
    min_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
) -> bool:
    """Keep rows with at least one ``class_name`` box at or above ``min_confidence``."""
    return keep_rows_with_class(
        row, class_name, min_confidence=min_confidence
    )


def prune_class_detections_to_high_confidence(
    row: dict[str, Any],
    class_name: str = "bear",
    *,
    min_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Prune ``class_name`` boxes below ``min_confidence``; return Map-style updates."""
    return prune_detections(
        row,
        min_confidence=min_confidence,
        only_classes={class_name},
        keep_other_classes=True,
        detection_field="detections"
    )


def keep_rows_with_detections(row: dict[str, Any]) -> bool:
    """Keep rows where ``Detect`` produced any non-empty detection list."""
    detections = row.get("detections")
    if not isinstance(detections, list):
        return False
    for item in detections:
        if isinstance(item, dict) and item.get("bboxes"):
            return True
    return False


# ---------------------------------------------------------------------------
# Vehicle detection (I24V / cross-camera join)
# ---------------------------------------------------------------------------

_VEHICLE_CLASSES = frozenset({"sedan", "suv", "truck"})
_VEHICLE_NMS_IOU = 0.65 # Two boxes are considered the same if their Intersection Over Union (IoU) ≥ threshold

# Color attributes for vehicle color prediction. Currently, we have 13 colors.
# May need to change color references to improve color prediction accuracy.
_COLOR_REFERENCES: dict[str, tuple[float, float, float]] = {
    "white": (235.0, 235.0, 235.0),
    "silver": (192.0, 192.0, 192.0),
    "light_gray": (175.0, 175.0, 175.0),
    "gray": (128.0, 128.0, 128.0),
    "dark_gray": (85.0, 85.0, 85.0),
    "black": (30.0, 30.0, 30.0),
    "beige": (210.0, 195.0, 160.0),
    "yellow": (220.0, 200.0, 60.0),
    "red": (180.0, 35.0, 35.0),
    "blue": (35.0, 60.0, 170.0),
    "green": (40.0, 110.0, 55.0),
    "brown": (120.0, 80.0, 50.0),
}


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


def _nms_boxes(
    boxes: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Greedy per-frame NMS (Non-Maximum Suppression) over Detect-style bbox dicts.

    Args:
        boxes: List of detection dicts, each with a 'bbox' (list of 4 numbers)
            and 'confidence' (score).
        iou_threshold: IoU threshold for suppression; [0, 1].

    Returns:
        List of dicts for selected (kept) boxes, each in the original dict format.

    Sorts all detected bounding boxes by confidence score in descending order.
    It then iteratively selects the highest-confidence box and removes all remaining boxes
    that overlap with it above a chosen IoU threshold. This process repeats until no boxes
    remain, leaving only the most confident, non-overlapping detections for each object.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")

    candidates: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        bbox = box.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not all(isinstance(value, (int, float)) for value in bbox):
            continue
        candidates.append(box)

    candidates.sort(
        key=lambda item: float(item.get("confidence"))
        if isinstance(item.get("confidence"), (int, float))
        else -1.0,
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        cand_bbox = [float(value) for value in candidate["bbox"]]
        suppress = False
        for kept_box in kept:
            kept_bbox = kept_box.get("bbox")
            if not isinstance(kept_bbox, list) or len(kept_bbox) != 4:
                continue
            if _bbox_iou(cand_bbox, [float(value) for value in kept_bbox]) >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    return kept


def nms_vehicle_detections(
    row: dict[str, Any],
    *,
    iou_threshold: float = _VEHICLE_NMS_IOU,
    detection_field: str = "detections",
) -> dict[str, Any]:
    """
    Map UDF: class-wise per-frame NMS for sedan/suv/truck after ``Detect``.
    
    Args:
        row: The input row containing the detections.
        iou_threshold: The IoU threshold for suppression.
        detection_field: The field containing the detections.

    Returns:
        List of dicts for selected (kept) boxes, each in the original dict format.
    
    Performs Non-Maximum Suppression (NMS) on vehicle detections to remove duplicate or overlapping boxes.
    It groups detections by frame index and then applies the NMS algorithm to each group independently.
    The resulting list of boxes is sorted by frame index and then by confidence score.
    """
    detections = row.get(detection_field)
    if not isinstance(detections, list):
        return {detection_field: []}

    nmsed: list[dict[str, Any]] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        vehicle_class = item.get("type")
        bboxes = item.get("bboxes")
        if vehicle_class not in _VEHICLE_CLASSES or not isinstance(bboxes, list):
            continue

        by_frame: dict[int, list[dict[str, Any]]] = {}
        for bbox_entry in bboxes:
            if not isinstance(bbox_entry, dict):
                continue
            frame_id = bbox_entry.get("frame_idx")
            if not isinstance(frame_id, int):
                continue
            by_frame.setdefault(frame_id, []).append(bbox_entry)

        kept_bboxes: list[dict[str, Any]] = []
        for frame_id in sorted(by_frame):
            kept_bboxes.extend(
                _nms_boxes(by_frame[frame_id], iou_threshold=iou_threshold)
            )
        kept_bboxes.sort(
            key=lambda entry: (entry.get("frame_idx", 0), entry.get("bbox", [0])[0])
        )
        if kept_bboxes:
            nmsed.append({"type": vehicle_class, "bboxes": kept_bboxes})

    return {detection_field: nmsed}


def nearest_named_color(rgb: tuple[float, float, float]) -> str:
    """Map an ``(r, g, b)`` triple to the closest named color in the vocab."""
    best_name = "gray" # Default color if no match is found
    best_distance = float("inf")
    for name, reference in _COLOR_REFERENCES.items():
        distance = sum((value - ref) ** 2 for value, ref in zip(rgb, reference))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def vehicle_sub_type_from_geometry(vehicle_class: str, bbox: list[float]) -> str:
    """Coarse subtype from class and box aspect ratio. Helper function for predict_vehicle_attributes."""
    if vehicle_class == "truck":
        return "pickup"
    if vehicle_class == "suv":
        return "suv"
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "sedan"
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    ratio = width / height
    if ratio >= 2.2:
        return "coupe"
    if ratio <= 1.5:
        return "hatchback"
    return "sedan" # Default subtype if no match is found


def _mean_rgb_from_crop(crop: Any) -> tuple[float, float, float] | None:
    """Calculate the mean RGB values from a cropped vehicle image. Helper function for predict_vehicle_attributes."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    mean = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
    if len(mean) < 3:
        return None
    blue, green, red = (float(mean[0]), float(mean[1]), float(mean[2]))
    return (red, green, blue)


def predict_vehicle_attributes(
    vehicle_class: str,
    bbox: list[float],
    crop: Any | None,
) -> dict[str, str]:
    """Return ``vehicle_color`` and ``vehicle_sub_type`` for a vehicle box."""
    mean_rgb = _mean_rgb_from_crop(crop)
    color = nearest_named_color(mean_rgb) if mean_rgb is not None else "gray"
    sub_type = vehicle_sub_type_from_geometry(vehicle_class, bbox)
    return {
        "vehicle_color": color,
        "vehicle_sub_type": sub_type,
    }


def _crop_from_frame(frame: Any, bbox: list[float]) -> Any | None:
    """Crop the bounding box to the rectangular frame dimensions."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _read_frame_at_index(video_path: str, frame_id: int) -> Any | None:
    """Read the frame at the given index from the video file. Used to load only frames with detections."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _video_path_from_row(row: dict[str, Any], *, video_field: str = "video") -> str | None:
    """Resolve the absolute path to the source video file referenced by this row."""
    from mmds.execution.ops.detect import _resolve_video_source
    from mmds.utilities.video import open_video

    raw = row.get(video_field)
    if raw is None:
        return None
    try:
        source = _resolve_video_source(raw)
        video = open_video(source)
    except Exception:
        return None
    if isinstance(video, list):
        return None
    return str(video.path)


def build_vehicle_frame_detections(
    row: dict[str, Any],
    *,
    detection_field: str = "detections",
    video_field: str = "video",
    output_field: str = "frame_detections",
) -> dict[str, Any]:
    """Map UDF: flatten NMS'd vehicle ``Detect`` output into frame detections.

    For each frame with a vehicle detection, this function:
    - Crops the bounding box to the rectangular frame dimensions.
    - Predicts the vehicle color and subtype.
    - Returns a dictionary record.
    
    Each output record has::

        {
            "frame_id": int,
            "camera_id": str,
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "vehicle_class": str,
            "color": str,
            "subtype": str,
        }
    """
    camera_id = row.get("camera_id")
    if not isinstance(camera_id, str):
        camera_id = ""

    detections = row.get(detection_field)
    if not isinstance(detections, list):
        return {output_field: []}

    boxes_by_frame: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for item in detections:
        if not isinstance(item, dict):
            continue
        vehicle_class = item.get("type")
        if vehicle_class not in _VEHICLE_CLASSES:
            continue
        bboxes = item.get("bboxes")
        if not isinstance(bboxes, list):
            continue
        for box in bboxes:
            if not isinstance(box, dict):
                continue
            frame_id = box.get("frame_idx")
            if not isinstance(frame_id, int):
                continue
            boxes_by_frame.setdefault(frame_id, []).append((vehicle_class, box))

    if not boxes_by_frame:
        return {output_field: []}

    video_path = _video_path_from_row(row, video_field=video_field)
    frame_cache: dict[int, Any] = {}
    if video_path:
        for frame_id in boxes_by_frame:
            frame = _read_frame_at_index(video_path, frame_id)
            if frame is not None:
                frame_cache[frame_id] = frame

    frame_detections: list[dict[str, Any]] = []
    for frame_id in sorted(boxes_by_frame):
        frame = frame_cache.get(frame_id)
        for vehicle_class, box in boxes_by_frame[frame_id]:
            bbox = box.get("bbox")
            confidence = box.get("confidence")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if not isinstance(confidence, (int, float)):
                continue
            crop = _crop_from_frame(frame, bbox) if frame is not None else None
            attrs = predict_vehicle_attributes(vehicle_class, bbox, crop)
            frame_detections.append(
                {
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "bbox": [float(value) for value in bbox],
                    "confidence": float(confidence),
                    "vehicle_class": vehicle_class,
                    "color": attrs["vehicle_color"],
                    "subtype": attrs["vehicle_sub_type"],
                }
            )

    return {output_field: frame_detections}
