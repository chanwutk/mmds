from __future__ import annotations

from typing import Any

from udfs.vehicle_color_model import predict_color_from_crop, predict_colors_from_crops


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
# Transient key used to carry a box's class through class-agnostic NMS; popped
# before the box is emitted, so it never leaks into output records.
_NMS_CLASS_KEY = "_nms_vehicle_class"

# The named-color vocabulary produced by the heuristic color classifier and the
# learned model's label mapping (see udfs/vehicle_color_model.py). Neutral
# colors (white/silver/gray/black) are decided by brightness; chromatic colors
# by hue — see classify_vehicle_color.
VEHICLE_COLOR_VOCAB: frozenset[str] = frozenset(
    {
        "white",
        "silver",
        "gray",
        "black",
        "beige",
        "yellow",
        "red",
        "green",
        "brown",
        "blue",
    }
)

# Saturation/brightness gates (on a 0..1 scale) separating neutral (achromatic)
# vehicles from chromatic ones. Tuned for traffic-camera crops where paint is
# rarely fully saturated; below _NEUTRAL_SAT_MAX we decide by brightness only.
_NEUTRAL_SAT_MAX = 0.18
_BLACK_VALUE_MAX = 0.25
_GRAY_VALUE_MAX = 0.55
_SILVER_VALUE_MAX = 0.80


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
    Map UDF: class-agnostic per-frame NMS for sedan/suv/truck after ``Detect``.

    Args:
        row: The input row containing the detections.
        iou_threshold: The IoU threshold for suppression.
        detection_field: The field containing the detections.

    Returns:
        List of dicts for selected (kept) boxes, each in the original dict format.

    Performs Non-Maximum Suppression (NMS) on vehicle detections to remove
    duplicate or overlapping boxes. All vehicle-class boxes in a frame are
    pooled and suppressed **together**, regardless of class: YOLOE's open-vocab
    label flickers (e.g. ``sedan`` vs ``suv``) for the same physical vehicle, so
    per-class NMS would keep two overlapping boxes for one car and double-count
    it downstream. The highest-confidence box wins and its class is retained.
    The kept boxes are regrouped by class and sorted by frame index then x.
    """
    detections = row.get(detection_field)
    if not isinstance(detections, list):
        return {detection_field: []}

    # Pool every vehicle box by frame, tagging each with its class so NMS can
    # run across classes and we can regroup afterwards. Class order is captured
    # from first appearance to keep the output group order deterministic.
    class_order: list[str] = []
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in detections:
        if not isinstance(item, dict):
            continue
        vehicle_class = item.get("type")
        bboxes = item.get("bboxes")
        if vehicle_class not in _VEHICLE_CLASSES or not isinstance(bboxes, list):
            continue
        if vehicle_class not in class_order:
            class_order.append(vehicle_class)
        for bbox_entry in bboxes:
            if not isinstance(bbox_entry, dict):
                continue
            frame_id = bbox_entry.get("frame_idx")
            if not isinstance(frame_id, int):
                continue
            tagged = dict(bbox_entry)
            tagged[_NMS_CLASS_KEY] = vehicle_class
            by_frame.setdefault(frame_id, []).append(tagged)

    kept_by_class: dict[str, list[dict[str, Any]]] = {}
    for frame_id in sorted(by_frame):
        for kept in _nms_boxes(by_frame[frame_id], iou_threshold=iou_threshold):
            vehicle_class = kept.pop(_NMS_CLASS_KEY)
            kept_by_class.setdefault(vehicle_class, []).append(kept)

    nmsed: list[dict[str, Any]] = []
    for vehicle_class in class_order:
        kept_bboxes = kept_by_class.get(vehicle_class)
        if not kept_bboxes:
            continue
        kept_bboxes.sort(
            key=lambda entry: (entry.get("frame_idx", 0), entry.get("bbox", [0])[0])
        )
        nmsed.append({"type": vehicle_class, "bboxes": kept_bboxes})

    return {detection_field: nmsed}


def classify_vehicle_color(rgb: tuple[float, float, float]) -> str:
    """Map an ``(r, g, b)`` triple (0..255) to a named vehicle color via HSV.

    Saturation decides neutral vs. chromatic: low-saturation pixels are
    white/silver/gray/black by brightness alone (paint, glass, and shadow are
    achromatic), while saturated pixels are named by hue. This avoids the
    everything-looks-gray failure of averaging raw RGB.
    """
    import colorsys

    red, green, blue = (max(0.0, min(255.0, float(value))) for value in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    hue_deg = hue * 360.0

    # Achromatic: decide by brightness only.
    if saturation < _NEUTRAL_SAT_MAX:
        if value < _BLACK_VALUE_MAX:
            return "black"
        if value < _GRAY_VALUE_MAX:
            return "gray"
        if value < _SILVER_VALUE_MAX:
            return "silver"
        return "white"

    # Very dark pixels read as black regardless of a noisy hue.
    if value < _BLACK_VALUE_MAX:
        return "black"

    # Chromatic: name by hue sector, mapped onto the vocabulary.
    if hue_deg < 15.0 or hue_deg >= 330.0:
        return "red"
    if hue_deg < 45.0:
        # Orange/amber: dark & muted reads as brown, otherwise beige.
        return "brown" if value < 0.55 else "beige"
    if hue_deg < 70.0:
        return "yellow"
    if hue_deg < 170.0:
        return "green"
    return "blue"


def vehicle_sub_type_from_geometry(vehicle_class: str, bbox: list[float]) -> str:
    """Coarse subtype from class and box aspect ratio. Helper function for predict_vehicle_attributes."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "sedan"
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(1.0, abs(x2 - x1))
    height = max(1.0, abs(y2 - y1))
    ratio = width / height
    if vehicle_class == "truck":
        if ratio >= 3.2:
            return "tractor_trailer"
        if ratio >= 2.4:
            return "flatbed_truck"
        if ratio >= 1.8:
            return "box_truck"
        return "pickup"
    if ratio >= 2.2:
        return "coupe"
    if ratio <= 1.35:
        return "suv"
    return "sedan" # Default subtype if no match is found


def _dominant_rgb_from_crop(crop: Any) -> tuple[float, float, float] | None:
    """Return a robust ``(r, g, b)`` (0..255) for a BGR vehicle crop, or ``None``.

    Samples the central body region (avoiding background/road at the box edges)
    and takes the **median** per channel — robust to windshield glare, shadow,
    and background bleed, unlike a whole-crop mean.
    """
    import numpy as np

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if getattr(crop, "ndim", 0) != 3 or crop.shape[-1] < 3:
        return None

    height, width = crop.shape[:2]
    # Central 50% vertically, central 40% horizontally: the vehicle body.
    y0, y1 = int(height * 0.25), max(1, int(height * 0.75))
    x0, x1 = int(width * 0.30), max(1, int(width * 0.70))
    region = crop[y0:y1, x0:x1]
    if getattr(region, "size", 0) == 0:
        region = crop

    median = np.median(region.reshape(-1, region.shape[-1]), axis=0)
    if len(median) < 3:
        return None
    blue, green, red = float(median[0]), float(median[1]), float(median[2])
    return (red, green, blue)


def predict_vehicle_attributes(
    vehicle_class: str,
    bbox: list[float],
    crop: Any | None,
) -> dict[str, str]:
    """Return ``vehicle_color`` and ``vehicle_sub_type`` for a vehicle box.

    Color prefers the learned MobileNetV3 classifier
    (:func:`udfs.vehicle_color_model.predict_color_from_crop`); when its weights
    are absent or inference is unavailable it returns ``None`` and we fall back
    to the HSV body-region heuristic (:func:`classify_vehicle_color`).
    """
    color = predict_color_from_crop(crop)
    if color is None:
        dominant_rgb = _dominant_rgb_from_crop(crop)
        color = classify_vehicle_color(dominant_rgb) if dominant_rgb is not None else "gray"
    sub_type = vehicle_sub_type_from_geometry(vehicle_class, bbox)
    return {
        "vehicle_color": color,
        "vehicle_sub_type": sub_type,
    }


def predict_vehicle_attributes_batch(
    vehicle_classes: list[str],
    bboxes: list[list[float]],
    crops: list[Any],
    *,
    batch_size: int = 32,
) -> list[dict[str, str]]:
    """Predict attributes for aligned vehicle boxes with batched color inference."""
    if not (len(vehicle_classes) == len(bboxes) == len(crops)):
        raise ValueError("vehicle_classes, bboxes, and crops must have equal lengths.")
    learned_colors = predict_colors_from_crops(crops, batch_size=batch_size)
    attributes: list[dict[str, str]] = []
    for vehicle_class, bbox, crop, learned_color in zip(
        vehicle_classes, bboxes, crops, learned_colors
    ):
        color = learned_color
        if color is None:
            dominant_rgb = _dominant_rgb_from_crop(crop)
            color = (
                classify_vehicle_color(dominant_rgb)
                if dominant_rgb is not None
                else "gray"
            )
        attributes.append(
            {
                "vehicle_color": color,
                "vehicle_sub_type": vehicle_sub_type_from_geometry(
                    vehicle_class, bbox
                ),
            }
        )
    return attributes


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
    from mmds.utilities.video import read_frames_at_indices

    return read_frames_at_indices(video_path, [frame_id]).get(frame_id)


def _video_path_from_row(row: dict[str, Any], *, video_field: str = "video") -> str | None:
    """Resolve the absolute path to the source video file referenced by this row."""
    from mmds.utilities.media import resolve_video_source
    from mmds.utilities.video import open_video

    raw = row.get(video_field)
    if raw is None:
        return None
    try:
        source = resolve_video_source(raw)
        video = open_video(source)
    except Exception:
        return None
    if isinstance(video, list):
        return None
    return str(video.path)


def crop_from_track_row(row: dict[str, Any], *, video_field: str = "video") -> Any | None:
    """Return the representative BGR crop for a track row, or ``None``.

    Reads the ``rep_frame_id`` frame of the track's source video and slices out
    the ``rep_bbox`` rectangle. Shared by ``crop_ops`` (JPEG crop for Gemini) and
    ``reid_ops`` (appearance embedding) so the extraction lives in one place.
    """
    rep_frame_id = row.get("rep_frame_id")
    rep_bbox = row.get("rep_bbox")
    if not isinstance(rep_frame_id, int):
        return None
    if not isinstance(rep_bbox, list) or len(rep_bbox) != 4:
        return None
    video_path = _video_path_from_row(row, video_field=video_field)
    if not video_path:
        return None
    frame = _read_frame_at_index(video_path, rep_frame_id)
    if frame is None:
        return None
    crop = _crop_from_frame(frame, rep_bbox)
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return crop


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
        from mmds.utilities.video import read_frames_at_indices

        frame_cache = read_frames_at_indices(video_path, boxes_by_frame)

    pending: list[tuple[int, str, list[float], float, Any | None]] = []
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
            pending.append(
                (
                    frame_id,
                    vehicle_class,
                    [float(value) for value in bbox],
                    float(confidence),
                    crop,
                )
            )

    attributes = predict_vehicle_attributes_batch(
        [vehicle_class for _, vehicle_class, _, _, _ in pending],
        [bbox for _, _, bbox, _, _ in pending],
        [crop for _, _, _, _, crop in pending],
    )
    frame_detections: list[dict[str, Any]] = []
    for (frame_id, vehicle_class, bbox, confidence, _), attrs in zip(
        pending, attributes
    ):
        frame_detections.append(
            {
                "frame_id": frame_id,
                "camera_id": camera_id,
                "bbox": bbox,
                "confidence": confidence,
                "vehicle_class": vehicle_class,
                "color": attrs["vehicle_color"],
                "subtype": attrs["vehicle_sub_type"],
            }
        )

    return {output_field: frame_detections}
