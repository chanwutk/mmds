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
