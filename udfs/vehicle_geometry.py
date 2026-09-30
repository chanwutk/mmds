"""Shared vehicle geometry helpers for detection and tracking UDFs."""

from __future__ import annotations

VEHICLE_CLASSES = frozenset({"sedan", "suv", "truck"})


def bbox_iou(left: list[float], right: list[float]) -> float:
    """Intersection over union between two ``[x1, y1, x2, y2]`` boxes."""
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


__all__ = [
    "VEHICLE_CLASSES",
    "bbox_iou",
]
