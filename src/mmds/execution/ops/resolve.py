from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, ResolveSpec, Row


def _apply_resolve(node: DatasetExpr, rows: Iterable[Row]) -> list[Row]:
    """Coalesce overlapping intervals without crossing explicit group keys."""
    spec = node.spec
    if not isinstance(spec, ResolveSpec):
        raise MMDSValidationError("Resolve nodes require a ResolveSpec.")

    groups: dict[tuple[Any, ...], list[tuple[float, float, Row]]] = {}
    for row in rows:
        missing = [field for field in spec.group_by if field not in row]
        if missing:
            raise MMDSValidationError(
                f"Resolve grouping fields are missing from a row: {missing!r}."
            )
        start = _boundary(row.get(spec.start_field), spec.start_field)
        end = _boundary(row.get(spec.end_field), spec.end_field)
        if start >= end:
            raise MMDSValidationError("Resolve requires every interval to satisfy start < end.")
        key = tuple(row[field] for field in spec.group_by)
        try:
            groups.setdefault(key, []).append((start, end, dict(row)))
        except TypeError as exc:
            raise MMDSValidationError(
                "Resolve grouping fields must contain hashable scalar values."
            ) from exc

    resolved: list[Row] = []
    for intervals in groups.values():
        intervals.sort(key=lambda value: (value[0], value[1]))
        current_start, current_end, current = intervals[0]
        for start, end, row in intervals[1:]:
            overlaps = start < current_end or (
                spec.merge_touching and start == current_end
            )
            if overlaps:
                current_end = max(current_end, end)
                current[spec.start_field] = current_start
                current[spec.end_field] = current_end
                continue
            current[spec.start_field] = current_start
            current[spec.end_field] = current_end
            resolved.append(current)
            current_start, current_end, current = start, end, row
        current[spec.start_field] = current_start
        current[spec.end_field] = current_end
        resolved.append(current)
    return resolved


def _boundary(value: object, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise MMDSValidationError(
            f"Resolve field {field!r} must be a finite non-negative number."
        )
    return float(value)


__all__ = ["_apply_resolve"]
