from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from math import isfinite
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, Row, ViewBudgetSpec


def _apply_view_budget(node: DatasetExpr, rows: Iterable[Row]) -> Iterator[Row]:
    spec = node.spec
    if not isinstance(spec, ViewBudgetSpec):
        raise MMDSValidationError("ViewBudget requires a ViewBudgetSpec.")

    usage: dict[tuple[Any, ...], tuple[int, float]] = {}
    for row in rows:
        try:
            key = tuple(row[field] for field in spec.group_by)
        except KeyError as exc:
            raise MMDSValidationError(
                f"ViewBudget grouping field {exc.args[0]!r} is missing."
            ) from exc
        try:
            count, seconds = usage.get(key, (0, 0.0))
        except TypeError as exc:
            raise MMDSValidationError(
                "ViewBudget grouping values must be hashable."
            ) from exc

        view = row.get(spec.field)
        if not isinstance(view, Mapping):
            raise MMDSValidationError(
                f"ViewBudget field {spec.field!r} must contain a video view."
            )
        start = view.get("start")
        end = view.get("end")
        if not _valid_time(start) or not _valid_time(end) or end <= start:
            raise MMDSValidationError(
                "ViewBudget video views require finite numeric start/end values "
                "with end greater than start."
            )

        if count >= spec.max_views or seconds >= spec.max_total_video_seconds:
            continue

        remaining = spec.max_total_video_seconds - seconds
        duration = float(end - start)
        output = dict(row)
        output_view = dict(view)
        if duration > remaining:
            output_view["end"] = float(start) + remaining
            duration = remaining
        output[spec.field] = output_view

        usage[key] = (count + 1, seconds + duration)
        yield output


def _valid_time(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )
