from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, Row


def _apply_coalesce(node: DatasetExpr, rows: Iterable[Row]) -> Iterator[Row]:
    field = node.field
    if field is None:
        raise MMDSValidationError("Coalesce requires an interval field.")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in node.group_by)
        interval = row[field]
        if not isinstance(interval, Mapping):
            raise MMDSValidationError(
                f"Coalesce field {field!r} must contain an interval."
            )

        interval = dict(interval)
        start = interval.get("start")
        end = interval.get("end")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            raise MMDSValidationError("Coalesce interval start must be numeric.")
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            raise MMDSValidationError("Coalesce interval end must be numeric.")
        if end <= start:
            raise MMDSValidationError(
                "Coalesce interval end must be greater than start."
            )

        groups.setdefault(key, []).append(interval)

    for key, intervals in groups.items():
        intervals.sort(key=lambda interval: interval["start"])
        current = intervals[0]

        for interval in intervals[1:]:
            if interval["start"] <= current["end"]:
                current["end"] = max(current["end"], interval["end"])
            else:
                yield _output_row(node, field, key, current)
                current = interval

        yield _output_row(node, field, key, current)


def _output_row(
    node: DatasetExpr,
    field: str,
    key: tuple[Any, ...],
    interval: dict[str, Any],
) -> Row:
    result = dict(zip(node.group_by, key, strict=True))
    result[field] = interval
    return result
