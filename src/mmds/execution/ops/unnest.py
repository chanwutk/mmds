from __future__ import annotations

from collections.abc import Iterable, Iterator

from ...model import DatasetExpr, MMDSValidationError, Row


def _apply_unnest(node: DatasetExpr, rows: Iterable[Row]) -> Iterator[Row]:
    field = node.field
    if field is None:
        raise MMDSValidationError("Unnest nodes require a field.")

    for row in rows:
        value = row.get(field)
        if value is None:
            if node.keep_empty:
                empty_row = dict(row)
                empty_row[field] = None
                yield empty_row
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                if node.keep_empty:
                    empty_row = dict(row)
                    empty_row[field] = None
                    yield empty_row
                continue
            for item in value:
                expanded = dict(row)
                expanded[field] = item
                yield expanded
            continue

        yield dict(row)
