from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, Row
from .._spec import PromptExecutor, _execute_spec


def _apply_reduce(
    node: DatasetExpr,
    rows: list[Row],
    prompt_executor: PromptExecutor | None,
) -> Iterator[Row]:
    groups: dict[tuple[Any, ...], list[Row]] = {}
    if node.group_by == ("_all",):
        groups[()] = [dict(row) for row in rows]
    else:
        for row in rows:
            key = tuple(row.get(field) for field in node.group_by)
            groups.setdefault(key, []).append(dict(row))

    def process_group(item: tuple[tuple[Any, ...], list[Row]]) -> Row:
        key, group_rows = item
        aggregate = _execute_spec(node, group_rows, prompt_executor)
        if not isinstance(aggregate, Mapping):
            raise MMDSValidationError("Reduce operations must return mapping-like aggregate rows.")
        output: Row = {}
        if node.group_by != ("_all",):
            output.update(dict(zip(node.group_by, key, strict=True)))
        output.update(dict(aggregate))
        return output

    with ThreadPoolExecutor() as ex:
        yield from ex.map(process_group, groups.items())
