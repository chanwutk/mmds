from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..model import (
    DatasetExpr,
    MMDSValidationError,
    QueryProgram,
    Row,
)
from ._spec import PromptExecutor, StaticPromptExecutor
from .ops.detect import _apply_detect
from .ops.filter import _apply_filter
from .ops.map import _apply_map
from .ops.reduce import _apply_reduce
from .ops.unnest import _apply_unnest


def execute(
    plan_or_query: DatasetExpr | QueryProgram,
    prompt_executor: PromptExecutor | None = None,
) -> list[Row]:
    base_path: Path | None = None
    if isinstance(plan_or_query, QueryProgram):
        plan = plan_or_query.output_expr
        if plan_or_query.path is not None:
            base_path = Path(plan_or_query.path).resolve().parent
    elif isinstance(plan_or_query, DatasetExpr):
        plan = plan_or_query
    else:
        raise TypeError("execute() expects a DatasetExpr or QueryProgram.")
    return list(_execute_node(plan, prompt_executor, base_path=base_path))


def _execute_node(
    node: DatasetExpr,
    prompt_executor: PromptExecutor | None,
    *,
    base_path: Path | None,
) -> Iterator[Row]:
    if node.kind == "input":
        yield from _load_input_rows(node.input_path, base_path=base_path)
        return

    source = _execute_node(node.source, prompt_executor, base_path=base_path)
    if node.kind == "map":
        with ThreadPoolExecutor() as ex:
            yield from ex.map(lambda row: _apply_map(node, row, prompt_executor), source)
    elif node.kind == "filter":
        with ThreadPoolExecutor() as ex:
            for row, keep in ex.map(lambda r: (r, _apply_filter(node, r, prompt_executor)), source):
                if keep:
                    yield row
    elif node.kind == "reduce":
        yield from _apply_reduce(node, list(source), prompt_executor)
    elif node.kind == "unnest":
        yield from _apply_unnest(node, source)
    elif node.kind == "detect":
        for row in source:
            yield _apply_detect(node, row)
    else:
        raise MMDSValidationError(f"Unsupported operator kind {node.kind!r}.")


def _load_input_rows(input_path: str | None, *, base_path: Path | None) -> list[Row]:
    if input_path is None:
        raise MMDSValidationError("Input nodes require a file path.")
    path = Path(input_path)
    if not path.is_absolute():
        path = (base_path or Path.cwd()) / path
    if not path.exists():
        raise MMDSValidationError(f"Input file {str(path)!r} does not exist.")
    if path.suffix not in {".json", ".jsonl"}:
        raise MMDSValidationError("Input files must use .json or .jsonl extensions.")
    if path.suffix == ".json":
        return _load_json_rows(path)
    return _load_jsonl_rows(path)


def _load_json_rows(path: Path) -> list[Row]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MMDSValidationError(f"Input file {str(path)!r} is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise MMDSValidationError(f"JSON input file {str(path)!r} must contain a top-level list of records.")
    return [_coerce_row(item) for item in payload]


def _load_jsonl_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MMDSValidationError(
                f"Input file {str(path)!r} contains invalid JSON on line {line_number}."
            ) from exc
        rows.append(_coerce_row(payload))
    return rows


def _coerce_row(value: Mapping[str, Any]) -> Row:
    if not isinstance(value, Mapping):
        raise TypeError("Input files must contain mapping-like row objects.")
    return dict(value)


__all__ = [
    "execute",
    "PromptExecutor",
    "StaticPromptExecutor",
]