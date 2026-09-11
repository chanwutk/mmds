from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..model import (
    DatasetExpr,
    MMDSValidationError,
    QueryProgram,
    Row,
)
from ._spec import PromptExecutor, StaticPromptExecutor
from .context import ExecutionContext, ExecutionStats, OperatorExecutionStat
from .concurrency import bounded_map
from .ops.filter import _apply_filter
from .ops.map import _apply_map
from .ops.reduce import _apply_reduce
from .ops.resolve import _apply_resolve
from .ops.unnest import _apply_unnest
from .ops.view import _apply_view

# NOTE: `.ops.detect` is intentionally NOT imported here. It pulls in the
# OpenCV/NumPy (and, at run time, torch/ultralytics) stack via
# `mmds.utilities.video`. Importing it lazily inside the "detect" branch keeps
# `import mmds` usable without the heavy CV dependencies installed.


def execute(
    plan_or_query: DatasetExpr | QueryProgram,
    prompt_executor: PromptExecutor | None = None,
    *,
    context: ExecutionContext | None = None,
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
    execution_context = context or ExecutionContext()
    return list(
        _execute_node(
            plan,
            prompt_executor,
            context=execution_context,
            base_path=base_path,
        )
    )


def _execute_node(
    node: DatasetExpr,
    prompt_executor: PromptExecutor | None,
    *,
    context: ExecutionContext,
    base_path: Path | None,
) -> Iterator[Row]:
    if node.kind == "input":
        started = time.perf_counter()
        rows = _load_input_rows(node.input_path, base_path=base_path)
        _record_operator(context, node, 0, len(rows), time.perf_counter() - started)
        yield from rows
        return

    source = _execute_node(
        node.source, prompt_executor, context=context, base_path=base_path
    )
    if node.kind == "map":
        input_rows = output_rows = 0
        processing_seconds = 0.0

        def apply(row: Row) -> tuple[Row, float]:
            started = time.perf_counter()
            result = _apply_map(node, row, prompt_executor, context.stats)
            return result, time.perf_counter() - started

        for result, elapsed in bounded_map(
            apply, source, max_workers=context.max_workers
        ):
            input_rows += 1
            output_rows += 1
            processing_seconds += elapsed
            yield result
        _record_operator(context, node, input_rows, output_rows, processing_seconds)
    elif node.kind == "filter":
        input_rows = output_rows = 0
        processing_seconds = 0.0

        def apply_filter(row: Row) -> tuple[Row, bool, float]:
            started = time.perf_counter()
            keep = _apply_filter(node, row, prompt_executor, context.stats)
            return row, keep, time.perf_counter() - started

        for row, keep, elapsed in bounded_map(
            apply_filter, source, max_workers=context.max_workers
        ):
            input_rows += 1
            processing_seconds += elapsed
            if keep:
                output_rows += 1
                yield row
        _record_operator(context, node, input_rows, output_rows, processing_seconds)
    elif node.kind == "reduce":
        rows = list(source)
        started = time.perf_counter()
        output = list(
            _apply_reduce(
                node,
                rows,
                prompt_executor,
                max_workers=context.max_workers,
                execution_stats=context.stats,
            )
        )
        _record_operator(
            context, node, len(rows), len(output), time.perf_counter() - started
        )
        yield from output
    elif node.kind == "unnest":
        input_rows = output_rows = 0
        processing_seconds = 0.0
        for row in source:
            input_rows += 1
            started = time.perf_counter()
            output = list(_apply_unnest(node, [row]))
            processing_seconds += time.perf_counter() - started
            output_rows += len(output)
            yield from output
        _record_operator(context, node, input_rows, output_rows, processing_seconds)
    elif node.kind == "resolve":
        rows = list(source)
        started = time.perf_counter()
        output = _apply_resolve(node, rows)
        _record_operator(
            context, node, len(rows), len(output), time.perf_counter() - started
        )
        yield from output
    elif node.kind == "view":
        input_rows = output_rows = 0
        processing_seconds = 0.0
        for row in source:
            input_rows += 1
            started = time.perf_counter()
            yield _apply_view(node, row, context, base_path=base_path)
            processing_seconds += time.perf_counter() - started
            output_rows += 1
        _record_operator(context, node, input_rows, output_rows, processing_seconds)
    elif node.kind == "detect":
        from .ops.detect import _apply_detect  # lazy: pulls OpenCV/NumPy only when used

        input_rows = output_rows = 0
        processing_seconds = 0.0
        for row in source:
            input_rows += 1
            started = time.perf_counter()
            yield _apply_detect(node, row)
            processing_seconds += time.perf_counter() - started
            output_rows += 1
        _record_operator(context, node, input_rows, output_rows, processing_seconds)
    else:
        raise MMDSValidationError(f"Unsupported operator kind {node.kind!r}.")


def _record_operator(
    context: ExecutionContext,
    node: DatasetExpr,
    input_rows: int,
    output_rows: int,
    elapsed_seconds: float,
) -> None:
    context.stats.record_operator(
        OperatorExecutionStat(
            kind=node.kind,
            name=node.name,
            input_rows=input_rows,
            output_rows=output_rows,
            elapsed_seconds=elapsed_seconds,
        )
    )


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
    "ExecutionContext",
    "ExecutionStats",
    "PromptExecutor",
    "StaticPromptExecutor",
]
