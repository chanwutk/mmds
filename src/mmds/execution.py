from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

from .model import (
    DatasetExpr,
    ForEachPrompt,
    MMDSValidationError,
    PromptSpec,
    QueryProgram,
    RecordPath,
    ResolvedPrompt,
    Row,
    UdfSpec,
)


class PromptExecutor(Protocol):
    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any: ...


class StaticPromptExecutor:
    """Deterministic prompt executor used for tests and local development."""

    def __init__(self, handlers: Mapping[tuple[str, str], Any]) -> None:
        self._handlers = dict(handlers)

    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any:
        key = (op_type, prompt.cache_key())
        if key not in self._handlers:
            raise MMDSValidationError(f"No prompt handler registered for {op_type!r} and {prompt.cache_key()!r}.")
        handler = self._handlers[key]
        if callable(handler):
            return handler(resolved_prompt, payload, context)
        return handler


def execute(
    plan_or_query: DatasetExpr | QueryProgram,
    inputs: Mapping[str, Iterable[Mapping[str, Any]]],
    prompt_executor: PromptExecutor | None = None,
) -> list[Row]:
    if isinstance(plan_or_query, QueryProgram):
        plan = plan_or_query.output_expr
    elif isinstance(plan_or_query, DatasetExpr):
        plan = plan_or_query
    else:
        raise TypeError("execute() expects a DatasetExpr or QueryProgram.")
    return _execute_node(plan, inputs, prompt_executor)


def _execute_node(
    node: DatasetExpr,
    inputs: Mapping[str, Iterable[Mapping[str, Any]]],
    prompt_executor: PromptExecutor | None,
) -> list[Row]:
    if node.kind == "input":
        if node.input_name not in inputs:
            raise MMDSValidationError(f"Missing input dataset {node.input_name!r}.")
        return [_coerce_row(row) for row in inputs[node.input_name]]

    source_rows = _execute_node(node.source, inputs, prompt_executor)
    if node.kind == "map":
        return [_apply_map(node, row, prompt_executor) for row in source_rows]
    if node.kind == "filter":
        return [row for row in source_rows if _apply_filter(node, row, prompt_executor)]
    if node.kind == "reduce":
        return _apply_reduce(node, source_rows, prompt_executor)
    if node.kind == "unnest":
        return _apply_unnest(node, source_rows)
    raise MMDSValidationError(f"Unsupported operator kind {node.kind!r}.")


def _coerce_row(value: Mapping[str, Any]) -> Row:
    if not isinstance(value, Mapping):
        raise TypeError("Execution inputs must be iterables of mapping-like rows.")
    return dict(value)


def _apply_map(node: DatasetExpr, row: Row, prompt_executor: PromptExecutor | None) -> Row:
    updates = _execute_spec(node, row, prompt_executor)
    if not isinstance(updates, Mapping):
        raise MMDSValidationError("Map operations must return mapping-like field updates.")
    result = dict(row)
    result.update(dict(updates))
    return result


def _apply_filter(node: DatasetExpr, row: Row, prompt_executor: PromptExecutor | None) -> bool:
    return bool(_execute_spec(node, row, prompt_executor))


def _apply_reduce(
    node: DatasetExpr,
    rows: list[Row],
    prompt_executor: PromptExecutor | None,
) -> list[Row]:
    groups: dict[tuple[Any, ...], list[Row]] = {}
    if node.group_by == ("_all",):
        groups[()] = [dict(row) for row in rows]
    else:
        for row in rows:
            key = tuple(row.get(field) for field in node.group_by)
            groups.setdefault(key, []).append(dict(row))

    outputs: list[Row] = []
    for key, group_rows in groups.items():
        aggregate = _execute_spec(node, group_rows, prompt_executor)
        if not isinstance(aggregate, Mapping):
            raise MMDSValidationError("Reduce operations must return mapping-like aggregate rows.")
        output: Row = {}
        if node.group_by != ("_all",):
            output.update(dict(zip(node.group_by, key, strict=True)))
        output.update(dict(aggregate))
        outputs.append(output)
    return outputs


def _apply_unnest(node: DatasetExpr, rows: list[Row]) -> list[Row]:
    outputs: list[Row] = []
    field = node.field
    if field is None:
        raise MMDSValidationError("Unnest nodes require a field.")

    for row in rows:
        value = row.get(field)
        if value is None:
            if node.keep_empty:
                empty_row = dict(row)
                empty_row[field] = None
                outputs.append(empty_row)
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                if node.keep_empty:
                    empty_row = dict(row)
                    empty_row[field] = None
                    outputs.append(empty_row)
                continue
            for item in value:
                expanded = dict(row)
                expanded[field] = item
                outputs.append(expanded)
            continue

        outputs.append(dict(row))

    return outputs


def _execute_spec(
    node: DatasetExpr,
    payload: Any,
    prompt_executor: PromptExecutor | None,
) -> Any:
    spec = node.spec
    if isinstance(spec, PromptSpec):
        if prompt_executor is None:
            raise MMDSValidationError(
                f"{node.kind} uses a prompt spec but no prompt executor was provided."
            )
        resolved_prompt = _resolve_prompt(node.kind, spec, payload)
        return prompt_executor.execute(
            node.kind,
            spec,
            resolved_prompt,
            payload,
            {"operator_name": node.name, "group_by": node.group_by, "field": node.field},
        )
    if isinstance(spec, UdfSpec):
        udf = spec.load()
        return udf(payload)
    raise MMDSValidationError(f"{node.kind} requires a semantic spec.")


def _resolve_prompt(op_kind: str, prompt: PromptSpec, payload: Any) -> ResolvedPrompt:
    resolved_parts: list[Any] = []
    for part in prompt.parts:
        resolved_parts.extend(_resolve_prompt_part(op_kind, part, payload))
    return ResolvedPrompt(parts=tuple(resolved_parts), output_schema=prompt.output_schema)


def _resolve_prompt_part(op_kind: str, part: str | RecordPath | ForEachPrompt, payload: Any) -> list[Any]:
    if isinstance(part, str):
        return [part]
    if isinstance(part, RecordPath):
        if op_kind == "reduce":
            raise MMDSValidationError("Reduce prompts must access row fields through ForEach(...).")
        if not isinstance(payload, Mapping):
            raise MMDSValidationError("Record field references require a mapping-like row payload.")
        return [_resolve_record_path(payload, part)]
    if isinstance(part, ForEachPrompt):
        if op_kind != "reduce":
            raise MMDSValidationError("ForEach(...) is only valid for Reduce prompts.")
        if not isinstance(payload, list):
            raise MMDSValidationError("Reduce prompt expansion requires the grouped payload rows.")
        resolved: list[Any] = []
        for row in payload:
            if not isinstance(row, Mapping):
                raise MMDSValidationError("Reduce grouped rows must be mapping-like values.")
            for nested in part.parts:
                if isinstance(nested, ForEachPrompt):
                    raise MMDSValidationError("Nested ForEach(...) expressions are not supported.")
                if isinstance(nested, str):
                    resolved.append(nested)
                else:
                    resolved.append(_resolve_record_path(row, nested))
        return resolved
    raise MMDSValidationError(f"Unsupported prompt part {part!r}.")


def _resolve_record_path(row: Mapping[str, Any], reference: RecordPath) -> Any:
    if reference.is_root():
        raise MMDSValidationError("Record must select at least one field.")
    value: Any = row
    for field_name in reference.path:
        if not isinstance(value, Mapping) or field_name not in value:
            dotted = ".".join(reference.path)
            raise MMDSValidationError(f"Prompt field reference {dotted!r} is missing from the row payload.")
        value = value[field_name]
    return value
