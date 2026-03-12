from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from ..model import (
    DatasetExpr,
    ForEachPrompt,
    MMDSValidationError,
    PromptSpec,
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
