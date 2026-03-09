from __future__ import annotations

import keyword
import re
from collections import defaultdict
from typing import Any

from .model import (
    Assignment,
    DatasetExpr,
    ForEachPrompt,
    JsonValue,
    PromptSpec,
    QueryProgram,
    RecordPath,
    UdfSpec,
)


def render_query(plan_or_query: DatasetExpr | QueryProgram) -> str:
    program = plan_or_query if isinstance(plan_or_query, QueryProgram) else program_from_plan(plan_or_query)
    lines = ["from mmds import Input, Map, Filter, Reduce, Unnest, Record, ForEach"]

    grouped_udfs: dict[str, list[str]] = defaultdict(list)
    for spec in program.used_udfs():
        grouped_udfs[spec.module].append(spec.name)
    for module in sorted(grouped_udfs):
        names = ", ".join(sorted(grouped_udfs[module]))
        lines.append(f"from {module} import {names}")

    if program.assignments:
        lines.append("")

    node_names = {assignment.expr: assignment.target for assignment in program.assignments}
    for assignment in program.assignments:
        lines.append(f"{assignment.target} = {_render_expr(assignment.expr, node_names)}")

    return "\n".join(lines).rstrip() + "\n"


def program_from_plan(plan: DatasetExpr) -> QueryProgram:
    nodes = list(plan.walk_postorder())
    assignments: list[Assignment] = []
    used_names: set[str] = set()
    node_names: dict[DatasetExpr, str] = {}
    step_index = 1

    for node in nodes:
        if node is plan:
            target = "output"
        elif node.kind == "input":
            target = _make_unique_name(used_names, f"source_{_sanitize_identifier(node.input_name or 'input')}")
        else:
            target = _make_unique_name(used_names, f"step_{step_index}")
            step_index += 1
        used_names.add(target)
        node_names[node] = target
        assignments.append(Assignment(target=target, expr=node))

    return QueryProgram(assignments=tuple(assignments), output_name=node_names[plan])


def _render_expr(expr: DatasetExpr, node_names: dict[DatasetExpr, str]) -> str:
    if expr.kind == "input":
        return f"Input({_quote(expr.input_name)})"

    source_name = node_names[expr.source]
    name_suffix = _render_name_suffix(expr.name)
    if expr.kind == "map":
        return f"Map({source_name}, {_render_spec(expr.spec, include_schema=True)}{name_suffix})"
    if expr.kind == "filter":
        return f"Filter({source_name}, {_render_spec(expr.spec, include_schema=False)}{name_suffix})"
    if expr.kind == "reduce":
        group_by = _render_group_by(expr.group_by)
        return f"Reduce({source_name}, {group_by}, {_render_spec(expr.spec, include_schema=True)}{name_suffix})"
    if expr.kind == "unnest":
        flags = [_quote(expr.field)]
        if expr.keep_empty:
            flags.append("keep_empty=True")
        if expr.name is not None:
            flags.append(f"name={_quote(expr.name)}")
        return f"Unnest({source_name}, {', '.join(flags)})"
    raise ValueError(f"Unsupported operator kind {expr.kind!r}.")


def _render_spec(spec: PromptSpec | UdfSpec | None, *, include_schema: bool) -> str:
    if isinstance(spec, PromptSpec):
        prompt = _render_prompt_spec(spec)
        if include_schema:
            if spec.output_schema is None:
                raise ValueError("Prompt-backed Map/Reduce specs require an output schema.")
            prompt = f"{prompt}, schema={_render_literal(spec.output_schema)}"
        return prompt
    if isinstance(spec, UdfSpec):
        return spec.name
    raise ValueError("Expected a prompt or UDF spec.")


def _render_prompt_spec(spec: PromptSpec) -> str:
    if len(spec.parts) == 1 and isinstance(spec.parts[0], str):
        return _quote(spec.parts[0])
    rendered = ", ".join(_render_prompt_part(part) for part in spec.parts)
    return f"[{rendered}]"


def _render_prompt_part(part: str | RecordPath | ForEachPrompt) -> str:
    if isinstance(part, str):
        return _quote(part)
    if isinstance(part, RecordPath):
        if part.is_root():
            raise ValueError("Cannot render an empty Record path.")
        rendered = "Record"
        for field_name in part.path:
            rendered += f"[{_quote(field_name)}]"
        return rendered
    if isinstance(part, ForEachPrompt):
        body = ", ".join(_render_prompt_part(child) for child in part.parts)
        return f"ForEach([{body}])"
    raise ValueError(f"Unsupported prompt part {part!r}.")


def _render_group_by(group_by: tuple[str, ...]) -> str:
    if len(group_by) == 1:
        return _quote(group_by[0])
    rendered = ", ".join(_quote(field) for field in group_by)
    return f"[{rendered}]"


def _render_name_suffix(name: str | None) -> str:
    if name is None:
        return ""
    return f", name={_quote(name)}"


def _render_literal(value: JsonValue) -> str:
    if isinstance(value, str):
        return _quote(value)
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_render_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{_quote(str(key))}: {_render_literal(item)}"
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
        return "{" + items + "}"
    raise ValueError(f"Unsupported literal value {value!r}.")


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"\W+", "_", value).strip("_") or "input"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    if keyword.iskeyword(sanitized):
        sanitized = f"{sanitized}_value"
    return sanitized


def _make_unique_name(used_names: set[str], base: str) -> str:
    if base not in used_names:
        return base
    index = 2
    while f"{base}_{index}" in used_names:
        index += 1
    return f"{base}_{index}"


def _quote(value: str | None) -> str:
    if value is None:
        raise ValueError("Cannot render a missing string value.")
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'
