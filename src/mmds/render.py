from __future__ import annotations

import keyword
import re
from collections import defaultdict
from typing import Any

from .model import (
    Assignment,
    DatasetExpr,
    ForEachPrompt,
    JoinSpec,
    JsonValue,
    PromptSpec,
    QueryProgram,
    RecordPath,
    SplitSpec,
    UdfSpec,
)


def render_query(plan_or_query: DatasetExpr | QueryProgram) -> str:
    program = plan_or_query if isinstance(plan_or_query, QueryProgram) else program_from_plan(plan_or_query)
    lines = ["from mmds import Input, Map, Filter, Reduce, Unnest, Split, Join, Record, ForEach"]

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
            target = _make_unique_name(used_names, f"source_{_sanitize_identifier(_input_label(node.input_path))}")
        else:
            target = _make_unique_name(used_names, f"step_{step_index}")
            step_index += 1
        used_names.add(target)
        node_names[node] = target
        assignments.append(Assignment(target=target, expr=node))

    return QueryProgram(assignments=tuple(assignments), output_name=node_names[plan])


def _render_expr(expr: DatasetExpr, node_names: dict[DatasetExpr, str]) -> str:
    if expr.kind == "input":
        return f"Input({_quote(expr.input_path)})"

    if expr.kind == "join":
        if expr.source is None or expr.right_source is None:
            raise ValueError("Join nodes require left and right sources.")
        if not isinstance(expr.spec, JoinSpec):
            raise ValueError("Join nodes require a JoinSpec.")
        left_name = node_names[expr.source]
        right_name = node_names[expr.right_source]
        name_suffix = _render_name_suffix(expr.name)
        return _render_join_call(left_name, right_name, expr.spec, name_suffix)

    source_name = node_names[expr.source]
    name_suffix = _render_name_suffix(expr.name)
    if expr.kind == "map":
        return _render_map_call(source_name, expr.spec, expr.replace, expr.name)
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
    if expr.kind == "split":
        if not isinstance(expr.spec, SplitSpec):
            raise ValueError("Split nodes require a SplitSpec.")
        return _render_split_call(source_name, expr.spec, expr.name)
    raise ValueError(f"Unsupported operator kind {expr.kind!r}.")


def _render_split_call(
    source_name: str,
    spec: SplitSpec,
    name: str | None,
) -> str:
    flags: list[str] = []
    if spec.chunk_sec != 30.0:
        flags.append(f"chunk_sec={spec.chunk_sec}")
    if spec.doc_id_key != "camera_id":
        flags.append(f"doc_id_key={_quote(spec.doc_id_key)}")
    if spec.output_prefix != "split_video":
        flags.append(f"output_prefix={_quote(spec.output_prefix)}")
    if name is not None:
        flags.append(f"name={_quote(name)}")
    if flags:
        return f"Split({source_name}, {_quote(spec.video_field)}, {', '.join(flags)})"
    return f"Split({source_name}, {_quote(spec.video_field)})"


def _render_join_call(
    left_name: str,
    right_name: str,
    spec: JoinSpec,
    name_suffix: str,
) -> str:
    flags: list[str] = []
    if spec.keys:
        flags.append(f"on={_render_join_keys(spec.keys)}")
    if spec.one_to_one:
        flags.append("one_to_one=True")
    if spec.score is not None:
        flags.append(f"score={spec.score.name}")
    if spec.min_score is not None:
        flags.append(f"min_score={spec.min_score}")
    if spec.left_key:
        flags.append(f"left_key={_render_join_keys(spec.left_key)}")
    if spec.right_key:
        flags.append(f"right_key={_render_join_keys(spec.right_key)}")
    if spec.predicate is None:
        if not flags:
            raise ValueError("Join nodes require on= keys and/or a UDF predicate.")
        return f"Join({left_name}, {right_name}, {', '.join(flags)}{name_suffix})"
    predicate = spec.predicate.name
    if flags:
        return f"Join({left_name}, {right_name}, {predicate}, {', '.join(flags)}{name_suffix})"
    return f"Join({left_name}, {right_name}, {predicate}{name_suffix})"


def _render_join_keys(keys: tuple[str, ...]) -> str:
    if len(keys) == 1:
        return _quote(keys[0])
    rendered = ", ".join(_quote(key) for key in keys)
    return f"({rendered},)"


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


def _render_map_call(
    source_name: str,
    spec: PromptSpec | UdfSpec | None,
    replace: bool,
    name: str | None,
) -> str:
    """Render a Map call with the given source name, spec, replace flag, and name."""
    flags: list[str] = []
    if replace:
        flags.append("replace=True")
    if name is not None:
        flags.append(f"name={_quote(name)}")
    spec_rendered = _render_spec(spec, include_schema=True)
    if flags:
        return f"Map({source_name}, {spec_rendered}, {', '.join(flags)})"
    return f"Map({source_name}, {spec_rendered})"


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


def _input_label(path: str | None) -> str:
    if path is None:
        return "input"
    leaf = path.rsplit("/", 1)[-1]
    if leaf.endswith(".jsonl"):
        return leaf[:-6]
    if leaf.endswith(".json"):
        return leaf[:-5]
    return leaf


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
