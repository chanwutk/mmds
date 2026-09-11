from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .dsl import ForEach
from .model import (
    Assignment,
    BuiltinSpec,
    DatasetExpr,
    JsonValue,
    MMDSValidationError,
    PadIntervalSpec,
    PromptSpec,
    QueryProgram,
    Record,
    RecordPath,
    ReconcileIntervalsSpec,
    ResolveSpec,
    UdfSpec,
    ViewSpec,
    normalize_output_schema,
    normalize_group_by,
)

_MMDS_IMPORTS = {
    "Input", "Map", "Filter", "Reduce", "Unnest", "Resolve", "View",
    "PadInterval", "ReconcileIntervals", "Record", "ForEach",
}


def load_query(source: str | Path) -> QueryProgram:
    path: Path | None = None
    if isinstance(source, Path):
        path = source
        text = path.read_text(encoding="utf-8")
    else:
        if "\n" in source or "\r" in source:
            text = source
        else:
            maybe_path = Path(source)
            try:
                exists = maybe_path.exists()
            except OSError:
                exists = False
            if exists:
                path = maybe_path
                text = maybe_path.read_text(encoding="utf-8")
            else:
                text = source
    return parse_query(text, path=path)


def parse_query(source: str, *, path: Path | None = None) -> QueryProgram:
    tree = ast.parse(source, filename=str(path) if path else "<mmds-query>")
    udf_imports: dict[str, UdfSpec] = {}
    assignments: list[Assignment] = []
    bindings: dict[str, DatasetExpr] = {}

    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.ImportFrom):
            _parse_import(node, udf_imports)
            continue
        if isinstance(node, ast.Assign):
            assignment = _parse_assignment(node, bindings, udf_imports)
            bindings[assignment.target] = assignment.expr
            assignments.append(assignment)
            continue
        raise MMDSValidationError(
            f"Unsupported statement {type(node).__name__}. Only imports and top-level assignments are allowed."
        )

    if not assignments:
        raise MMDSValidationError("Queries must contain at least one DSL assignment.")
    return QueryProgram(assignments=tuple(assignments), output_name=assignments[-1].target, path=str(path) if path else None)


def _parse_import(node: ast.ImportFrom, udf_imports: dict[str, UdfSpec]) -> None:
    if node.module is None or node.level != 0:
        raise MMDSValidationError("Only absolute imports are supported in MMDS queries.")
    if node.module == "mmds":
        for alias in node.names:
            if alias.asname is not None:
                raise MMDSValidationError("Aliased imports from mmds are not supported.")
            if alias.name not in _MMDS_IMPORTS:
                raise MMDSValidationError(f"Unsupported import from mmds: {alias.name!r}.")
        return
    if node.module.startswith("udfs"):
        for alias in node.names:
            if alias.asname is not None:
                raise MMDSValidationError("Aliased UDF imports are not supported.")
            udf_imports[alias.name] = UdfSpec(module=node.module, name=alias.name)
        return
    raise MMDSValidationError("Only imports from mmds and udfs.* are supported in MMDS queries.")


def _parse_assignment(
    node: ast.Assign,
    bindings: dict[str, DatasetExpr],
    udf_imports: dict[str, UdfSpec],
) -> Assignment:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        raise MMDSValidationError("Assignments must target a single variable name.")
    target = node.targets[0].id
    expr = _parse_call(node.value, bindings, udf_imports)
    return Assignment(target=target, expr=expr)


def _parse_call(
    node: ast.AST,
    bindings: dict[str, DatasetExpr],
    udf_imports: dict[str, UdfSpec],
) -> DatasetExpr:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise MMDSValidationError("Assignments must call one of the MMDS DSL operators directly.")

    operator = node.func.id
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords:
        raise MMDSValidationError("**kwargs are not supported in MMDS queries.")

    if operator == "Input":
        _expect_args(operator, node.args, 1, keywords, allowed_keywords=set())
        input_path = _parse_string(node.args[0], "Input path")
        _validate_input_path(input_path)
        return DatasetExpr(kind="input", input_path=input_path)

    if operator == "Map":
        _expect_args(operator, node.args, 2, keywords, allowed_keywords={"name", "schema"})
        source = _parse_source(node.args[0], bindings)
        spec = _parse_spec("map", node.args[1], udf_imports, schema_node=keywords.get("schema"))
        return DatasetExpr(kind="map", source=source, spec=spec, name=_parse_optional_name(keywords))

    if operator == "Filter":
        _expect_args(operator, node.args, 2, keywords, allowed_keywords={"name"})
        source = _parse_source(node.args[0], bindings)
        spec = _parse_spec("filter", node.args[1], udf_imports, schema_node=None)
        return DatasetExpr(kind="filter", source=source, spec=spec, name=_parse_optional_name(keywords))

    if operator == "Reduce":
        _expect_args(operator, node.args, 3, keywords, allowed_keywords={"name", "schema"})
        source = _parse_source(node.args[0], bindings)
        group_by = _parse_group_by(node.args[1])
        spec = _parse_spec("reduce", node.args[2], udf_imports, schema_node=keywords.get("schema"))
        return DatasetExpr(
            kind="reduce",
            source=source,
            group_by=group_by,
            spec=spec,
            name=_parse_optional_name(keywords),
        )

    if operator == "Unnest":
        _expect_args(operator, node.args, 2, keywords, allowed_keywords={"keep_empty", "name"})
        source = _parse_source(node.args[0], bindings)
        keep_empty = _parse_bool(keywords.get("keep_empty"), default=False)
        return DatasetExpr(
            kind="unnest",
            source=source,
            field=_parse_string(node.args[1], "Unnest field"),
            keep_empty=keep_empty,
            name=_parse_optional_name(keywords),
        )

    if operator == "Resolve":
        _expect_args(
            operator,
            node.args,
            4,
            keywords,
            allowed_keywords={"strategy", "merge_touching", "name"},
        )
        return DatasetExpr(
            kind="resolve",
            source=_parse_source(node.args[0], bindings),
            spec=ResolveSpec(
                group_by=_parse_group_by(node.args[1]),
                start_field=_parse_string(node.args[2], "Resolve start field"),
                end_field=_parse_string(node.args[3], "Resolve end field"),
                strategy=_parse_string_keyword(
                    keywords.get("strategy"),
                    "Resolve strategy",
                    default="overlap",
                ),
                merge_touching=_parse_bool(
                    keywords.get("merge_touching"), default=True
                ),
            ),
            name=_parse_optional_name(keywords),
        )

    if operator == "View":
        _expect_args(
            operator,
            node.args,
            4,
            keywords,
            allowed_keywords={"output_field", "name"},
        )
        return DatasetExpr(
            kind="view",
            source=_parse_source(node.args[0], bindings),
            spec=ViewSpec(
                video_field=_parse_string(node.args[1], "View video field"),
                start_field=_parse_string(node.args[2], "View start field"),
                end_field=_parse_string(node.args[3], "View end field"),
                output_field=_parse_string_keyword(
                    keywords.get("output_field"),
                    "View output field",
                    default="view",
                ),
            ),
            name=_parse_optional_name(keywords),
        )

    raise MMDSValidationError(f"Unsupported operator {operator!r}.")


def _parse_source(node: ast.AST, bindings: dict[str, DatasetExpr]) -> DatasetExpr:
    if not isinstance(node, ast.Name):
        raise MMDSValidationError("Operator sources must reference a previously assigned variable.")
    if node.id not in bindings:
        raise MMDSValidationError(f"Unknown source variable {node.id!r}.")
    return bindings[node.id]


def _parse_spec(
    op_kind: str,
    node: ast.AST,
    udf_imports: dict[str, UdfSpec],
    *,
    schema_node: ast.AST | None,
) -> PromptSpec | UdfSpec | BuiltinSpec:
    schema = _parse_schema(schema_node)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if schema is not None:
            raise MMDSValidationError(
                "schema= is not valid for built-in deterministic functions."
            )
        if op_kind == "filter":
            raise MMDSValidationError(
                "Filter does not currently accept built-in deterministic functions."
            )
        if node.func.id == "PadInterval":
            return _parse_pad_interval(node)
        if node.func.id == "ReconcileIntervals":
            return _parse_reconcile_intervals(node)

    if isinstance(node, ast.Name) and node.id in udf_imports:
        if schema is not None:
            raise MMDSValidationError("schema= is only valid for prompt-backed operators.")
        return udf_imports[node.id]

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if op_kind in {"map", "reduce"} and schema is None:
            raise MMDSValidationError(f"Prompt-backed {op_kind} operations require schema=...")
        return PromptSpec(parts=(node.value,), output_schema=schema)

    if isinstance(node, (ast.List, ast.Tuple)):
        parts = _parse_prompt_parts(op_kind, node.elts, within_foreach=False)
        if not parts:
            raise MMDSValidationError("Prompt lists cannot be empty.")
        if op_kind in {"map", "reduce"} and schema is None:
            raise MMDSValidationError(f"Prompt-backed {op_kind} operations require schema=...")
        return PromptSpec(parts=parts, output_schema=schema)

    raise MMDSValidationError(
        "Operator semantic specs must be prompt strings, prompt-part lists, "
        "built-in deterministic functions, or imported UDF names."
    )


def _parse_pad_interval(node: ast.Call) -> PadIntervalSpec:
    keywords = _keyword_nodes(node)
    _expect_args(
        "PadInterval",
        node.args,
        3,
        keywords,
        allowed_keywords={
            "input_start_field",
            "input_end_field",
            "output_start_field",
            "output_end_field",
        },
    )
    return PadIntervalSpec(
        interval_field=_parse_string(node.args[0], "PadInterval interval field"),
        duration_field=_parse_string(node.args[1], "PadInterval duration field"),
        padding_seconds=_parse_number(node.args[2], "PadInterval padding"),
        input_start_field=_parse_string_keyword(
            keywords.get("input_start_field"),
            "PadInterval input start field",
            default="start_seconds",
        ),
        input_end_field=_parse_string_keyword(
            keywords.get("input_end_field"),
            "PadInterval input end field",
            default="end_seconds",
        ),
        output_start_field=_parse_string_keyword(
            keywords.get("output_start_field"),
            "PadInterval output start field",
            default="window_start_seconds",
        ),
        output_end_field=_parse_string_keyword(
            keywords.get("output_end_field"),
            "PadInterval output end field",
            default="window_end_seconds",
        ),
    )


def _parse_reconcile_intervals(node: ast.Call) -> ReconcileIntervalsSpec:
    keywords = _keyword_nodes(node)
    _expect_args(
        "ReconcileIntervals",
        node.args,
        3,
        keywords,
        allowed_keywords={
            "output_field",
            "event_start_field",
            "event_end_field",
            "preserve_fields",
            "deduplication_tiou_threshold",
        },
    )
    preserve_node = keywords.get("preserve_fields")
    threshold_node = keywords.get("deduplication_tiou_threshold")
    return ReconcileIntervalsSpec(
        events_field=_parse_string(node.args[0], "ReconcileIntervals events field"),
        window_start_field=_parse_string(
            node.args[1], "ReconcileIntervals window start field"
        ),
        window_end_field=_parse_string(
            node.args[2], "ReconcileIntervals window end field"
        ),
        output_field=_parse_string_keyword(
            keywords.get("output_field"),
            "ReconcileIntervals output field",
            default="events",
        ),
        event_start_field=_parse_string_keyword(
            keywords.get("event_start_field"),
            "ReconcileIntervals event start field",
            default="start_seconds",
        ),
        event_end_field=_parse_string_keyword(
            keywords.get("event_end_field"),
            "ReconcileIntervals event end field",
            default="end_seconds",
        ),
        preserve_fields=(
            ()
            if preserve_node is None
            else _parse_string_sequence(
                preserve_node, "ReconcileIntervals preserve_fields"
            )
        ),
        deduplication_tiou_threshold=(
            0.8
            if threshold_node is None
            else _parse_number(
                threshold_node, "ReconcileIntervals tIoU threshold"
            )
        ),
    )


def _parse_prompt_parts(
    op_kind: str,
    nodes: list[ast.AST],
    *,
    within_foreach: bool,
) -> tuple[str | RecordPath | Any, ...]:
    parts: list[Any] = []
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
            continue
        record_ref = _parse_record_ref(node)
        if record_ref is not None:
            if op_kind == "reduce" and not within_foreach:
                raise MMDSValidationError(
                    "Reduce prompts must access row fields through ForEach([...])."
                )
            parts.append(record_ref)
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ForEach":
            if op_kind != "reduce" or within_foreach:
                raise MMDSValidationError("ForEach(...) is only valid at the top level of Reduce prompts.")
            if node.keywords:
                raise MMDSValidationError("ForEach(...) does not accept keyword arguments.")
            if len(node.args) != 1 or not isinstance(node.args[0], (ast.List, ast.Tuple)):
                raise MMDSValidationError("ForEach(...) expects a single prompt-part list argument.")
            parts.append(ForEach(_parse_prompt_parts(op_kind, node.args[0].elts, within_foreach=True)))
            continue
        raise MMDSValidationError(
            "Prompt lists may only contain strings, Record[...] references, and Reduce-level ForEach([...])."
        )
    return tuple(parts)


def _parse_record_ref(node: ast.AST) -> RecordPath | None:
    path: list[str] = []
    current = node
    while isinstance(current, ast.Subscript):
        path.append(_parse_string(current.slice, "Record field"))
        current = current.value
    if not isinstance(current, ast.Name) or current.id != "Record":
        return None
    reference = Record
    for field_name in reversed(path):
        reference = reference[field_name]
    if reference.is_root():
        raise MMDSValidationError("Record must select at least one field.")
    return reference


def _parse_group_by(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return normalize_group_by(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_parse_string(element, "group_by field") for element in node.elts]
        return normalize_group_by(values)
    raise MMDSValidationError("Reduce group_by must be a string or a list/tuple of strings.")


def _parse_schema(node: ast.AST | None) -> JsonValue | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise MMDSValidationError(
            "schema= must be a Python literal dictionary of output fields or a legacy object schema."
        ) from exc
    try:
        return normalize_output_schema(value)
    except TypeError as exc:
        raise MMDSValidationError(str(exc)) from exc


def _validate_input_path(path: str) -> None:
    if not (path.endswith(".json") or path.endswith(".jsonl")):
        raise MMDSValidationError("Input paths must point to .json or .jsonl files.")


def _parse_string(node: ast.AST, label: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise MMDSValidationError(f"{label} must be a string literal.")
    return node.value


def _parse_bool(node: ast.AST | None, *, default: bool) -> bool:
    if node is None:
        return default
    if not isinstance(node, ast.Constant) or not isinstance(node.value, bool):
        raise MMDSValidationError("Boolean operator flags must be literal True/False values.")
    return node.value


def _parse_number(node: ast.AST, label: str) -> float:
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise MMDSValidationError(f"{label} must be a numeric literal.")
    return float(node.value)


def _parse_string_keyword(node: ast.AST | None, label: str, *, default: str) -> str:
    return default if node is None else _parse_string(node, label)


def _parse_string_sequence(node: ast.AST, label: str) -> tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise MMDSValidationError(f"{label} must be a list or tuple of strings.")
    return tuple(_parse_string(item, label) for item in node.elts)


def _keyword_nodes(node: ast.Call) -> dict[str, ast.AST]:
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords:
        raise MMDSValidationError("**kwargs are not supported in MMDS queries.")
    return keywords


def _parse_optional_name(keywords: dict[str, ast.AST]) -> str | None:
    node = keywords.get("name")
    if node is None:
        return None
    return _parse_string(node, "Operator name")


def _expect_args(
    operator: str,
    args: list[ast.AST],
    expected: int,
    keywords: dict[str, ast.AST],
    *,
    allowed_keywords: set[str],
) -> None:
    if len(args) != expected:
        raise MMDSValidationError(f"{operator} expects exactly {expected} positional arguments.")
    unexpected = set(keywords) - allowed_keywords
    if unexpected:
        raise MMDSValidationError(f"{operator} does not support keyword arguments: {sorted(unexpected)!r}.")
