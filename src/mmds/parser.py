from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .dsl import ForEach
from .model import (
    Assignment,
    DatasetExpr,
    DetectSpec,
    JoinSpec,
    JsonValue,
    MMDSValidationError,
    PromptSpec,
    QueryProgram,
    Record,
    RecordPath,
    UdfSpec,
    WindowSpec,
    normalize_join_keys,
    normalize_output_schema,
    normalize_group_by,
)

_MMDS_IMPORTS = {
    "Input",
    "Map",
    "Filter",
    "Reduce",
    "Unnest",
    "Join",
    "Detect",
    "Window",
    "Coalesce",
    "Record",
    "ForEach",
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
    mmds_imports: set[str] = set()
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
            _parse_import(node, udf_imports, mmds_imports)
            continue
        if isinstance(node, ast.Assign):
            assignment = _parse_assignment(node, bindings, udf_imports, mmds_imports)
            bindings[assignment.target] = assignment.expr
            assignments.append(assignment)
            continue
        raise MMDSValidationError(
            f"Unsupported statement {type(node).__name__}. Only imports and top-level assignments are allowed."
        )

    if not assignments:
        raise MMDSValidationError("Queries must contain at least one DSL assignment.")
    return QueryProgram(assignments=tuple(assignments), output_name=assignments[-1].target, path=str(path) if path else None)


def _parse_import(
    node: ast.ImportFrom,
    udf_imports: dict[str, UdfSpec],
    mmds_imports: set[str],
) -> None:
    if node.module is None or node.level != 0:
        raise MMDSValidationError("Only absolute imports are supported in MMDS queries.")
    if node.module == "mmds":
        for alias in node.names:
            if alias.asname is not None:
                raise MMDSValidationError("Aliased imports from mmds are not supported.")
            if alias.name not in _MMDS_IMPORTS:
                raise MMDSValidationError(f"Unsupported import from mmds: {alias.name!r}.")
            mmds_imports.add(alias.name)
        return
    if node.module.startswith("udfs"):
        for alias in node.names:
            if alias.asname is not None:
                raise MMDSValidationError("Aliased UDF imports are not supported.")
            udf_imports[alias.name] = UdfSpec(module=node.module, name=alias.name)
        return
    raise MMDSValidationError("Only imports from mmds and udfs.* are supported in MMDS queries.")


def _require_mmds_import(name: str, mmds_imports: set[str], *, kind: str) -> None:
    if name not in _MMDS_IMPORTS:
        if kind == "operator":
            raise MMDSValidationError(f"Unsupported operator {name!r}.")
        raise MMDSValidationError(f"Unsupported MMDS helper {name!r}.")
    if name not in mmds_imports:
        raise MMDSValidationError(
            f"{name} must be imported from mmds before it is used."
        )


def _parse_assignment(
    node: ast.Assign,
    bindings: dict[str, DatasetExpr],
    udf_imports: dict[str, UdfSpec],
    mmds_imports: set[str],
) -> Assignment:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        raise MMDSValidationError("Assignments must target a single variable name.")
    target = node.targets[0].id
    expr = _parse_call(node.value, bindings, udf_imports, mmds_imports)
    return Assignment(target=target, expr=expr)


def _parse_call(
    node: ast.AST,
    bindings: dict[str, DatasetExpr],
    udf_imports: dict[str, UdfSpec],
    mmds_imports: set[str],
) -> DatasetExpr:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise MMDSValidationError("Assignments must call one of the MMDS DSL operators directly.")

    operator = node.func.id
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    if None in keywords:
        raise MMDSValidationError("**kwargs are not supported in MMDS queries.")
    _require_mmds_import(operator, mmds_imports, kind="operator")

    if operator == "Input":
        _expect_args(operator, node.args, 1, keywords, allowed_keywords=set())
        input_path = _parse_string(node.args[0], "Input path")
        _validate_input_path(input_path)
        return DatasetExpr(kind="input", input_path=input_path)

    if operator == "Map":
        _expect_args(
            operator,
            node.args,
            2,
            keywords,
            allowed_keywords={"name", "replace", "schema"},
        )
        source = _parse_source(node.args[0], bindings)
        spec = _parse_spec(
            "map",
            node.args[1],
            udf_imports,
            mmds_imports,
            schema_node=keywords.get("schema"),
        )
        return DatasetExpr(
            kind="map",
            source=source,
            spec=spec,
            replace=_parse_bool(keywords.get("replace"), default=False),
            name=_parse_optional_name(keywords),
        )

    if operator == "Filter":
        _expect_args(operator, node.args, 2, keywords, allowed_keywords={"name"})
        source = _parse_source(node.args[0], bindings)
        spec = _parse_spec(
            "filter",
            node.args[1],
            udf_imports,
            mmds_imports,
            schema_node=None,
        )
        return DatasetExpr(kind="filter", source=source, spec=spec, name=_parse_optional_name(keywords))

    if operator == "Reduce":
        _expect_args(operator, node.args, 3, keywords, allowed_keywords={"name", "schema"})
        source = _parse_source(node.args[0], bindings)
        group_by = _parse_group_by(node.args[1])
        spec = _parse_spec(
            "reduce",
            node.args[2],
            udf_imports,
            mmds_imports,
            schema_node=keywords.get("schema"),
        )
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

    if operator == "Detect":
        _expect_args(
            operator,
            node.args,
            3,
            keywords,
            allowed_keywords={
                "model",
                "output_field",
                "frame_stride",
                "conf",
                "imgsz",
                "name",
            },
        )
        return DatasetExpr(
            kind="detect",
            source=_parse_source(node.args[0], bindings),
            spec=DetectSpec(
                video_field=_parse_string(node.args[1], "Detect video_field"),
                classes=_parse_string_sequence(
                    node.args[2],
                    "Detect classes",
                    allow_empty=False,
                ),
                model=_parse_optional_string(
                    keywords.get("model"),
                    label="Detect model",
                    default="yoloe-11s-seg.pt",
                ),
                output_field=_parse_optional_string(
                    keywords.get("output_field"),
                    label="Detect output_field",
                    default="detections",
                ),
                frame_stride=_parse_optional_int(
                    keywords.get("frame_stride"),
                    label="Detect frame_stride",
                    default=1,
                ),
                conf=_parse_number_or_none(
                    keywords.get("conf"),
                    label="Detect conf",
                ),
                imgsz=_parse_int_or_none(
                    keywords.get("imgsz"),
                    label="Detect imgsz",
                ),
            ),
            name=_parse_optional_name(keywords),
        )

    if operator == "Window":
        _expect_args(
            operator,
            node.args,
            5,
            keywords,
            allowed_keywords={"name"},
        )
        return DatasetExpr(
            kind="window",
            source=_parse_source(node.args[0], bindings),
            spec=WindowSpec(
                video_field=_parse_string(node.args[1], "Window video_field"),
                candidate_field=_parse_string(
                    node.args[2], "Window candidate_field"
                ),
                output_field=_parse_string(node.args[3], "Window output_field"),
                padding_time=_parse_number(
                    node.args[4], "Window padding_time"
                ),
            ),
            name=_parse_optional_name(keywords),
        )

    if operator == "Coalesce":
        _expect_args(
            operator,
            node.args,
            3,
            keywords,
            allowed_keywords={"name"},
        )
        field = _parse_string(node.args[2], "Coalesce field")
        if not field:
            raise MMDSValidationError(
                "Coalesce field must be a non-empty string."
            )
        return DatasetExpr(
            kind="coalesce",
            source=_parse_source(node.args[0], bindings),
            group_by=_parse_group_by(node.args[1]),
            field=field,
            name=_parse_optional_name(keywords),
        )

    if operator == "Join":
        allowed_keywords = {
            "on",
            "one_to_one",
            "score",
            "min_score",
            "left_key",
            "right_key",
            "name",
        }
        unexpected = set(keywords) - allowed_keywords
        if unexpected:
            raise MMDSValidationError(
                f"Join does not support keyword arguments: {sorted(unexpected)!r}."
            )
        if len(node.args) not in {2, 3}:
            raise MMDSValidationError(
                "Join expects 2 or 3 positional arguments."
            )
        predicate = (
            _parse_join_udf(node.args[2], udf_imports, label="predicate")
            if len(node.args) == 3
            else None
        )
        score_node = keywords.get("score")
        score = (
            _parse_join_udf(score_node, udf_imports, label="score")
            if score_node is not None
            else None
        )
        return DatasetExpr(
            kind="join",
            source=_parse_source(node.args[0], bindings),
            right_source=_parse_source(node.args[1], bindings),
            spec=JoinSpec(
                keys=_parse_join_keys(keywords.get("on")),
                predicate=predicate,
                one_to_one=_parse_bool(
                    keywords.get("one_to_one"), default=False
                ),
                score=score,
                min_score=_parse_optional_number(keywords.get("min_score")),
                left_key=_parse_join_keys(keywords.get("left_key")),
                right_key=_parse_join_keys(keywords.get("right_key")),
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
    mmds_imports: set[str],
    *,
    schema_node: ast.AST | None,
) -> PromptSpec | UdfSpec:
    schema = _parse_schema(schema_node)

    if isinstance(node, ast.Name) and node.id in udf_imports:
        if schema is not None:
            raise MMDSValidationError("schema= is only valid for prompt-backed operators.")
        return udf_imports[node.id]

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if op_kind in {"map", "reduce"} and schema is None:
            raise MMDSValidationError(f"Prompt-backed {op_kind} operations require schema=...")
        return PromptSpec(parts=(node.value,), output_schema=schema)

    if isinstance(node, (ast.List, ast.Tuple)):
        parts = _parse_prompt_parts(
            op_kind,
            node.elts,
            mmds_imports,
            within_foreach=False,
        )
        if not parts:
            raise MMDSValidationError("Prompt lists cannot be empty.")
        if op_kind in {"map", "reduce"} and schema is None:
            raise MMDSValidationError(f"Prompt-backed {op_kind} operations require schema=...")
        return PromptSpec(parts=parts, output_schema=schema)

    raise MMDSValidationError(
        "Operator semantic specs must be prompt strings, prompt-part lists, or imported UDF names."
    )


def _parse_prompt_parts(
    op_kind: str,
    nodes: list[ast.AST],
    mmds_imports: set[str],
    *,
    within_foreach: bool,
) -> tuple[str | RecordPath | Any, ...]:
    parts: list[Any] = []
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
            continue
        record_ref = _parse_record_ref(node, mmds_imports)
        if record_ref is not None:
            if op_kind == "reduce" and not within_foreach:
                raise MMDSValidationError(
                    "Reduce prompts must access row fields through ForEach([...])."
                )
            parts.append(record_ref)
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ForEach":
            _require_mmds_import("ForEach", mmds_imports, kind="helper")
            if op_kind != "reduce" or within_foreach:
                raise MMDSValidationError("ForEach(...) is only valid at the top level of Reduce prompts.")
            if node.keywords:
                raise MMDSValidationError("ForEach(...) does not accept keyword arguments.")
            if len(node.args) != 1 or not isinstance(node.args[0], (ast.List, ast.Tuple)):
                raise MMDSValidationError("ForEach(...) expects a single prompt-part list argument.")
            parts.append(
                ForEach(
                    _parse_prompt_parts(
                        op_kind,
                        node.args[0].elts,
                        mmds_imports,
                        within_foreach=True,
                    )
                )
            )
            continue
        raise MMDSValidationError(
            "Prompt lists may only contain strings, Record[...] references, and Reduce-level ForEach([...])."
        )
    return tuple(parts)


def _parse_record_ref(node: ast.AST, mmds_imports: set[str]) -> RecordPath | None:
    path: list[str] = []
    current = node
    while isinstance(current, ast.Subscript):
        path.append(_parse_string(current.slice, "Record field"))
        current = current.value
    if not isinstance(current, ast.Name) or current.id != "Record":
        return None
    _require_mmds_import("Record", mmds_imports, kind="helper")
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


def _parse_string_sequence(
    node: ast.AST,
    label: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise MMDSValidationError(f"{label} must be a list/tuple of strings.")
    values = tuple(_parse_string(item, f"{label} entry") for item in node.elts)
    if not allow_empty and not values:
        raise MMDSValidationError(f"{label} must be non-empty.")
    return values


def _parse_optional_string(
    node: ast.AST | None,
    *,
    label: str,
    default: str,
) -> str:
    return default if node is None else _parse_string(node, label)


def _parse_number(node: ast.AST, label: str) -> float:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise MMDSValidationError(f"{label} must be a numeric literal.") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MMDSValidationError(f"{label} must be a numeric literal.")
    return float(value)


def _parse_optional_int(
    node: ast.AST | None,
    *,
    label: str,
    default: int,
) -> int:
    return default if node is None else _parse_int(node, label)


def _parse_int(node: ast.AST, label: str) -> int:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise MMDSValidationError(f"{label} must be an integer literal.") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise MMDSValidationError(f"{label} must be an integer literal.")
    return value


def _parse_number_or_none(node: ast.AST | None, *, label: str) -> float | None:
    if node is None or (
        isinstance(node, ast.Constant) and node.value is None
    ):
        return None
    return _parse_number(node, label)


def _parse_int_or_none(node: ast.AST | None, *, label: str) -> int | None:
    if node is None or (
        isinstance(node, ast.Constant) and node.value is None
    ):
        return None
    return _parse_int(node, label)


def _parse_bool(node: ast.AST | None, *, default: bool) -> bool:
    if node is None:
        return default
    if not isinstance(node, ast.Constant) or not isinstance(node.value, bool):
        raise MMDSValidationError("Boolean operator flags must be literal True/False values.")
    return node.value


def _parse_optional_name(keywords: dict[str, ast.AST]) -> str | None:
    node = keywords.get("name")
    if node is None:
        return None
    return _parse_string(node, "Operator name")


def _parse_join_udf(
    node: ast.AST,
    udf_imports: dict[str, UdfSpec],
    *,
    label: str,
) -> UdfSpec:
    if isinstance(node, ast.Name) and node.id in udf_imports:
        return udf_imports[node.id]
    raise MMDSValidationError(
        f"Join {label} must reference an imported UDF name."
    )


def _parse_join_keys(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    try:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return normalize_join_keys(node.value)
        if isinstance(node, (ast.List, ast.Tuple)):
            return normalize_join_keys(
                tuple(
                    _parse_string(element, "Join key")
                    for element in node.elts
                )
            )
    except (TypeError, MMDSValidationError) as exc:
        raise MMDSValidationError(str(exc)) from exc
    raise MMDSValidationError(
        "Join keys must be a string or a list/tuple of strings."
    )


def _parse_optional_number(node: ast.AST | None) -> float | None:
    if node is None:
        return None
    if (
        not isinstance(node, ast.Constant)
        or isinstance(node.value, bool)
        or not isinstance(node.value, (int, float))
    ):
        raise MMDSValidationError("Join min_score= must be a numeric literal.")
    return float(node.value)


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
