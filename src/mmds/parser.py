from __future__ import annotations

import ast
from pathlib import Path

from .model import Assignment, DatasetExpr, MMDSValidationError, QueryProgram, PromptSpec, UdfSpec, normalize_group_by

_OPERATORS = {"Input", "Map", "Filter", "Reduce", "Unnest"}


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
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
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
            if alias.name not in _OPERATORS:
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
        return DatasetExpr(kind="input", input_name=_parse_string(node.args[0], "Input name"))

    if operator in {"Map", "Filter"}:
        _expect_args(operator, node.args, 2, keywords, allowed_keywords={"name"})
        source = _parse_source(node.args[0], bindings)
        spec = _parse_spec(node.args[1], udf_imports)
        return DatasetExpr(kind=operator.lower(), source=source, spec=spec, name=_parse_optional_name(keywords))

    if operator == "Reduce":
        _expect_args(operator, node.args, 3, keywords, allowed_keywords={"name"})
        source = _parse_source(node.args[0], bindings)
        group_by = _parse_group_by(node.args[1])
        spec = _parse_spec(node.args[2], udf_imports)
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

    raise MMDSValidationError(f"Unsupported operator {operator!r}.")


def _parse_source(node: ast.AST, bindings: dict[str, DatasetExpr]) -> DatasetExpr:
    if not isinstance(node, ast.Name):
        raise MMDSValidationError("Operator sources must reference a previously assigned variable.")
    if node.id not in bindings:
        raise MMDSValidationError(f"Unknown source variable {node.id!r}.")
    return bindings[node.id]


def _parse_spec(node: ast.AST, udf_imports: dict[str, UdfSpec]) -> PromptSpec | UdfSpec:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return PromptSpec(node.value)
    if isinstance(node, ast.Name) and node.id in udf_imports:
        return udf_imports[node.id]
    raise MMDSValidationError("Operator semantic specs must be prompt strings or imported UDF names.")


def _parse_group_by(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return normalize_group_by(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_parse_string(element, "group_by field") for element in node.elts]
        return normalize_group_by(values)
    raise MMDSValidationError("Reduce group_by must be a string or a list/tuple of strings.")


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
