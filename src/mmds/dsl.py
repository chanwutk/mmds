from __future__ import annotations

from typing import Any, Callable

from .model import DatasetExpr, PromptSpec, SemanticSpec, UdfSpec, normalize_group_by, udf_spec_from_callable


def Input(name: str) -> DatasetExpr:
    if not isinstance(name, str) or not name:
        raise TypeError("Input names must be non-empty strings.")
    return DatasetExpr(kind="input", input_name=name)


def Map(data: DatasetExpr, spec: str | Callable[..., Any], *, name: str | None = None) -> DatasetExpr:
    return DatasetExpr(kind="map", source=_normalize_source(data), spec=_normalize_spec(spec), name=name)


def Filter(data: DatasetExpr, spec: str | Callable[..., Any], *, name: str | None = None) -> DatasetExpr:
    return DatasetExpr(kind="filter", source=_normalize_source(data), spec=_normalize_spec(spec), name=name)


def Reduce(
    data: DatasetExpr,
    group_by: str | list[str] | tuple[str, ...],
    reducer: str | Callable[..., Any],
    *,
    name: str | None = None,
) -> DatasetExpr:
    return DatasetExpr(
        kind="reduce",
        source=_normalize_source(data),
        spec=_normalize_spec(reducer),
        group_by=normalize_group_by(group_by),
        name=name,
    )


def Unnest(
    data: DatasetExpr,
    field: str,
    *,
    keep_empty: bool = False,
    name: str | None = None,
) -> DatasetExpr:
    if not isinstance(field, str) or not field:
        raise TypeError("Unnest field names must be non-empty strings.")
    return DatasetExpr(
        kind="unnest",
        source=_normalize_source(data),
        field=field,
        keep_empty=bool(keep_empty),
        name=name,
    )


def _normalize_source(data: DatasetExpr) -> DatasetExpr:
    if not isinstance(data, DatasetExpr):
        raise TypeError("Operator sources must be DatasetExpr instances created by the MMDS DSL.")
    return data


def _normalize_spec(spec: str | Callable[..., Any]) -> SemanticSpec:
    if isinstance(spec, str):
        return PromptSpec(spec)
    if callable(spec):
        return udf_spec_from_callable(spec)
    raise TypeError("Semantic specs must be prompt strings or imported UDF callables.")
