from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from typing import Any, Callable, Iterator, Literal, TypeAlias

Row: TypeAlias = dict[str, Any]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
FieldSchemaValue: TypeAlias = str | dict[str, JsonValue]
RecordSchema: TypeAlias = dict[str, FieldSchemaValue]
OperatorKind: TypeAlias = Literal[
    "input",
    "map",
    "filter",
    "reduce",
    "unnest",
    "detect",
    "window",
    "coalesce",
    "drop_fields",
    "video_map",
    "video_map_each",
    "view_budget",
]


class MMDSValidationError(ValueError):
    """Raised when a query or plan violates the supported MMDS subset."""


@dataclass(frozen=True)
class RecordPath:
    path: tuple[str, ...] = ()

    def __getitem__(self, field_name: str) -> RecordPath:
        if not isinstance(field_name, str) or not field_name:
            raise TypeError("Record field references must use non-empty string keys.")
        return RecordPath(self.path + (field_name,))

    def is_root(self) -> bool:
        return not self.path


Record = RecordPath()


@dataclass(frozen=True)
class ForEachPrompt:
    parts: tuple[Any, ...]


PromptPart: TypeAlias = str | RecordPath | ForEachPrompt


@dataclass(frozen=True)
class PromptSpec:
    parts: tuple[PromptPart, ...]
    output_schema: RecordSchema | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_schema", normalize_output_schema(self.output_schema)
        )

    def cache_key(self) -> str | tuple[Any, ...]:
        if len(self.parts) == 1 and isinstance(self.parts[0], str):
            return self.parts[0]
        return tuple(_prompt_part_cache_key(part) for part in self.parts)

    def __hash__(self) -> int:
        return hash((self.parts, _freeze_json_value(self.output_schema)))


@dataclass(frozen=True)
class ResolvedPrompt:
    parts: tuple[Any, ...]
    output_schema: RecordSchema | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_schema", normalize_output_schema(self.output_schema)
        )


@dataclass(frozen=True)
class UdfSpec:
    module: str
    name: str

    def load(self) -> Callable[..., Any]:
        module = import_module(self.module)
        value = getattr(module, self.name)
        spec = udf_spec_from_callable(value)
        if spec != self:
            raise MMDSValidationError(
                f"Resolved callable {self.module}.{self.name} no longer matches its UDF spec."
            )
        return value


@dataclass(frozen=True)
class DetectSpec:
    """Spec for the Detect operator: runs YOLOE on every frame of a video field."""

    video_field: str
    classes: tuple[str, ...]
    model: str = "yoloe-11s-seg.pt"
    output_field: str = "detections"

    def __post_init__(self) -> None:
        if not self.video_field:
            raise MMDSValidationError(
                "DetectSpec video_field must be a non-empty string."
            )
        if not self.classes:
            raise MMDSValidationError("DetectSpec classes must be non-empty.")
        if any(not isinstance(c, str) or not c for c in self.classes):
            raise MMDSValidationError(
                "DetectSpec classes must all be non-empty strings."
            )
        if not self.model:
            raise MMDSValidationError("DetectSpec model must be a non-empty string.")
        if not self.output_field:
            raise MMDSValidationError(
                "DetectSpec output_field must be a non-empty string."
            )


@dataclass(frozen=True)
class WindowSpec:
    video_field: str
    candidate_field: str
    output_field: str = "clip"
    padding_time: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.video_field, str) or not self.video_field:
            raise MMDSValidationError(
                "WindowSpec video_field must be a non-empty string."
            )

        if not isinstance(self.candidate_field, str) or not self.candidate_field:
            raise MMDSValidationError(
                "WindowSpec candidate_field must be a non-empty string."
            )

        if not isinstance(self.output_field, str) or not self.output_field:
            raise MMDSValidationError(
                "WindowSpec output_field must be a non-empty string."
            )

        padding = self.padding_time
        if (
            not isinstance(padding, (int, float))
            or isinstance(padding, bool)
            or not isfinite(padding)
            or padding < 0
        ):
            raise MMDSValidationError(
                "WindowSpec padding_time must be a finite non-negative number."
            )

        object.__setattr__(self, "padding_time", float(padding))


@dataclass(frozen=True)
class DropFieldsSpec:
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, (list, tuple)):
            raise MMDSValidationError(
                "DropFieldsSpec fields must be a sequence of strings."
            )
        fields = tuple(self.fields)
        if not fields:
            raise MMDSValidationError("DropFieldsSpec requires at least one field.")
        if any(not isinstance(field, str) or not field for field in fields):
            raise MMDSValidationError(
                "DropFieldsSpec fields must be non-empty strings."
            )
        if len(set(fields)) != len(fields):
            raise MMDSValidationError("DropFieldsSpec fields must be unique.")
        object.__setattr__(self, "fields", fields)


@dataclass(frozen=True)
class VideoMapSpec:
    video_field: str
    views_field: str
    group_by: tuple[str, ...]
    map_spec: PromptSpec | UdfSpec
    padding_time: float = 0.0
    max_views: int = 8
    max_total_video_seconds: float = 600.0
    clip_field: str = "clip"

    def __post_init__(self) -> None:
        for label, value in (
            ("video_field", self.video_field),
            ("views_field", self.views_field),
            ("clip_field", self.clip_field),
        ):
            if not isinstance(value, str) or not value:
                raise MMDSValidationError(
                    f"VideoMapSpec {label} must be a non-empty string."
                )
        if len({self.video_field, self.views_field, self.clip_field}) != 3:
            raise MMDSValidationError(
                "VideoMapSpec video, views, and clip fields must be distinct."
            )
        object.__setattr__(self, "group_by", normalize_group_by(self.group_by))
        if not isinstance(self.map_spec, (PromptSpec, UdfSpec)):
            raise MMDSValidationError(
                "VideoMapSpec map_spec must be a PromptSpec or UdfSpec."
            )
        if isinstance(self.map_spec, PromptSpec) and self.map_spec.output_schema is None:
            raise MMDSValidationError(
                "Prompt-backed VideoMap operations require an output schema."
            )
        padding = self.padding_time
        if (
            not isinstance(padding, (int, float))
            or isinstance(padding, bool)
            or not isfinite(padding)
            or padding < 0
        ):
            raise MMDSValidationError(
                "VideoMapSpec padding_time must be a finite non-negative number."
            )
        if (
            not isinstance(self.max_views, int)
            or isinstance(self.max_views, bool)
            or self.max_views <= 0
        ):
            raise MMDSValidationError(
                "VideoMapSpec max_views must be a positive integer."
            )
        total_seconds = self.max_total_video_seconds
        if (
            not isinstance(total_seconds, (int, float))
            or isinstance(total_seconds, bool)
            or not isfinite(total_seconds)
            or total_seconds <= 0
        ):
            raise MMDSValidationError(
                "VideoMapSpec max_total_video_seconds must be a finite positive number."
            )
        object.__setattr__(self, "padding_time", float(padding))
        object.__setattr__(
            self, "max_total_video_seconds", float(total_seconds)
        )


@dataclass(frozen=True)
class ViewBudgetSpec:
    group_by: tuple[str, ...]
    field: str
    max_views: int
    max_total_video_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_by", normalize_group_by(self.group_by))
        if not isinstance(self.field, str) or not self.field:
            raise MMDSValidationError(
                "ViewBudgetSpec field must be a non-empty string."
            )
        if (
            not isinstance(self.max_views, int)
            or isinstance(self.max_views, bool)
            or self.max_views <= 0
        ):
            raise MMDSValidationError(
                "ViewBudgetSpec max_views must be a positive integer."
            )
        seconds = self.max_total_video_seconds
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not isfinite(seconds)
            or seconds <= 0
        ):
            raise MMDSValidationError(
                "ViewBudgetSpec max_total_video_seconds must be a finite positive number."
            )
        object.__setattr__(self, "max_total_video_seconds", float(seconds))


SemanticSpec: TypeAlias = (
    PromptSpec
    | UdfSpec
    | DetectSpec
    | WindowSpec
    | DropFieldsSpec
    | VideoMapSpec
    | ViewBudgetSpec
)


@dataclass(frozen=True)
class DatasetExpr:
    kind: OperatorKind
    source: DatasetExpr | None = None
    input_path: str | None = None
    spec: SemanticSpec | None = None
    group_by: tuple[str, ...] = ()
    field: str | None = None
    keep_empty: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "input":
            if self.source is not None or self.input_path is None:
                raise MMDSValidationError(
                    "Input nodes require only an input file path."
                )
            return
        if self.source is None:
            raise MMDSValidationError(f"{self.kind} nodes require a source expression.")
        if self.kind in {"map", "filter", "reduce"} and self.spec is None:
            raise MMDSValidationError(f"{self.kind} nodes require a semantic spec.")
        if self.kind == "reduce" and not self.group_by:
            raise MMDSValidationError("Reduce nodes require one or more grouping keys.")
        if self.kind == "unnest" and self.field is None:
            raise MMDSValidationError("Unnest nodes require a field to expand.")
        if self.kind == "detect" and not isinstance(self.spec, DetectSpec):
            raise MMDSValidationError("detect nodes require a DetectSpec.")
        if self.kind == "window" and not isinstance(self.spec, WindowSpec):
            raise MMDSValidationError("window nodes require a WindowSpec.")
        if self.kind == "coalesce":
            if not self.group_by:
                raise MMDSValidationError(
                    "Coalesce nodes require one or more grouping fields."
                )
            if self.field is None:
                raise MMDSValidationError(
                    "Coalesce nodes require an interval field."
                )
        if self.kind == "drop_fields" and not isinstance(
            self.spec, DropFieldsSpec
        ):
            raise MMDSValidationError(
                "drop_fields nodes require a DropFieldsSpec."
            )
        if self.kind in {"video_map", "video_map_each"} and not isinstance(
            self.spec, VideoMapSpec
        ):
            raise MMDSValidationError(
                f"{self.kind} nodes require a VideoMapSpec."
            )
        if self.kind == "view_budget" and not isinstance(
            self.spec, ViewBudgetSpec
        ):
            raise MMDSValidationError(
                "view_budget nodes require a ViewBudgetSpec."
            )

    def children(self) -> tuple[DatasetExpr, ...]:
        if self.source is None:
            return ()
        return (self.source,)

    def walk_postorder(self) -> Iterator[DatasetExpr]:
        seen: set[DatasetExpr] = set()

        def visit(node: DatasetExpr) -> Iterator[DatasetExpr]:
            if node in seen:
                return
            seen.add(node)
            if node.source is not None:
                yield from visit(node.source)
            yield node

        yield from visit(self)


@dataclass(frozen=True)
class Assignment:
    target: str
    expr: DatasetExpr


@dataclass(frozen=True)
class QueryProgram:
    assignments: tuple[Assignment, ...]
    output_name: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.assignments:
            raise MMDSValidationError(
                "A query program must contain at least one assignment."
            )
        if self.output_name not in {
            assignment.target for assignment in self.assignments
        }:
            raise MMDSValidationError(
                f"Output variable {self.output_name!r} is not defined in the query program."
            )

    @property
    def output_expr(self) -> DatasetExpr:
        for assignment in reversed(self.assignments):
            if assignment.target == self.output_name:
                return assignment.expr
        raise MMDSValidationError(f"Output variable {self.output_name!r} is missing.")

    def input_paths(self) -> tuple[str, ...]:
        paths = {
            assignment.expr.input_path
            for assignment in self.assignments
            if assignment.expr.kind == "input"
        }
        return tuple(sorted(path for path in paths if path is not None))

    def used_udfs(self) -> tuple[UdfSpec, ...]:
        specs: set[UdfSpec] = set()
        for assignment in self.assignments:
            for node in assignment.expr.walk_postorder():
                if isinstance(node.spec, UdfSpec):
                    specs.add(node.spec)
                elif isinstance(node.spec, VideoMapSpec) and isinstance(
                    node.spec.map_spec, UdfSpec
                ):
                    specs.add(node.spec.map_spec)
        return tuple(sorted(specs, key=lambda spec: (spec.module, spec.name)))


def normalize_group_by(group_by: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(group_by, str):
        normalized = (group_by,)
    elif isinstance(group_by, (list, tuple)):
        normalized = tuple(group_by)
    else:
        raise TypeError("group_by must be a string or a sequence of strings.")
    if not normalized:
        raise MMDSValidationError("group_by cannot be empty.")
    if any(not isinstance(field, str) for field in normalized):
        raise TypeError("group_by fields must all be strings.")
    return normalized


def udf_spec_from_callable(value: Callable[..., Any]) -> UdfSpec:
    module = getattr(value, "__module__", "")
    name = getattr(value, "__name__", "")
    qualname = getattr(value, "__qualname__", name)
    if not module.startswith("udfs"):
        raise MMDSValidationError(
            "UDF callables must be imported from the ./udfs package."
        )
    if not name or name == "<lambda>" or "<locals>" in qualname:
        raise MMDSValidationError(
            "Inline lambdas and nested functions are not supported UDFs."
        )
    return UdfSpec(module=module, name=name)


def prompt_uses_record_helpers(spec: PromptSpec) -> bool:
    return any(_prompt_part_uses_helpers(part) for part in spec.parts)


def normalize_output_schema(value: JsonValue | None) -> RecordSchema | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(
            "schema= must be a dictionary mapping output field names to schema fragments."
        )
    if _looks_like_legacy_object_schema(value):
        return _normalize_legacy_object_schema(value)
    return _normalize_record_schema(value)


def expand_output_schema(schema: RecordSchema | None) -> dict[str, JsonValue] | None:
    if schema is None:
        return None
    properties: dict[str, JsonValue] = {}
    for field_name, field_schema in sorted(schema.items()):
        properties[field_name] = _expand_field_schema(field_schema)
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
    }


def _prompt_part_cache_key(part: PromptPart) -> Any:
    if isinstance(part, str):
        return part
    if isinstance(part, RecordPath):
        return ("record", part.path)
    if isinstance(part, ForEachPrompt):
        return ("foreach", tuple(_prompt_part_cache_key(child) for child in part.parts))
    raise TypeError(f"Unsupported prompt part {part!r}.")


def _prompt_part_uses_helpers(part: PromptPart) -> bool:
    if isinstance(part, str):
        return False
    if isinstance(part, RecordPath):
        return True
    if isinstance(part, ForEachPrompt):
        return True
    raise TypeError(f"Unsupported prompt part {part!r}.")


def _freeze_json_value(value: JsonValue | None) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _freeze_json_value(item)) for key, item in value.items())
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _looks_like_legacy_object_schema(value: dict[str, JsonValue]) -> bool:
    return "properties" in value or "required" in value or value.get("type") == "object"


def _normalize_legacy_object_schema(value: dict[str, JsonValue]) -> RecordSchema:
    allowed_keys = {"type", "properties", "required"}
    unexpected = sorted(str(key) for key in value if key not in allowed_keys)
    if unexpected:
        raise TypeError(
            "Full JSON-schema objects are only supported in the legacy object form with "
            "'type', 'properties', and 'required' keys."
        )
    if value.get("type") != "object":
        raise TypeError("schema= object wrappers must use {'type': 'object', ...}.")

    properties = value.get("properties")
    if not isinstance(properties, dict):
        raise TypeError(
            "schema= object wrappers must include a dictionary 'properties' entry."
        )

    required = value.get("required")
    if not isinstance(required, list) or any(
        not isinstance(item, str) for item in required
    ):
        raise TypeError(
            "schema= object wrappers must include a string-list 'required' entry."
        )
    property_names = sorted(str(key) for key in properties)
    if sorted(required) != property_names:
        raise TypeError(
            "schema= object wrappers must mark every output field as required."
        )

    return _normalize_record_schema(properties)


def _normalize_record_schema(value: dict[str, JsonValue]) -> RecordSchema:
    schema: RecordSchema = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("schema= field names must be non-empty strings.")
        schema[key] = _normalize_field_schema(item)
    return schema


def _normalize_field_schema(value: JsonValue) -> FieldSchemaValue:
    if isinstance(value, str):
        if not value:
            raise TypeError("schema= field type shorthands must be non-empty strings.")
        return value
    if isinstance(value, dict):
        normalized = _normalize_json_object(value)
        if set(normalized) == {"type"} and isinstance(normalized["type"], str):
            return normalized["type"]
        return normalized
    raise TypeError(
        "schema= field schemas must be string type shorthands or JSON-schema fragment dictionaries."
    )


def _normalize_json_object(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("schema= JSON-schema fragment keys must be strings.")
        normalized[key] = _normalize_json_value(item)
    return normalized


def _normalize_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _normalize_json_object(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("schema= values must be JSON-compatible literals.")


def _expand_field_schema(value: FieldSchemaValue) -> JsonValue:
    if isinstance(value, str):
        return {"type": value}
    return value
