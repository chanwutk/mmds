from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Iterator, Literal, TypeAlias

Row: TypeAlias = dict[str, Any]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
FieldSchemaValue: TypeAlias = str | dict[str, JsonValue]
RecordSchema: TypeAlias = dict[str, FieldSchemaValue]
OperatorKind: TypeAlias = Literal[
    "input", "map", "filter", "reduce", "unnest", "split", "detect", "join"
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
class JoinSpec:
    """Join operator spec: optional equi-join keys and/or a binary UDF predicate."""

    keys: tuple[str, ...] = ()
    predicate: UdfSpec | None = None
    one_to_one: bool = False
    score: UdfSpec | None = None
    min_score: float | None = None
    left_key: tuple[str, ...] = ()
    right_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.keys and self.predicate is None:
            raise MMDSValidationError(
                "Join requires on= keys, a UDF predicate, or both."
            )
        if any(not isinstance(key, str) or not key for key in self.keys):
            raise MMDSValidationError(
                "Join on= field names must all be non-empty strings."
            )
        if self.one_to_one:
            if self.score is None:
                raise MMDSValidationError(
                    "Join one_to_one=True requires a score= UDF callable as score(left, right) -> number."
                )
            if not self.left_key or not self.right_key:
                raise MMDSValidationError(
                    "Join one_to_one=True requires non-empty left_key= and right_key= field tuples."
                )
            if any(
                not isinstance(key, str) or not key
                for key in self.left_key + self.right_key
            ):
                raise MMDSValidationError(
                    "Join left_key= and right_key= field names must all be non-empty strings."
                )
        elif (
            self.score is not None
            or self.min_score is not None
            or self.left_key
            or self.right_key
        ):
            raise MMDSValidationError(
                "Join score=, min_score=, left_key=, and right_key= require one_to_one=True."
            )
        if self.min_score is not None and not isinstance(self.min_score, (int, float)):
            raise MMDSValidationError("Join min_score= must be numeric when provided.")

    def load_predicate(self) -> Callable[..., Any] | None:
        if self.predicate is None:
            return None
        return self.predicate.load()

    def load_score(self) -> Callable[..., Any] | None:
        if self.score is None:
            return None
        return self.score.load()


@dataclass(frozen=True)
class SplitSpec:
    """Spec for the Split operator: fan out rows into fixed-duration video chunks."""

    video_field: str
    chunk_sec: float = 30.0
    doc_id_key: str = "camera_id"
    output_prefix: str = "split_video"

    def __post_init__(self) -> None:
        if not self.video_field:
            raise MMDSValidationError(
                "SplitSpec video_field must be a non-empty string."
            )
        if not isinstance(self.chunk_sec, (int, float)) or self.chunk_sec <= 0:
            raise MMDSValidationError("SplitSpec chunk_sec must be a positive number.")
        if not self.doc_id_key:
            raise MMDSValidationError(
                "SplitSpec doc_id_key must be a non-empty string."
            )
        if not self.output_prefix:
            raise MMDSValidationError(
                "SplitSpec output_prefix must be a non-empty string."
            )


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


SemanticSpec: TypeAlias = PromptSpec | UdfSpec | SplitSpec | DetectSpec | JoinSpec


@dataclass(frozen=True)
class DatasetExpr:
    kind: OperatorKind
    source: DatasetExpr | None = None
    right_source: DatasetExpr | None = None
    input_path: str | None = None
    spec: SemanticSpec | None = None
    group_by: tuple[str, ...] = ()
    field: str | None = None
    keep_empty: bool = False
    replace: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "input":
            if self.source is not None or self.input_path is None:
                raise MMDSValidationError(
                    "Input nodes require only an input file path."
                )
            return
        if self.kind == "join":
            if self.source is None or self.right_source is None:
                raise MMDSValidationError(
                    "Join nodes require left and right source expressions."
                )
            if not isinstance(self.spec, JoinSpec):
                raise MMDSValidationError(
                    "Join nodes require a JoinSpec with on= keys and/or a UDF predicate."
                )
            return
        if self.source is None:
            raise MMDSValidationError(f"{self.kind} nodes require a source expression.")
        if self.right_source is not None:
            raise MMDSValidationError(
                f"{self.kind} nodes do not accept a right_source expression."
            )
        if self.kind in {"map", "filter", "reduce"} and self.spec is None:
            raise MMDSValidationError(f"{self.kind} nodes require a semantic spec.")
        if self.kind == "reduce" and not self.group_by:
            raise MMDSValidationError("Reduce nodes require one or more grouping keys.")
        if self.kind == "unnest" and self.field is None:
            raise MMDSValidationError("Unnest nodes require a field to expand.")
        if self.kind == "split" and not isinstance(self.spec, SplitSpec):
            raise MMDSValidationError("split nodes require a SplitSpec.")
        if self.replace and self.kind != "map":
            raise MMDSValidationError("replace=True is only supported on Map nodes.")
        if self.kind == "detect" and not isinstance(self.spec, DetectSpec):
            raise MMDSValidationError("detect nodes require a DetectSpec.")

    def children(self) -> tuple[DatasetExpr, ...]:
        if self.kind == "join":
            if self.source is None or self.right_source is None:
                return ()
            return (self.source, self.right_source)
        if self.source is None:
            return ()
        return (self.source,)

    def walk_postorder(self) -> Iterator[DatasetExpr]:
        seen: set[DatasetExpr] = set()

        def visit(node: DatasetExpr) -> Iterator[DatasetExpr]:
            if node in seen:
                return
            seen.add(node)
            for child in node.children():
                yield from visit(child)
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
                elif isinstance(node.spec, JoinSpec):
                    if node.spec.predicate is not None:
                        specs.add(node.spec.predicate)
                    if node.spec.score is not None:
                        specs.add(node.spec.score)
        return tuple(sorted(specs, key=lambda spec: (spec.module, spec.name)))


def normalize_join_keys(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        if not value:
            raise TypeError("Join on= field names must be non-empty strings.")
        return (value,)
    if isinstance(value, (list, tuple)):
        if not value:
            raise MMDSValidationError("Join on= cannot be empty.")
        if any(not isinstance(field, str) or not field for field in value):
            raise TypeError("Join on= field names must all be non-empty strings.")
        return tuple(value)
    raise TypeError("Join on= must be a field name or sequence of field names.")


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
