from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, TypeAlias

from .model import (
    BuiltinSpec,
    DatasetExpr,
    DetectSpec,
    ForEachPrompt,
    JsonValue,
    MMDSValidationError,
    PromptPart,
    PromptSpec,
    PadIntervalSpec,
    Record,
    RecordPath,
    ReconcileIntervalsSpec,
    ResolveSpec,
    SemanticSpec,
    ViewSpec,
    normalize_output_schema,
    normalize_group_by,
    udf_spec_from_callable,
)

PromptInputPart: TypeAlias = str | RecordPath | ForEachPrompt
PromptInput: TypeAlias = str | Sequence[PromptInputPart]


def Input(name: str) -> DatasetExpr:
    if not isinstance(name, str) or not name:
        raise TypeError("Input paths must be non-empty strings.")
    _validate_input_path(name)
    return DatasetExpr(kind="input", input_path=name)


def Map(
    data: DatasetExpr,
    spec: str | Sequence[PromptInputPart] | Callable[..., Any] | BuiltinSpec,
    *,
    schema: JsonValue | None = None,
    name: str | None = None,
) -> DatasetExpr:
    return DatasetExpr(
        kind="map",
        source=_normalize_source(data),
        spec=_normalize_spec(spec, op_kind="map", schema=schema),
        name=name,
    )


def Filter(
    data: DatasetExpr,
    spec: str | Sequence[PromptInputPart] | Callable[..., Any],
    *,
    name: str | None = None,
) -> DatasetExpr:
    return DatasetExpr(
        kind="filter",
        source=_normalize_source(data),
        spec=_normalize_spec(spec, op_kind="filter", schema=None),
        name=name,
    )


def Reduce(
    data: DatasetExpr,
    group_by: str | list[str] | tuple[str, ...],
    reducer: str | Sequence[PromptInputPart] | Callable[..., Any] | BuiltinSpec,
    *,
    schema: JsonValue | None = None,
    name: str | None = None,
) -> DatasetExpr:
    return DatasetExpr(
        kind="reduce",
        source=_normalize_source(data),
        spec=_normalize_spec(reducer, op_kind="reduce", schema=schema),
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


def Resolve(
    data: DatasetExpr,
    group_by: str | list[str] | tuple[str, ...],
    start_field: str,
    end_field: str,
    *,
    strategy: str = "overlap",
    merge_touching: bool = True,
    name: str | None = None,
) -> DatasetExpr:
    """Coalesce overlapping source-time intervals independently per group."""
    return DatasetExpr(
        kind="resolve",
        source=_normalize_source(data),
        spec=ResolveSpec(
            group_by=normalize_group_by(group_by),
            start_field=start_field,
            end_field=end_field,
            strategy=strategy,
            merge_touching=merge_touching,
        ),
        name=name,
    )


def View(
    data: DatasetExpr,
    video_field: str,
    start_field: str,
    end_field: str,
    *,
    output_field: str = "view",
    name: str | None = None,
) -> DatasetExpr:
    """Materialize a bounded standalone video value for every tuple."""
    return DatasetExpr(
        kind="view",
        source=_normalize_source(data),
        spec=ViewSpec(
            video_field=video_field,
            start_field=start_field,
            end_field=end_field,
            output_field=output_field,
        ),
        name=name,
    )


def PadInterval(
    interval_field: str,
    duration_field: str,
    padding_seconds: float,
    *,
    input_start_field: str = "start_seconds",
    input_end_field: str = "end_seconds",
    output_start_field: str = "window_start_seconds",
    output_end_field: str = "window_end_seconds",
) -> PadIntervalSpec:
    """Create a deterministic Map function that pads one nested interval."""
    return PadIntervalSpec(
        interval_field=interval_field,
        duration_field=duration_field,
        padding_seconds=padding_seconds,
        input_start_field=input_start_field,
        input_end_field=input_end_field,
        output_start_field=output_start_field,
        output_end_field=output_end_field,
    )


def ReconcileIntervals(
    events_field: str,
    window_start_field: str,
    window_end_field: str,
    *,
    output_field: str = "events",
    event_start_field: str = "start_seconds",
    event_end_field: str = "end_seconds",
    preserve_fields: list[str] | tuple[str, ...] = (),
    deduplication_tiou_threshold: float = 0.8,
) -> ReconcileIntervalsSpec:
    """Create a deterministic Reduce function for clip-level event records."""
    if not isinstance(preserve_fields, (list, tuple)):
        raise TypeError("ReconcileIntervals preserve_fields must be a list or tuple.")
    return ReconcileIntervalsSpec(
        events_field=events_field,
        window_start_field=window_start_field,
        window_end_field=window_end_field,
        output_field=output_field,
        event_start_field=event_start_field,
        event_end_field=event_end_field,
        preserve_fields=tuple(preserve_fields),
        deduplication_tiou_threshold=deduplication_tiou_threshold,
    )


def Detect(
    data: DatasetExpr,
    video_field: str,
    classes: list[str] | tuple[str, ...],
    *,
    model: str = "yoloe-11s-seg.pt",
    output_field: str = "detections",
    name: str | None = None,
) -> DatasetExpr:
    """Run YOLOE object detection on every frame of a video field.

    Args:
        data: Source dataset expression.
        video_field: Name of the row field that holds the video.  The value
            may be a string path/URL or a dict with a ``"source"``,
            ``"path"``, or ``"uri"`` key.
        classes: Non-empty list of class names to detect (open-vocabulary).
        model: YOLOE weights file.  Defaults to ``"yoloe-11s-seg.pt"``.
        output_field: Name of the output field that receives the detection
            list.  Defaults to ``"detections"``.
        name: Optional operator label.

    The output field contains a list of objects, one per detected class::

        [{"type": "dog", "bboxes": [{"frame_idx": 0, "bbox": [x1,y1,x2,y2], "confidence": 0.9}, ...]}, ...]
    """
    if not isinstance(video_field, str) or not video_field:
        raise TypeError("Detect video_field must be a non-empty string.")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise TypeError("Detect classes must be a non-empty list of strings.")
    if any(not isinstance(c, str) or not c for c in classes):
        raise TypeError("Detect classes must all be non-empty strings.")
    if not isinstance(model, str) or not model:
        raise TypeError("Detect model must be a non-empty string.")
    if not isinstance(output_field, str) or not output_field:
        raise TypeError("Detect output_field must be a non-empty string.")
    return DatasetExpr(
        kind="detect",
        source=_normalize_source(data),
        spec=DetectSpec(
            video_field=video_field,
            classes=tuple(classes),
            model=model,
            output_field=output_field,
        ),
        name=name,
    )


def ForEach(parts: Sequence[PromptInputPart]) -> ForEachPrompt:
    normalized_parts = _normalize_prompt_parts(
        parts, allow_foreach=False, allow_record=True
    )
    return ForEachPrompt(parts=normalized_parts)


def _normalize_source(data: DatasetExpr) -> DatasetExpr:
    if not isinstance(data, DatasetExpr):
        raise TypeError(
            "Operator sources must be DatasetExpr instances created by the MMDS DSL."
        )
    return data


def _validate_input_path(path: str) -> None:
    if not (path.endswith(".json") or path.endswith(".jsonl")):
        raise TypeError("Input paths must point to .json or .jsonl files.")


def _normalize_spec(
    spec: str | Sequence[PromptInputPart] | Callable[..., Any] | BuiltinSpec,
    *,
    op_kind: str,
    schema: JsonValue | None,
) -> SemanticSpec:
    if isinstance(spec, BuiltinSpec):
        if schema is not None:
            raise TypeError("schema= is not valid for built-in deterministic functions.")
        if op_kind == "filter":
            raise TypeError("Filter does not currently accept built-in deterministic functions.")
        return spec
    if callable(spec):
        if schema is not None:
            raise TypeError("schema= is only valid for prompt-backed operators.")
        return udf_spec_from_callable(spec)

    allow_foreach = op_kind == "reduce"
    allow_record = op_kind != "reduce"
    parts = _normalize_prompt_spec(
        spec, allow_foreach=allow_foreach, allow_record=allow_record
    )
    if op_kind in {"map", "reduce"} and schema is None:
        raise TypeError(f"Prompt-backed {op_kind} operations require schema=...")
    return PromptSpec(parts=parts, output_schema=normalize_output_schema(schema))


def _normalize_prompt_spec(
    value: str | Sequence[PromptInputPart],
    *,
    allow_foreach: bool,
    allow_record: bool,
) -> tuple[PromptPart, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = _normalize_prompt_parts(
            value, allow_foreach=allow_foreach, allow_record=allow_record
        )
        if not parts:
            raise TypeError("Prompt part lists cannot be empty.")
        return parts
    raise TypeError(
        "Prompt specs must be strings, prompt-part lists, or imported UDF callables."
    )


def _normalize_prompt_parts(
    values: Sequence[PromptInputPart],
    *,
    allow_foreach: bool,
    allow_record: bool,
) -> tuple[PromptPart, ...]:
    parts: list[PromptPart] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
            continue
        if isinstance(value, RecordPath):
            if value.is_root():
                raise MMDSValidationError(
                    "Record cannot be used without selecting at least one field."
                )
            if not allow_record:
                raise MMDSValidationError(
                    "Record[...] is only valid for single-record prompts or inside Reduce ForEach(...)."
                )
            parts.append(value)
            continue
        if isinstance(value, ForEachPrompt):
            if not allow_foreach:
                raise MMDSValidationError(
                    "ForEach(...) is only valid at the top level of Reduce prompts."
                )
            parts.append(value)
            continue
        raise TypeError(f"Unsupported prompt part {value!r}.")
    return tuple(parts)
