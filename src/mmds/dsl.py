from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, TypeAlias

from .model import (
    DatasetExpr,
    DetectSpec,
    ForEachPrompt,
    JsonValue,
    JoinSpec,
    MMDSValidationError,
    PromptPart,
    PromptSpec,
    Record,
    RecordPath,
    SemanticSpec,
    SplitSpec,
    normalize_join_keys,
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
    spec: str | Sequence[PromptInputPart] | Callable[..., Any],
    *,
    schema: JsonValue | None = None,
    replace: bool = False,
    name: str | None = None,
) -> DatasetExpr:
    return DatasetExpr(
        kind="map",
        source=_normalize_source(data),
        spec=_normalize_spec(spec, op_kind="map", schema=schema),
        replace=bool(replace),
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
    reducer: str | Sequence[PromptInputPart] | Callable[..., Any],
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


def Split(
    data: DatasetExpr,
    video_field: str,
    *,
    chunk_sec: float = 30.0,
    doc_id_key: str = "camera_id",
    output_prefix: str = "split_video",
    name: str | None = None,
) -> DatasetExpr:
    """Fan out each row into fixed-duration video chunks.

    Reads ``video_field`` as a ``VideoView`` (``start``/``end`` seconds) or a
    string path plus row ``duration_sec``. Each output row keeps parent fields
    and adds:

    - ``{output_prefix}_id`` — value of ``doc_id_key``
    - ``{output_prefix}_chunk_num`` — 0-based chunk index
    - ``{output_prefix}_chunk_start`` / ``{output_prefix}_chunk_end`` — absolute seconds
    - ``video_field`` — narrowed ``VideoView`` for that chunk
    """
    if not isinstance(video_field, str) or not video_field:
        raise TypeError("Split video_field must be a non-empty string.")
    if not isinstance(chunk_sec, (int, float)) or chunk_sec <= 0:
        raise TypeError("Split chunk_sec must be a positive number.")
    if not isinstance(doc_id_key, str) or not doc_id_key:
        raise TypeError("Split doc_id_key must be a non-empty string.")
    if not isinstance(output_prefix, str) or not output_prefix:
        raise TypeError("Split output_prefix must be a non-empty string.")
    return DatasetExpr(
        kind="split",
        source=_normalize_source(data),
        spec=SplitSpec(
            video_field=video_field,
            chunk_sec=float(chunk_sec),
            doc_id_key=doc_id_key,
            output_prefix=output_prefix,
        ),
        name=name,
    )


def Join(
    left: DatasetExpr,
    right: DatasetExpr,
    predicate: Callable[..., Any] | None = None,
    *,
    on: str | Sequence[str] | None = None,
    one_to_one: bool = False,
    score: Callable[..., Any] | None = None,
    min_score: float | None = None,
    left_key: str | Sequence[str] | None = None,
    right_key: str | Sequence[str] | None = None,
    name: str | None = None,
) -> DatasetExpr:
    """Pair-wise join over two row collections.

    When ``on`` names one or more fields, the executor builds a hash index on
    the right rows and probes by key — ``O(|left| + |right|)`` instead of a
    nested loop. An optional ``predicate`` UDF further filters candidate pairs.

    When ``one_to_one=True``, the executor keeps at most one match per left and
    right track identity (``left_key`` / ``right_key``). Candidate pairs must
    pass ``predicate`` and have ``score(left, right) >= min_score`` when
    ``min_score`` is set. Pairs are taken in descending score order; unmatched
    tracks are omitted.

    At least one of ``on`` or ``predicate`` is required. Output rows are
    ``{"left": <left row>, "right": <right row>}``, plus ``match_score`` when
    ``one_to_one=True``.
    """
    keys = normalize_join_keys(on) if on is not None else ()
    predicate_spec = None
    if predicate is not None:
        if not callable(predicate):
            raise TypeError("Join predicates must be imported UDF callables.")
        predicate_spec = udf_spec_from_callable(predicate)
    score_spec = None
    if score is not None:
        if not callable(score):
            raise TypeError("Join score functions must be imported UDF callables.")
        score_spec = udf_spec_from_callable(score)
    left_key_fields = normalize_join_keys(left_key) if left_key is not None else ()
    right_key_fields = normalize_join_keys(right_key) if right_key is not None else ()
    join_spec = JoinSpec(
        keys=keys,
        predicate=predicate_spec,
        one_to_one=bool(one_to_one),
        score=score_spec,
        min_score=min_score,
        left_key=left_key_fields,
        right_key=right_key_fields,
    )
    return DatasetExpr(
        kind="join",
        source=_normalize_source(left),
        right_source=_normalize_source(right),
        spec=join_spec,
        name=name,
    )


def Detect(
    data: DatasetExpr,
    video_field: str,
    classes: list[str] | tuple[str, ...],
    *,
    model: str = "yoloe-11s-seg.pt",
    output_field: str = "detections",
    frame_stride: int = 1,
    conf: float | None = None,
    imgsz: int | None = None,
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
        frame_stride: Run inference on every ``frame_stride``-th frame
            (``1`` = every frame, the default). Frames that are skipped are
            not sent through the model, trading temporal density for lower
            inference cost; use this when detections are expected to be
            consistent across a few consecutive frames (e.g. slow-moving or
            already-tracked objects).
        conf: Minimum detection confidence passed to the model. ``None``
            (the default) uses the model's own default threshold. Lower it
            (e.g. ``0.1``) to recover small or low-contrast objects that the
            default threshold discards.
        imgsz: Inference image size passed to the model. ``None`` (the
            default) uses the model's own default (typically ``640``). Raise
            it (e.g. ``1280``) so that small objects—vehicles in high-mounted
            traffic footage, for example—survive downscaling.
        name: Optional operator label.

    The output field contains a list of objects, one per detected class::

        [{"type": "dog", "bboxes": [{"frame_idx": 0, "bbox": [x1,y1,x2,y2], "confidence": 0.9}, ...]}, ...]

    Only frames actually sampled under ``frame_stride`` contribute ``bboxes``;
    their ``frame_idx`` values remain absolute indices into the source video.
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
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
        raise TypeError("Detect frame_stride must be an integer >= 1.")
    if conf is not None and (
        isinstance(conf, bool)
        or not isinstance(conf, (int, float))
        or not (0.0 <= float(conf) <= 1.0)
    ):
        raise TypeError("Detect conf must be a number in [0.0, 1.0] or None.")
    if imgsz is not None and (
        isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz < 1
    ):
        raise TypeError("Detect imgsz must be a positive integer or None.")
    return DatasetExpr(
        kind="detect",
        source=_normalize_source(data),
        spec=DetectSpec(
            video_field=video_field,
            classes=tuple(classes),
            model=model,
            output_field=output_field,
            frame_stride=frame_stride,
            conf=conf,
            imgsz=imgsz,
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
    spec: str | Sequence[PromptInputPart] | Callable[..., Any],
    *,
    op_kind: str,
    schema: JsonValue | None,
) -> SemanticSpec:
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
