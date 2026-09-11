from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from ...media import (
    local_media_path,
    materialized_video_value,
    media_fingerprint,
    resolve_media_source,
)
from ...model import DatasetExpr, MMDSExecutionError, MMDSValidationError, Row, ViewSpec
from ..context import ExecutionContext, ViewExecutionStat
from ..media import FFmpegMaterializer, MATERIALIZATION_CONTRACT


def _apply_view(
    node: DatasetExpr,
    row: Row,
    context: ExecutionContext,
    *,
    base_path: Path | None,
) -> Row:
    spec = node.spec
    if not isinstance(spec, ViewSpec):
        raise MMDSValidationError("View nodes require a ViewSpec.")
    if spec.video_field not in row:
        raise MMDSValidationError(
            f"View video field {spec.video_field!r} is missing from the row."
        )
    start = _boundary(row.get(spec.start_field), spec.start_field)
    end = _boundary(row.get(spec.end_field), spec.end_field)
    if start >= end:
        raise MMDSValidationError("View requires source boundaries satisfying start < end.")
    video = row[spec.video_field]
    source = resolve_media_source(video)
    result = dict(row)

    if context.workspace is None:
        raise MMDSExecutionError(
            "Materialized View requires ExecutionContext(workspace=...)."
        )
    source_path = local_media_path(source, base_path=base_path)
    key = _view_key(video, source_path, start, end)
    destination = context.workspace / "views" / f"{key}.mp4"
    with context.view_lock(key):
        clip = context.cached_view(key)
        reused = clip is not None
        if clip is not None and (
            not clip.path.is_file() or clip.path.stat().st_size != clip.size_bytes
        ):
            raise MMDSExecutionError(
                f"Materialized View changed during query execution: {clip.path}"
            )
        if clip is None:
            materializer = context.materializer or FFmpegMaterializer()
            clip = materializer.materialize(source_path, destination, start, end)
            context.cache_view(key, clip)
    result[spec.output_field] = materialized_video_value(
        video,
        clip.path,
        start=start,
        end=end,
        sha256=clip.sha256,
    )
    context.stats.record_view(
        ViewExecutionStat(
            name=node.name,
            source=str(source_path),
            start_seconds=start,
            end_seconds=end,
            output_path=str(clip.path),
            output_bytes=clip.size_bytes,
            reused=reused,
        )
    )
    return result


def _view_key(
    video: object, source_path: Path, start: float, end: float
) -> str:
    payload = {
        "source": media_fingerprint(video, source_path),
        "start_seconds": start,
        "end_seconds": end,
        "contract": MATERIALIZATION_CONTRACT,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _boundary(value: object, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise MMDSValidationError(
            f"View field {field!r} must be a finite non-negative number."
        )
    return float(value)


__all__ = ["_apply_view"]
