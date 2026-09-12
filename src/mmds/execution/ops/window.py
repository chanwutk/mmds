from __future__ import annotations

from ...model import DatasetExpr, MMDSValidationError, Row, WindowSpec


def _apply_window(node: DatasetExpr, row: Row) -> Row:
    spec = node.spec
    if not isinstance(spec, WindowSpec):
        raise MMDSValidationError("Window requires a WindowSpec.")

    candidate = row[spec.candidate_field]
    start = candidate["start"]
    end = candidate["end"]

    if end <= start:
        raise MMDSValidationError("Window end must be greater than start.")

    clip = dict(row[spec.video_field])
    clip["type"] = "VideoView"
    clip["start"] = max(0, start - spec.padding_time)
    clip["end"] = end + spec.padding_time

    result = dict(row)
    result[spec.output_field] = clip
    return result