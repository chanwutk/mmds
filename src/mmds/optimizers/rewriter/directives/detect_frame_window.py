from __future__ import annotations

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....model import (
    DatasetExpr,
    DetectSpec,
    PromptSpec,
    RecordPath,
    UdfSpec,
    VideoMapSpec,
)
from ..core import (
    DirectiveMetadata,
    PlanIndex,
    RewriteMatch,
)
from ..errors import MMDSRewriteError
from ._prompt import (
    format_record_paths,
    prompt_map_entries,
    record_paths,
)
from .detect_gate import _source_has_detect_gate


_DEFAULT_MODEL = "yoloe-11s-seg.pt"
_OUTPUT_FIELD = "detections"
_VIEWS_FIELD = "_mmds_candidate_views"
_CLIP_FIELD = "clip"
_INTERVAL_UDF = UdfSpec(
    module="udfs.detection_ops",
    name="detections_to_candidate_views",
)


class DetectedFrameWindowBeforeMapParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(
        min_length=1,
        description="Input field containing the video to detect and window.",
    )
    classes: list[str] = Field(
        min_length=1,
        description=(
            "Open-vocabulary YOLOE class names used as a high-recall proxy for "
            "frames that should reach the semantic Map."
        ),
    )
    padding_seconds: float = Field(
        default=0.0,
        description=(
            "Symmetric Window padding in seconds. Bridging gaps between "
            "one-frame detection hits so Coalesce can merge nearby views."
        ),
    )
    min_confidence: float | None = Field(
        default=None,
        description=(
            "Optional YOLOE confidence threshold in [0, 1] applied during Detect."
        ),
    )
    model: str = Field(
        default=_DEFAULT_MODEL,
        min_length=1,
        description="YOLOE weights file for the Detect stage.",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> DetectedFrameWindowBeforeMapParams:
        if not self.video_field.strip():
            raise ValueError("video_field cannot be blank.")
        if not self.model.strip():
            raise ValueError("model cannot be blank.")
        if any(not isinstance(name, str) or not name.strip() for name in self.classes):
            raise ValueError("classes must contain non-empty strings.")
        normalized = tuple(name.strip() for name in self.classes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("classes must be unique.")
        if (
            not isinstance(self.padding_seconds, (int, float))
            or isinstance(self.padding_seconds, bool)
            or not isfinite(self.padding_seconds)
            or self.padding_seconds < 0
        ):
            raise ValueError("padding_seconds must be a finite non-negative number.")
        if self.min_confidence is not None:
            if (
                not isinstance(self.min_confidence, (int, float))
                or isinstance(self.min_confidence, bool)
                or not isfinite(self.min_confidence)
                or not (0.0 <= float(self.min_confidence) <= 1.0)
            ):
                raise ValueError("min_confidence must be a finite number in [0, 1].")
        object.__setattr__(self, "video_field", self.video_field.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "classes", list(normalized))
        object.__setattr__(self, "padding_seconds", float(self.padding_seconds))
        if self.min_confidence is not None:
            object.__setattr__(self, "min_confidence", float(self.min_confidence))
        return self


class DetectedFrameWindowBeforeMap:
    """Detect objects, window to hit frames, then jointly Map over those views."""

    metadata = DirectiveMetadata(
        name="detected_frame_window_before_map",
        description=(
            "Run YOLOE Detect, convert hit frames into candidate video windows, "
            "and answer once with VideoMap over coalesced detected-frame views."
        ),
        when_to_use=(
            "Use for object-centric questions on long video where the object "
            "appears in few frames and empty spans dominate VLM cost. Prefer "
            "detect_gate_before_map when whole empty clips should be dropped but "
            "survivors can still watch the full clip."
        ),
    )
    params_type = DetectedFrameWindowBeforeMapParams

    def __init__(
        self,
        *,
        identity_fields: str | list[str] | tuple[str, ...],
    ) -> None:
        if isinstance(identity_fields, str):
            normalized = (identity_fields,)
        elif isinstance(identity_fields, (list, tuple)):
            normalized = tuple(identity_fields)
        else:
            raise TypeError(
                "DetectedFrameWindowBeforeMap identity_fields must be a string "
                "or sequence of strings."
            )
        if not normalized:
            raise ValueError(
                "DetectedFrameWindowBeforeMap identity_fields must contain at "
                "least one stable source key."
            )
        if any(
            not isinstance(field, str) or not field.strip() for field in normalized
        ):
            raise TypeError(
                "DetectedFrameWindowBeforeMap identity_fields must contain "
                "non-empty strings."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "DetectedFrameWindowBeforeMap identity_fields must be unique."
            )
        reserved = {_VIEWS_FIELD, _CLIP_FIELD, _OUTPUT_FIELD, "_mmds_video_fps"}
        if reserved.intersection(normalized):
            raise ValueError(
                "DetectedFrameWindowBeforeMap identity_fields cannot use reserved "
                "rewrite fields."
            )
        self.identity_fields = normalized

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        return tuple(
            RewriteMatch(
                path=entry.path,
                summary=(
                    "Prompt-backed Map reading "
                    + format_record_paths(record_paths(entry.node.spec.parts))
                ),
            )
            for entry in prompt_map_entries(index)
            if isinstance(entry.node.spec, PromptSpec)
        )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, DetectedFrameWindowBeforeMapParams):
            raise MMDSRewriteError(
                "DetectedFrameWindowBeforeMap received invalid parameters."
            )

        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "DetectedFrameWindowBeforeMap requires a prompt-backed Map."
            )
        if node.source is None:
            raise MMDSRewriteError(
                "DetectedFrameWindowBeforeMap requires a Map source."
            )
        if params.video_field in self.identity_fields:
            raise MMDSRewriteError(
                "DetectedFrameWindowBeforeMap identity_fields cannot include the "
                "video field because video values are not stable grouping keys."
            )

        paths = record_paths(node.spec.parts)
        video_reference = RecordPath((params.video_field,))
        if video_reference not in paths:
            raise MMDSRewriteError(
                f"Matched Map does not directly reference video field "
                f"{params.video_field!r}."
            )
        nested = tuple(path for path in paths if len(path.path) != 1)
        if nested:
            raise MMDSRewriteError(
                "DetectedFrameWindowBeforeMap currently supports only top-level "
                f"prompt fields; found {format_record_paths(nested)}."
            )
        if _source_has_detect_gate(node.source, params.video_field):
            raise MMDSRewriteError(
                "Matched Map already sits behind a Detect gate on "
                f"{params.video_field!r}."
            )

        group_by = self._derive_group_by(index, match, paths, params.video_field)
        detected = DatasetExpr(
            kind="detect",
            source=node.source,
            spec=DetectSpec(
                video_field=params.video_field,
                classes=tuple(params.classes),
                model=params.model,
                output_field=_OUTPUT_FIELD,
                conf=params.min_confidence,
            ),
            name="rewrite_detect_frames",
        )
        candidates = DatasetExpr(
            kind="map",
            source=detected,
            spec=_INTERVAL_UDF,
            name="rewrite_detection_windows",
        )
        replacement = DatasetExpr(
            kind="video_map",
            source=candidates,
            spec=VideoMapSpec(
                video_field=params.video_field,
                views_field=_VIEWS_FIELD,
                group_by=group_by,
                map_spec=node.spec,
                padding_time=params.padding_seconds,
                clip_field=_CLIP_FIELD,
            ),
            name=node.name,
        )
        return index.replace(match.path, replacement)

    def _derive_group_by(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        prompt_paths: tuple[RecordPath, ...],
        video_field: str,
    ) -> tuple[str, ...]:
        fields: list[str] = list(self.identity_fields)
        match_steps = match.path.steps

        for entry in index.entries:
            entry_steps = entry.path.steps
            if (
                len(entry_steps) < len(match_steps)
                and match_steps[: len(entry_steps)] == entry_steps
            ):
                fields.extend(entry.node.group_by)

        fields.extend(
            path.path[0]
            for path in prompt_paths
            if path.path and path.path[0] != video_field
        )
        group_by = tuple(dict.fromkeys(fields))
        reserved = {_VIEWS_FIELD, _CLIP_FIELD, _OUTPUT_FIELD, "_mmds_video_fps"}
        if reserved.intersection(group_by):
            raise MMDSRewriteError(
                "Derived grouping fields collide with reserved rewrite fields."
            )
        return group_by
