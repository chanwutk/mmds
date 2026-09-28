from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....model import (
    DatasetExpr,
    DetectSpec,
    PromptSpec,
    RecordPath,
    UdfSpec,
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


_DEFAULT_MODEL = "yoloe-11s-seg.pt"
_OUTPUT_FIELD = "detections"
_KEEP_UDF = UdfSpec(
    module="udfs.detection_ops",
    name="keep_rows_with_detections",
)


class DetectGateBeforeMapParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(
        min_length=1,
        description="Input field containing the video to gate with YOLOE.",
    )
    classes: list[str] = Field(
        min_length=1,
        description=(
            "Open-vocabulary YOLOE class names used as a high-recall presence "
            "gate before the semantic Map."
        ),
    )
    model: str = Field(
        default=_DEFAULT_MODEL,
        min_length=1,
        description="YOLOE weights file for the Detect gate.",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> DetectGateBeforeMapParams:
        if not self.video_field.strip():
            raise ValueError("video_field cannot be blank.")
        if not self.model.strip():
            raise ValueError("model cannot be blank.")
        if any(not isinstance(name, str) or not name.strip() for name in self.classes):
            raise ValueError("classes must contain non-empty strings.")
        normalized = tuple(name.strip() for name in self.classes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("classes must be unique.")
        object.__setattr__(self, "video_field", self.video_field.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "classes", list(normalized))
        return self


class DetectGateBeforeMap:
    """Insert YOLOE Detect + keep Filter before a video-reading Map."""

    metadata = DirectiveMetadata(
        name="detect_gate_before_map",
        description=(
            "Run YOLOE Detect and keep only clips with detections before an "
            "expensive prompt-backed Map over video."
        ),
        when_to_use=(
            "Use when the Map looks for object presence or appearance that "
            "open-vocabulary detection can high-recall prune before a VLM call."
        ),
    )
    params_type = DetectGateBeforeMapParams

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
        if not isinstance(params, DetectGateBeforeMapParams):
            raise MMDSRewriteError(
                "DetectGateBeforeMap received invalid parameters."
            )

        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "DetectGateBeforeMap requires a prompt-backed Map."
            )
        if node.source is None:
            raise MMDSRewriteError(
                "DetectGateBeforeMap requires a Map source."
            )

        video_reference = RecordPath((params.video_field,))
        paths = record_paths(node.spec.parts)
        if video_reference not in paths:
            raise MMDSRewriteError(
                f"Matched Map does not directly reference video field "
                f"{params.video_field!r}."
            )
        nested = tuple(path for path in paths if len(path.path) != 1)
        if nested:
            raise MMDSRewriteError(
                "DetectGateBeforeMap currently supports only top-level prompt "
                f"fields; found {format_record_paths(nested)}."
            )

        if _source_has_detect_gate(node.source, params.video_field):
            raise MMDSRewriteError(
                "Matched Map already sits behind a Detect gate on "
                f"{params.video_field!r}."
            )

        detected = DatasetExpr(
            kind="detect",
            source=node.source,
            spec=DetectSpec(
                video_field=params.video_field,
                classes=tuple(params.classes),
                model=params.model,
                output_field=_OUTPUT_FIELD,
            ),
            name="rewrite_detect_gate",
        )
        gated = DatasetExpr(
            kind="filter",
            source=detected,
            spec=_KEEP_UDF,
            name="rewrite_keep_detections",
        )
        replacement = replace(node, source=gated)
        return index.replace(match.path, replacement)


def _source_has_detect_gate(source: DatasetExpr, video_field: str) -> bool:
    """Return True when source is Filter(Detect(...)) for the same video field."""

    if source.kind != "filter" or source.source is None:
        return False
    detect = source.source
    if detect.kind != "detect" or not isinstance(detect.spec, DetectSpec):
        return False
    return detect.spec.video_field == video_field
