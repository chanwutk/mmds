from __future__ import annotations

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
_PRESENCE_UDF = "map_detection_presence"


class DetectPresenceMapParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(
        min_length=1,
        description="Input field containing the video whose objects answer the Map.",
    )
    classes: list[str] = Field(
        min_length=1,
        description=(
            "Open-vocabulary YOLOE class names. Finding one track of any of "
            "these classes answers the boolean as true."
        ),
    )
    flag_field: str = Field(
        min_length=1,
        description=(
            "Boolean Map output field replaced by detection presence. It must "
            "be the Map's only schema field."
        ),
    )
    model: str = Field(
        default=_DEFAULT_MODEL,
        min_length=1,
        description="YOLOE weights file for the Detect stage.",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> DetectPresenceMapParams:
        if not self.video_field.strip():
            raise ValueError("video_field cannot be blank.")
        if not self.flag_field.strip():
            raise ValueError("flag_field cannot be blank.")
        if not self.model.strip():
            raise ValueError("model cannot be blank.")
        if any(not isinstance(name, str) or not name.strip() for name in self.classes):
            raise ValueError("classes must contain non-empty strings.")
        normalized = tuple(name.strip() for name in self.classes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("classes must be unique.")
        object.__setattr__(self, "video_field", self.video_field.strip())
        object.__setattr__(self, "flag_field", self.flag_field.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "classes", list(normalized))
        return self


class DetectPresenceMap:
    """Replace a boolean presence Map with Detect, a keep Filter, and a code Map."""

    metadata = DirectiveMetadata(
        name="detect_presence_map",
        description=(
            "Replace a single-boolean prompt Map with YOLOE Detect, a filter "
            "that drops rows with no boxes, and a code Map that sets the "
            "boolean from those detections. The prompt Map is removed."
        ),
        when_to_use=(
            "Use when the Map's only output is a boolean meaning the named "
            "objects are visible, so the first detection is the answer and "
            "the VLM call can be removed."
        ),
    )
    params_type = DetectPresenceMapParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        matches: list[RewriteMatch] = []
        for entry in prompt_map_entries(index):
            spec = entry.node.spec
            if not isinstance(spec, PromptSpec):
                continue
            if _single_boolean_field(spec) is None:
                continue
            paths = record_paths(spec.parts)
            summary = (
                "Boolean prompt Map reading " + format_record_paths(paths)
                if paths
                else "Boolean prompt Map"
            )
            matches.append(RewriteMatch(path=entry.path, summary=summary))
        return tuple(matches)

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, DetectPresenceMapParams):
            raise MMDSRewriteError(
                "DetectPresenceMap received invalid parameters."
            )

        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "DetectPresenceMap requires a prompt-backed Map."
            )
        if node.source is None:
            raise MMDSRewriteError(
                "DetectPresenceMap requires a Map source."
            )
        if _single_boolean_field(node.spec) != params.flag_field:
            raise MMDSRewriteError(
                "DetectPresenceMap requires a Map whose schema is exactly "
                f"{{{params.flag_field!r}: 'boolean'}}."
            )

        video_reference = RecordPath((params.video_field,))
        if video_reference not in record_paths(node.spec.parts):
            raise MMDSRewriteError(
                "Matched Map does not directly reference video field "
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
                stop_after_n=1,
            ),
            name="rewrite_detect_presence",
        )
        gated = DatasetExpr(
            kind="filter",
            source=detected,
            spec=_KEEP_UDF,
            name="rewrite_keep_detections",
        )
        replacement = DatasetExpr(
            kind="map",
            source=gated,
            spec=UdfSpec(
                module="udfs.detection_ops",
                name=_PRESENCE_UDF,
                args=(params.flag_field,),
            ),
            name=node.name,
            replace=node.replace,
        )
        return index.replace(match.path, replacement)


def _single_boolean_field(spec: PromptSpec) -> str | None:
    schema = spec.output_schema or {}
    if len(schema) != 1:
        return None
    field, value = next(iter(schema.items()))
    if value != "boolean":
        return None
    return field
