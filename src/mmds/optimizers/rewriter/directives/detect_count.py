from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ....model import (
    MAX_TRACK_FRAME_GAP,
    DatasetExpr,
    DetectSpec,
    PromptSpec,
    UdfSpec,
)
from ..core import (
    DirectiveMetadata,
    PlanIndex,
    RewriteMatch,
)
from ..errors import MMDSRewriteError
from ._detect import (
    DETECTIONS_FIELD,
    DetectClassesParams,
    require_video_reference,
    single_schema_field,
)
from ._prompt import (
    format_record_paths,
    prompt_map_entries,
    record_paths,
)


_COUNT_UDF = "map_detection_track_count"


class DetectCountMapParams(DetectClassesParams):
    count_field: str = Field(
        min_length=1,
        description=(
            "Integer Map output field replaced by the number of distinct "
            "tracks. It must be the Map's only schema field."
        ),
    )
    frame_stride: int = Field(
        default=1,
        description=(
            "Run YOLOE on every Nth frame. Must be at most "
            f"{MAX_TRACK_FRAME_GAP}; a wider stride splits one object into "
            "a new track at every sample."
        ),
    )
    min_track_boxes: int = Field(
        default=1,
        description=(
            "Count a track only when it has at least this many boxes. Values "
            "above 1 drop short fragments from false positives or from fast "
            "objects that the tracker split into several tracks."
        ),
    )

    @model_validator(mode="after")
    def validate_count_fields(self) -> DetectCountMapParams:
        if not self.count_field.strip():
            raise ValueError("count_field cannot be blank.")
        if (
            isinstance(self.min_track_boxes, bool)
            or not isinstance(self.min_track_boxes, int)
            or self.min_track_boxes < 1
        ):
            raise ValueError("min_track_boxes must be an integer >= 1.")
        if (
            isinstance(self.frame_stride, bool)
            or not isinstance(self.frame_stride, int)
            or not 1 <= self.frame_stride <= MAX_TRACK_FRAME_GAP
        ):
            raise ValueError(
                f"frame_stride must be an integer in [1, {MAX_TRACK_FRAME_GAP}]."
            )
        object.__setattr__(self, "count_field", self.count_field.strip())
        return self


class DetectCountMap:
    """Replace an integer count Map with Detect and a track-count code Map."""

    metadata = DirectiveMetadata(
        name="detect_count_map",
        description=(
            "Replace a single-integer prompt Map with YOLOE Detect over the "
            "whole video and a code Map that sets the integer to the number "
            "of distinct tracks. The prompt Map is removed."
        ),
        when_to_use=(
            "Use when the Map's only output is how many of the named objects "
            "appear, and one tracked object is one count: a single continuous "
            "shot from a fixed camera where each object crosses the view once. "
            "Not for attributes, actions, edited compilations, or objects that "
            "leave and re-enter the frame, which the tracker counts again."
        ),
    )
    params_type = DetectCountMapParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        matches: list[RewriteMatch] = []
        for entry in prompt_map_entries(index):
            spec = entry.node.spec
            if not isinstance(spec, PromptSpec):
                continue
            if single_schema_field(spec, "integer") is None:
                continue
            paths = record_paths(spec.parts)
            summary = (
                "Integer prompt Map reading " + format_record_paths(paths)
                if paths
                else "Integer prompt Map"
            )
            matches.append(RewriteMatch(path=entry.path, summary=summary))
        return tuple(matches)

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, DetectCountMapParams):
            raise MMDSRewriteError("DetectCountMap received invalid parameters.")

        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError("DetectCountMap requires a prompt-backed Map.")
        if node.source is None:
            raise MMDSRewriteError("DetectCountMap requires a Map source.")
        if single_schema_field(node.spec, "integer") != params.count_field:
            raise MMDSRewriteError(
                "DetectCountMap requires a Map whose schema is exactly "
                f"{{{params.count_field!r}: 'integer'}}."
            )
        require_video_reference(node.spec, params.video_field)

        detected = DatasetExpr(
            kind="detect",
            source=node.source,
            spec=DetectSpec(
                video_field=params.video_field,
                classes=tuple(params.classes),
                model=params.model,
                output_field=DETECTIONS_FIELD,
                frame_stride=params.frame_stride,
            ),
            name="rewrite_detect_count",
        )
        count_args: tuple[str, ...] = (params.count_field,)
        if params.min_track_boxes > 1:
            count_args += (str(params.min_track_boxes),)
        replacement = DatasetExpr(
            kind="map",
            source=detected,
            spec=UdfSpec(
                module="udfs.detection_ops",
                name=_COUNT_UDF,
                args=count_args,
            ),
            name=node.name,
            replace=node.replace,
        )
        return index.replace(match.path, replacement)
