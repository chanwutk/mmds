from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ....model import (
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


_KEEP_UDF = UdfSpec(
    module="udfs.detection_ops",
    name="keep_rows_with_detections",
)
_PRESENCE_UDF = "map_detection_presence"
_AT_LEAST_KEEP_UDF = "keep_rows_with_at_least_n_tracks"
_AT_LEAST_MAP_UDF = "map_at_least_n_tracks"


class DetectPresenceMapParams(DetectClassesParams):
    min_tracks: int = Field(
        default=1,
        description=(
            "Minimum number of distinct tracks required. 1 means presence. "
            "A larger value means at least that many instances. The scan stops "
            "once this many tracks exist."
        ),
    )
    flag_field: str = Field(
        min_length=1,
        description=(
            "Boolean Map output field replaced by detection presence. It must "
            "be the Map's only schema field."
        ),
    )

    @model_validator(mode="after")
    def validate_presence_fields(self) -> DetectPresenceMapParams:
        if not self.flag_field.strip():
            raise ValueError("flag_field cannot be blank.")
        if (
            isinstance(self.min_tracks, bool)
            or not isinstance(self.min_tracks, int)
            or self.min_tracks < 1
        ):
            raise ValueError("min_tracks must be an integer >= 1.")
        object.__setattr__(self, "flag_field", self.flag_field.strip())
        return self


class DetectPresenceMap:
    """Replace a boolean presence Map with Detect, a keep Filter, and a code Map."""

    metadata = DirectiveMetadata(
        name="detect_presence_map",
        description=(
            "Replace a single-boolean prompt Map with YOLOE Detect, a filter "
            "that drops rows below the track threshold, and a code Map that "
            "sets the boolean from that count. The prompt Map is removed."
        ),
        when_to_use=(
            "Use when the Map's only output is a boolean meaning at least n "
            "instances of the named objects are visible. n=1 is presence. "
            "The first n tracks are the answer and the VLM call can be removed."
        ),
    )
    params_type = DetectPresenceMapParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        matches: list[RewriteMatch] = []
        for entry in prompt_map_entries(index):
            spec = entry.node.spec
            if not isinstance(spec, PromptSpec):
                continue
            if single_schema_field(spec, "boolean") is None:
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
        if single_schema_field(node.spec, "boolean") != params.flag_field:
            raise MMDSRewriteError(
                "DetectPresenceMap requires a Map whose schema is exactly "
                f"{{{params.flag_field!r}: 'boolean'}}."
            )
        require_video_reference(node.spec, params.video_field)

        if params.min_tracks == 1:
            keep_spec = _KEEP_UDF
            map_spec = UdfSpec(
                module="udfs.detection_ops",
                name=_PRESENCE_UDF,
                args=(params.flag_field,),
            )
        else:
            threshold = str(params.min_tracks)
            keep_spec = UdfSpec(
                module="udfs.detection_ops",
                name=_AT_LEAST_KEEP_UDF,
                args=(threshold,),
            )
            map_spec = UdfSpec(
                module="udfs.detection_ops",
                name=_AT_LEAST_MAP_UDF,
                args=(params.flag_field, threshold),
            )
        detected = DatasetExpr(
            kind="detect",
            source=node.source,
            spec=DetectSpec(
                video_field=params.video_field,
                classes=tuple(params.classes),
                model=params.model,
                output_field=DETECTIONS_FIELD,
                stop_after_n=params.min_tracks,
            ),
            name="rewrite_detect_presence",
        )
        gated = DatasetExpr(
            kind="filter",
            source=detected,
            spec=keep_spec,
            name="rewrite_keep_detections",
        )
        replacement = DatasetExpr(
            kind="map",
            source=gated,
            spec=map_spec,
            name=node.name,
            replace=node.replace,
        )
        return index.replace(match.path, replacement)
