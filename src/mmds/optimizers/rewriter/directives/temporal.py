from __future__ import annotations

from dataclasses import replace
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from ....model import (
    DatasetExpr,
    PromptSpec,
    RecordPath,
    UdfSpec,
    VideoMapSpec,
)
from ..directive import (
    DirectiveMetadata,
    MMDSRewriteError,
    PlanIndex,
    RewriteMatch,
    make_match,
    prompt_map_nodes,
    record_paths,
)


INTERVALS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
        },
        "required": ["start", "end"],
        "additionalProperties": False,
    },
}


class TemporalPushdownParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(
        min_length=1,
        description="Input field containing the complete video.",
    )
    transcript_field: str = Field(
        min_length=1,
        description="Input field containing timestamped transcript cues.",
    )
    query_field: str = Field(
        min_length=1,
        description="Input field containing the user's event query.",
    )
    candidate_prompt: str = Field(
        min_length=1,
        description=(
            "Instruction used only by the transcript candidate stage. Ask for "
            "high-recall source-time intervals from the transcript and query; do "
            "not ask this stage to extract video, verify events, rebase times, or "
            "reconcile results."
        ),
    )


class _TemporalDirective:
    params_type = TemporalPushdownParams
    views_field = "_mmds_candidate_views"

    def __init__(
        self,
        *,
        identity_fields: str | list[str] | tuple[str, ...] = (),
        padding_seconds: float = 0.0,
        max_views: int = 8,
        max_total_video_seconds: float = 600.0,
    ) -> None:
        if isinstance(identity_fields, str):
            normalized_identity = (identity_fields,)
        elif isinstance(identity_fields, (list, tuple)):
            normalized_identity = tuple(identity_fields)
        else:
            raise TypeError(
                "Temporal identity_fields must be a string or sequence of strings."
            )
        if any(
            not isinstance(field, str) or not field
            for field in normalized_identity
        ):
            raise TypeError(
                "Temporal identity_fields must contain non-empty strings."
            )
        if len(set(normalized_identity)) != len(normalized_identity):
            raise ValueError("Temporal identity_fields must be unique.")
        if (
            not isinstance(padding_seconds, (int, float))
            or isinstance(padding_seconds, bool)
            or not isfinite(padding_seconds)
            or padding_seconds < 0
        ):
            raise ValueError(
                "Temporal padding_seconds must be a finite non-negative number."
            )
        if (
            not isinstance(max_views, int)
            or isinstance(max_views, bool)
            or max_views <= 0
        ):
            raise ValueError("Temporal max_views must be a positive integer.")
        if (
            not isinstance(max_total_video_seconds, (int, float))
            or isinstance(max_total_video_seconds, bool)
            or not isfinite(max_total_video_seconds)
            or max_total_video_seconds <= 0
        ):
            raise ValueError(
                "Temporal max_total_video_seconds must be a finite positive number."
            )
        self.identity_fields = normalized_identity
        self.padding_seconds = float(padding_seconds)
        self.max_views = max_views
        self.max_total_video_seconds = float(max_total_video_seconds)

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        return tuple(
            make_match(
                self.metadata,
                entry.path,
                "Prompt-backed Map with fields "
                + ", ".join(sorted(entry.field_effects.reads)),
            )
            for entry in prompt_map_nodes(index)
        )

    def _prepare(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> tuple[
        DatasetExpr,
        TemporalPushdownParams,
        PromptSpec,
        DatasetExpr,
        tuple[str, ...],
    ]:
        if not isinstance(params, TemporalPushdownParams):
            raise MMDSRewriteError("Temporal directive received invalid parameters.")
        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "Temporal pushdown requires a prompt-backed Map."
            )
        if node.source is None:
            raise MMDSRewriteError("Temporal pushdown requires a Map source.")
        video_reference = RecordPath((params.video_field,))
        paths = record_paths(node.spec.parts)
        if video_reference not in paths:
            raise MMDSRewriteError(
                f"Matched Map does not reference video field {params.video_field!r}."
            )
        group_by = self._derive_group_by(index, match, paths, params.video_field)
        if self.views_field in index.known_fields():
            raise MMDSRewriteError(
                f"Temporary field {self.views_field!r} collides with an existing field."
            )

        candidates = DatasetExpr(
            kind="map",
            source=node.source,
            spec=PromptSpec(
                parts=(
                    params.candidate_prompt,
                    f"\n{params.query_field}:\n",
                    RecordPath((params.query_field,)),
                    f"\n{params.transcript_field}:\n",
                    RecordPath((params.transcript_field,)),
                ),
                output_schema={self.views_field: INTERVALS_SCHEMA},
            ),
            name="rewrite_transcript_candidates",
        )
        return node, params, node.spec, candidates, group_by

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
                fields.extend(entry.expr.group_by)
        fields.extend(
            path.path[0]
            for path in prompt_paths
            if path.path and path.path[0] != video_field
        )
        group_by = tuple(dict.fromkeys(fields))
        if not group_by:
            raise MMDSRewriteError(
                "Temporal pushdown could not derive grouping fields. Configure "
                "identity_fields on the directive or preserve a non-video prompt field."
            )
        return group_by

    def _video_spec(
        self,
        params: TemporalPushdownParams,
        map_spec: PromptSpec,
        group_by: tuple[str, ...],
    ) -> VideoMapSpec:
        return VideoMapSpec(
            video_field=params.video_field,
            views_field=self.views_field,
            group_by=group_by,
            map_spec=map_spec,
            padding_time=self.padding_seconds,
            max_views=self.max_views,
            max_total_video_seconds=self.max_total_video_seconds,
            clip_field="clip",
        )


class JointTemporalPushdown(_TemporalDirective):
    metadata = DirectiveMetadata(
        name="temporal_pushdown_joint",
        description="Use transcript-selected video views to answer once across all selected views.",
        when_to_use="Use for classification, extraction, or question answering with one final answer.",
    )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        node, typed, original_spec, candidates, group_by = self._prepare(
            index, match, params
        )
        replacement = DatasetExpr(
            kind="video_map",
            source=candidates,
            spec=self._video_spec(typed, original_spec, group_by),
            name=node.name,
        )
        return index.replace(match.path, replacement)


class PerViewTemporalPushdown(_TemporalDirective):
    metadata = DirectiveMetadata(
        name="temporal_pushdown_per_view",
        description="Use transcript-selected video views to localize events in each view and reconcile them.",
        when_to_use="Use for event localization or exhaustive retrieval with source-time intervals.",
    )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        node, typed, original_spec, candidates, group_by = self._prepare(
            index, match, params
        )
        output_schema = original_spec.output_schema or {}
        if set(output_schema) != {"events"}:
            raise MMDSRewriteError(
                "Per-view temporal pushdown currently requires schema={'events': ...}."
            )
        local_spec = replace(
            original_spec,
            parts=(
                "Inspect only this video view. Return event times relative to "
                "the beginning of the view under clip_events.\n",
                *original_spec.parts,
            ),
            output_schema={"clip_events": output_schema["events"]},
        )
        localized = DatasetExpr(
            kind="video_map_each",
            source=candidates,
            spec=self._video_spec(typed, local_spec, group_by),
            name="rewrite_verify_views",
        )
        rebased = DatasetExpr(
            kind="map",
            source=localized,
            spec=UdfSpec(
                module="udfs.temporal_ops",
                name="rebase_clip_events",
            ),
            name="rewrite_rebase_events",
        )
        reconciled = DatasetExpr(
            kind="reduce",
            source=rebased,
            group_by=group_by,
            spec=UdfSpec(
                module="udfs.temporal_ops",
                name="reconcile_events",
            ),
            name=node.name,
        )
        return index.replace(match.path, reconciled)
