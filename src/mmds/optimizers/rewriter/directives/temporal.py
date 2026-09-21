from __future__ import annotations

from dataclasses import replace
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....model import (
    DatasetExpr,
    PromptSpec,
    RecordPath,
    UdfSpec,
    VideoMapSpec,
)
from ..core import DirectiveMetadata, PlanIndex, RewriteMatch
from ..errors import MMDSRewriteError
from ._prompt import format_record_paths, prompt_map_entries, record_paths


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
        description="Input field containing the user's query.",
    )
    candidate_prompt: str = Field(
        min_length=1,
        description=(
            "Instruction for finding high-recall source-time intervals from the "
            "transcript and query. It must not ask this stage to inspect video, "
            "verify events, rebase times, or reconcile results."
        ),
    )

    @model_validator(mode="after")
    def validate_fields(self) -> TemporalPushdownParams:
        fields = (self.video_field, self.transcript_field, self.query_field)
        if any(not field.strip() for field in fields):
            raise ValueError("Temporal field names cannot be blank.")
        if len(set(fields)) != len(fields):
            raise ValueError(
                "Temporal video, transcript, and query fields must be distinct."
            )
        if not self.candidate_prompt.strip():
            raise ValueError("Temporal candidate_prompt cannot be blank.")
        return self


class _TemporalDirective:
    params_type = TemporalPushdownParams
    views_field = "_mmds_candidate_views"
    clip_field = "clip"

    def __init__(
        self,
        *,
        identity_fields: str | list[str] | tuple[str, ...],
        padding_seconds: float = 0.0,
    ) -> None:
        if isinstance(identity_fields, str):
            normalized_identity = (identity_fields,)
        elif isinstance(identity_fields, (list, tuple)):
            normalized_identity = tuple(identity_fields)
        else:
            raise TypeError(
                "Temporal identity_fields must be a string or sequence of strings."
            )
        if not normalized_identity:
            raise ValueError(
                "Temporal identity_fields must contain at least one stable source key."
            )
        if any(
            not isinstance(field, str) or not field.strip()
            for field in normalized_identity
        ):
            raise TypeError(
                "Temporal identity_fields must contain non-empty strings."
            )
        if len(set(normalized_identity)) != len(normalized_identity):
            raise ValueError("Temporal identity_fields must be unique.")
        if self.views_field in normalized_identity or self.clip_field in normalized_identity:
            raise ValueError(
                "Temporal identity_fields cannot use reserved rewrite fields."
            )
        if (
            not isinstance(padding_seconds, (int, float))
            or isinstance(padding_seconds, bool)
            or not isfinite(padding_seconds)
            or padding_seconds < 0
        ):
            raise ValueError(
                "Temporal padding_seconds must be a finite non-negative number."
            )

        self.identity_fields = normalized_identity
        self.padding_seconds = float(padding_seconds)

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
        if params.video_field in self.identity_fields:
            raise MMDSRewriteError(
                "Temporal identity_fields cannot include the video field because "
                "video values are not stable grouping keys."
            )

        paths = record_paths(node.spec.parts)
        video_reference = RecordPath((params.video_field,))
        query_reference = RecordPath((params.query_field,))
        if video_reference not in paths:
            raise MMDSRewriteError(
                f"Matched Map does not directly reference video field "
                f"{params.video_field!r}."
            )
        if query_reference not in paths:
            raise MMDSRewriteError(
                f"Matched Map does not directly reference query field "
                f"{params.query_field!r}."
            )
        nested_paths = tuple(path for path in paths if len(path.path) != 1)
        if nested_paths:
            raise MMDSRewriteError(
                "Temporal pushdown currently supports only top-level prompt fields; "
                f"found {format_record_paths(nested_paths)}."
            )

        group_by = self._derive_group_by(
            index,
            match,
            paths,
            params.video_field,
        )
        candidates = DatasetExpr(
            kind="map",
            source=node.source,
            spec=PromptSpec(
                parts=(
                    params.candidate_prompt,
                    f"\n{params.query_field}:\n",
                    query_reference,
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

        # Nodes with shorter prefix paths are downstream ancestors of the match.
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
        if self.views_field in group_by or self.clip_field in group_by:
            raise MMDSRewriteError(
                "Derived grouping fields collide with reserved rewrite fields."
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
            clip_field=self.clip_field,
        )


class JointTemporalPushdown(_TemporalDirective):
    """Select transcript candidates, then answer once across all video views."""

    metadata = DirectiveMetadata(
        name="temporal_pushdown_joint",
        description=(
            "Use transcript-selected video views to answer once across all selected views."
        ),
        when_to_use=(
            "Use for classification, extraction, or question answering with one final answer."
        ),
    )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        node, typed, original_spec, candidates, group_by = self._prepare(
            index,
            match,
            params,
        )
        replacement = DatasetExpr(
            kind="video_map",
            source=candidates,
            spec=self._video_spec(typed, original_spec, group_by),
            name=node.name,
        )
        return index.replace(match.path, replacement)


class PerViewTemporalPushdown(_TemporalDirective):
    """Select transcript candidates, localize per view, rebase, then collect."""

    metadata = DirectiveMetadata(
        name="temporal_pushdown_per_view",
        description=(
            "Use transcript-selected video views to localize events in every view."
        ),
        when_to_use=(
            "Use for event localization or exhaustive retrieval with source-time intervals."
        ),
    )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        node, typed, original_spec, candidates, group_by = self._prepare(
            index,
            match,
            params,
        )
        output_schema = original_spec.output_schema or {}
        if set(output_schema) != {"events"}:
            raise MMDSRewriteError(
                "Per-view temporal pushdown currently requires an output schema "
                "with exactly one field named 'events'."
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
