"""Explicit cross-modal temporal-pushdown rewrites."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..dsl import Map, PadInterval, ReconcileIntervals, Reduce, Resolve, Unnest, View
from ..model import (
    DatasetExpr,
    ForEachPrompt,
    MMDSValidationError,
    PromptPart,
    PromptSpec,
    RecordPath,
)


@dataclass(frozen=True)
class ModalitySubstitution:
    """Replace one named semantic Map with a schema-compatible prompt."""

    target_name: str
    replacement_prompt: PromptSpec

    def rewrite(self, plan: DatasetExpr) -> DatasetExpr:
        replacement_count = 0

        def visit(node: DatasetExpr) -> DatasetExpr:
            nonlocal replacement_count
            source = visit(node.source) if node.source is not None else None
            rebuilt = replace(node, source=source)
            if rebuilt.kind != "map" or rebuilt.name != self.target_name:
                return rebuilt
            if not isinstance(rebuilt.spec, PromptSpec):
                raise MMDSValidationError(
                    "O1 can substitute only a prompt-backed Map."
                )
            if rebuilt.spec.output_schema != self.replacement_prompt.output_schema:
                raise MMDSValidationError(
                    "O1 replacement prompt must preserve the Map output schema."
                )
            replacement_count += 1
            return replace(rebuilt, spec=self.replacement_prompt)

        rewritten = visit(plan)
        if replacement_count != 1:
            raise MMDSValidationError(
                f"O1 expected exactly one Map named {self.target_name!r}; "
                f"found {replacement_count}."
            )
        return rewritten


@dataclass(frozen=True)
class CrossModalTemporalPushdown:
    """Configuration for the O2 transcript-to-video rewrite.

    The rewrite targets one named prompt-backed Map. It uses a transcript
    candidate prompt to retain source-time intervals, pads and resolves those
    intervals, materializes clips, applies the original video prompt to each
    clip, and reconciles clip-relative results by input key.
    """

    target_name: str
    candidate_prompt: PromptSpec
    group_by: tuple[str, ...]
    preserve_fields: tuple[str, ...]
    video_field: str = "video"
    transcript_field: str = "timestamped_transcript"
    duration_field: str = "duration_seconds"
    candidate_field: str = "candidate_intervals"
    event_field: str = "events"
    window_start_field: str = "window_start_seconds"
    window_end_field: str = "window_end_seconds"
    clip_field: str = "candidate_video"
    padding_seconds: float = 30.0
    deduplication_tiou_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not self.target_name:
            raise MMDSValidationError("O2 requires a non-empty target operator name.")
        if not self.group_by:
            raise MMDSValidationError("O2 requires one or more reconciliation keys.")
        schema = self.candidate_prompt.output_schema
        if schema is None or self.candidate_field not in schema:
            raise MMDSValidationError(
                f"O2 candidate prompt schema must declare {self.candidate_field!r}."
            )
        if not any(
            _contains_record_path(part, (self.transcript_field,))
            for part in self.candidate_prompt.parts
        ):
            raise MMDSValidationError(
                "O2 candidate prompt must read the configured transcript field."
            )

    def rewrite(self, plan: DatasetExpr) -> DatasetExpr:
        """Rewrite exactly one named video-localization Map in ``plan``."""
        replacement_count = 0

        def visit(node: DatasetExpr) -> DatasetExpr:
            nonlocal replacement_count
            source = visit(node.source) if node.source is not None else None
            rebuilt = replace(node, source=source)
            if rebuilt.kind == "map" and rebuilt.name == self.target_name:
                replacement_count += 1
                return self._replace_localizer(rebuilt)
            return rebuilt

        rewritten = visit(plan)
        if replacement_count != 1:
            raise MMDSValidationError(
                f"O2 expected exactly one Map named {self.target_name!r}; "
                f"found {replacement_count}."
            )
        return rewritten

    def _replace_localizer(self, localizer: DatasetExpr) -> DatasetExpr:
        if localizer.source is None or not isinstance(localizer.spec, PromptSpec):
            raise MMDSValidationError(
                "O2 can rewrite only a prompt-backed Map with an input source."
            )
        localized_prompt = _replace_record_path(
            localizer.spec, (self.video_field,), (self.clip_field,)
        )
        if (
            localizer.spec.output_schema is None
            or self.event_field not in localizer.spec.output_schema
        ):
            raise MMDSValidationError(
                f"O2 video-localizer schema must declare {self.event_field!r}."
            )

        candidates = DatasetExpr(
            kind="map",
            source=localizer.source,
            spec=self.candidate_prompt,
            name="o2_transcript_candidates",
        )
        one_candidate = Unnest(
            candidates, self.candidate_field, name="o2_one_candidate_per_tuple"
        )
        padded = Map(
            one_candidate,
            PadInterval(
                self.candidate_field,
                self.duration_field,
                self.padding_seconds,
                output_start_field=self.window_start_field,
                output_end_field=self.window_end_field,
            ),
            name="o2_pad_candidate_window",
        )
        resolved = Resolve(
            padded,
            self.group_by,
            self.window_start_field,
            self.window_end_field,
            name="o2_resolve_overlapping_windows",
        )
        clips = View(
            resolved,
            self.video_field,
            self.window_start_field,
            self.window_end_field,
            output_field=self.clip_field,
            name="o2_materialize_candidate_clips",
        )
        clip_results = DatasetExpr(
            kind="map",
            source=clips,
            spec=localized_prompt,
            name=localizer.name,
        )
        return Reduce(
            clip_results,
            self.group_by,
            ReconcileIntervals(
                self.event_field,
                self.window_start_field,
                self.window_end_field,
                output_field=self.event_field,
                preserve_fields=self.preserve_fields,
                deduplication_tiou_threshold=self.deduplication_tiou_threshold,
            ),
            name="o2_reconcile_clip_results",
        )


def _replace_record_path(
    prompt: PromptSpec, old_path: tuple[str, ...], new_path: tuple[str, ...]
) -> PromptSpec:
    replacements = 0

    def rewrite_part(part: PromptPart) -> PromptPart:
        nonlocal replacements
        if isinstance(part, RecordPath) and part.path == old_path:
            replacements += 1
            return RecordPath(new_path)
        if isinstance(part, ForEachPrompt):
            return ForEachPrompt(tuple(rewrite_part(child) for child in part.parts))
        return part

    parts = tuple(rewrite_part(part) for part in prompt.parts)
    if replacements == 0:
        raise MMDSValidationError(
            f"O2 target prompt does not reference video field {old_path[0]!r}."
        )
    return PromptSpec(parts=parts, output_schema=prompt.output_schema)


def _contains_record_path(part: PromptPart, path: tuple[str, ...]) -> bool:
    if isinstance(part, RecordPath):
        return part.path == path
    if isinstance(part, ForEachPrompt):
        return any(_contains_record_path(child, path) for child in part.parts)
    return False


__all__ = ["CrossModalTemporalPushdown", "ModalitySubstitution"]
