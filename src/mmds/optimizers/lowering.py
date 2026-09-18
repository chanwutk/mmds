from __future__ import annotations

from dataclasses import replace

from ..model import (
    DatasetExpr,
    ForEachPrompt,
    MMDSValidationError,
    PromptPart,
    PromptSpec,
    RecordPath,
    UdfSpec,
    VideoMapSpec,
    ViewBudgetSpec,
    WindowSpec,
)


def lower_video_ops(plan: DatasetExpr) -> DatasetExpr:
    """Lower logical VideoMap nodes into executable MMDS primitives."""
    source = lower_video_ops(plan.source) if plan.source is not None else None
    rebuilt = replace(plan, source=source)
    if rebuilt.kind not in {"video_map", "video_map_each"}:
        return rebuilt
    return _lower_video_map(rebuilt)


def _lower_video_map(node: DatasetExpr) -> DatasetExpr:
    spec = node.spec
    if not isinstance(spec, VideoMapSpec) or node.source is None:
        raise MMDSValidationError("VideoMap lowering requires a VideoMapSpec and source.")

    unnested = DatasetExpr(
        kind="unnest",
        source=node.source,
        field=spec.views_field,
        name=_stage_name(node.name, "views"),
    )
    windowed = DatasetExpr(
        kind="window",
        source=unnested,
        spec=WindowSpec(
            video_field=spec.video_field,
            candidate_field=spec.views_field,
            output_field=spec.clip_field,
            padding_time=spec.padding_time,
        ),
        name=_stage_name(node.name, "window"),
    )
    coalesced = DatasetExpr(
        kind="coalesce",
        source=windowed,
        group_by=spec.group_by,
        field=spec.clip_field,
        name=_stage_name(node.name, "coalesce"),
    )
    budgeted = DatasetExpr(
        kind="view_budget",
        source=coalesced,
        spec=ViewBudgetSpec(
            group_by=spec.group_by,
            field=spec.clip_field,
            max_views=spec.max_views,
            max_total_video_seconds=spec.max_total_video_seconds,
        ),
        name=_stage_name(node.name, "budget"),
    )

    if node.kind == "video_map_each":
        return DatasetExpr(
            kind="map",
            source=budgeted,
            spec=_replace_video_reference(
                spec.map_spec,
                video_field=spec.video_field,
                clip_field=spec.clip_field,
            ),
            name=node.name,
        )

    if isinstance(spec.map_spec, UdfSpec):
        raise MMDSValidationError(
            "Joint VideoMap does not support UDF-backed map specifications."
        )
    return DatasetExpr(
        kind="reduce",
        source=budgeted,
        group_by=spec.group_by,
        spec=PromptSpec(
            parts=_map_parts_to_reduce_parts(
                spec.map_spec.parts,
                video_field=spec.video_field,
                clip_field=spec.clip_field,
            ),
            output_schema=spec.map_spec.output_schema,
        ),
        name=node.name,
    )


def _replace_video_reference(
    spec: PromptSpec | UdfSpec,
    *,
    video_field: str,
    clip_field: str,
) -> PromptSpec | UdfSpec:
    if isinstance(spec, UdfSpec):
        return spec
    old = RecordPath((video_field,))
    new = RecordPath((clip_field,))
    return replace(
        spec,
        parts=tuple(new if part == old else part for part in spec.parts),
    )


def _map_parts_to_reduce_parts(
    parts: tuple[PromptPart, ...],
    *,
    video_field: str,
    clip_field: str,
) -> tuple[PromptPart, ...]:
    converted: list[PromptPart] = []
    for part in parts:
        if isinstance(part, str):
            converted.append(part)
            continue
        if isinstance(part, ForEachPrompt):
            raise MMDSValidationError(
                "Map prompts cannot contain ForEach while lowering VideoMap."
            )
        reference = (
            RecordPath((clip_field,))
            if part == RecordPath((video_field,))
            else part
        )
        converted.append(ForEachPrompt(parts=(reference,)))
    return tuple(converted)


def _stage_name(name: str | None, suffix: str) -> str | None:
    if name is None:
        return None
    return f"{name}_{suffix}"
