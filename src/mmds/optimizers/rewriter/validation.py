from __future__ import annotations

from ...model import DatasetExpr, PromptSpec, QueryProgram, VideoMapSpec
from ...parser import parse_query
from ...render import program_from_plan, render_query
from .directive import MMDSRewriteError, PlanIndex


_RESERVED_PREFIX = "_mmds_"


def validate_rewrite(original: QueryProgram, rewritten: DatasetExpr) -> None:
    original_inputs = _reachable_input_paths(original.output_expr)
    rewritten_inputs = _reachable_input_paths(rewritten)
    if rewritten_inputs != original_inputs:
        raise MMDSRewriteError(
            "Rewrites must preserve the reachable Input(...) file paths."
        )

    original_schema = semantic_output_schema(original.output_expr)
    rewritten_schema = semantic_output_schema(rewritten)
    if (
        original_schema is not None
        and rewritten_schema is not None
        and rewritten_schema != original_schema
    ):
        raise MMDSRewriteError(
            "Rewrites must preserve the final declared output schema."
        )

    _validate_reserved_fields(rewritten)

    try:
        rendered = render_query(program_from_plan(rewritten))
        reparsed = parse_query(rendered)
    except (TypeError, ValueError) as exc:
        raise MMDSRewriteError(
            f"Rewritten plan could not round-trip through normalized Python: {exc}"
        ) from exc
    if reparsed.output_expr != rewritten:
        raise MMDSRewriteError(
            "Rewritten plan changed during its render/parse round trip."
        )


def semantic_output_schema(plan: DatasetExpr):
    node: DatasetExpr | None = plan
    while node is not None:
        if node.kind in {"map", "reduce"}:
            if isinstance(node.spec, PromptSpec):
                return node.spec.output_schema
            return None
        if node.kind in {"video_map", "video_map_each"}:
            if isinstance(node.spec, VideoMapSpec) and isinstance(
                node.spec.map_spec, PromptSpec
            ):
                return node.spec.map_spec.output_schema
            return None
        node = node.source
    return None


def _reachable_input_paths(plan: DatasetExpr) -> tuple[str, ...]:
    return tuple(
        sorted(
            node.input_path
            for node in plan.walk_postorder()
            if node.kind == "input" and node.input_path is not None
        )
    )


def _validate_reserved_fields(plan: DatasetExpr) -> None:
    available: set[str] = set()
    for entry in reversed(PlanIndex.build(plan).entries):
        effects = entry.field_effects
        missing_reads = {
            field
            for field in effects.reads
            if field.startswith(_RESERVED_PREFIX) and field not in available
        }
        if missing_reads:
            raise MMDSRewriteError(
                f"Reserved rewrite fields are read before they are written: {sorted(missing_reads)!r}."
            )
        missing_drops = {
            field
            for field in effects.drops
            if field.startswith(_RESERVED_PREFIX) and field not in available
        }
        if missing_drops:
            raise MMDSRewriteError(
                f"Reserved rewrite fields are dropped before they are written: {sorted(missing_drops)!r}."
            )
        available.difference_update(effects.drops)
        available.update(effects.writes)
