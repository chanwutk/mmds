from __future__ import annotations

from ...model import DatasetExpr, PromptSpec, QueryProgram, VideoMapSpec
from ...parser import parse_query
from ...render import program_from_plan, render_query
from .errors import MMDSRewriteError


def validate_rewrite_structure(
    original: QueryProgram,
    rewritten: DatasetExpr,
) -> None:
    """Validate structural invariants, not semantic equivalence."""

    if _reachable_input_paths(rewritten) != _reachable_input_paths(
        original.output_expr
    ):
        raise MMDSRewriteError(
            "Rewrites must preserve the reachable Input(...) file paths."
        )

    original_schema = _declared_output_schema(original.output_expr)
    rewritten_schema = _declared_output_schema(rewritten)
    if original_schema is not None and rewritten_schema != original_schema:
        raise MMDSRewriteError(
            "Rewrites must preserve the final declared output schema."
        )

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


def _declared_output_schema(plan: DatasetExpr):
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
