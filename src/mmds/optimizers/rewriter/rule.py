from __future__ import annotations

from dataclasses import replace

from ...model import DatasetExpr


def optimize(plan: DatasetExpr) -> DatasetExpr:
    """Rebuild the plan while preserving object-identity sharing.

    Nodes reached through several parents (for example a self-join's shared
    upstream) are visited once. Structurally equal but distinct subplans are
    *not* fused: sharing is by Python object identity (or the same assignment
    name in source), not by value equality. That keeps non-deterministic
    ``Map`` / ``Detect`` nodes from collapsing into a single evaluation.
    """
    memo: dict[int, DatasetExpr] = {}

    def visit(node: DatasetExpr) -> DatasetExpr:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        source = visit(node.source) if node.source is not None else None
        right_source = (
            visit(node.right_source) if node.right_source is not None else None
        )
        rebuilt = replace(node, source=source, right_source=right_source)
        memo[id(node)] = rebuilt
        return rebuilt

    return visit(plan)


canonicalize = optimize
