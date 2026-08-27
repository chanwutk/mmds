from __future__ import annotations

from dataclasses import replace

from ...model import DatasetExpr


def optimize(plan: DatasetExpr) -> DatasetExpr:
    memo: dict[DatasetExpr, DatasetExpr] = {}

    def visit(node: DatasetExpr) -> DatasetExpr:
        source = visit(node.source) if node.source is not None else None
        right_source = (
            visit(node.right_source) if node.right_source is not None else None
        )
        rebuilt = replace(node, source=source, right_source=right_source)
        cached = memo.get(rebuilt)
        if cached is not None:
            return cached
        memo[rebuilt] = rebuilt
        return rebuilt

    return visit(plan)


canonicalize = optimize
