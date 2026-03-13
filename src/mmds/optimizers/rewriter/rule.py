from __future__ import annotations

from dataclasses import replace

from ...model import DatasetExpr


def optimize(plan: DatasetExpr) -> DatasetExpr:
    memo: dict[DatasetExpr, DatasetExpr] = {}

    def visit(node: DatasetExpr) -> DatasetExpr:
        source = visit(node.source) if node.source is not None else None
        rebuilt = replace(node, source=source)
        cached = memo.get(rebuilt)
        if cached is not None:
            return cached
        memo[rebuilt] = rebuilt
        return rebuilt

    return visit(plan)


canonicalize = optimize
