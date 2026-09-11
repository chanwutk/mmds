from __future__ import annotations

from ...model import DatasetExpr, Row
from .._spec import PromptExecutor, _execute_spec
from ..context import ExecutionStats


def _apply_filter(
    node: DatasetExpr,
    row: Row,
    prompt_executor: PromptExecutor | None,
    execution_stats: ExecutionStats | None = None,
) -> bool:
    return bool(_execute_spec(node, row, prompt_executor, execution_stats))
