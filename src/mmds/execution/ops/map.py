from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...model import DatasetExpr, MMDSValidationError, Row
from .._spec import PromptExecutor, _execute_spec
from ..context import ExecutionStats


def _apply_map(
    node: DatasetExpr,
    row: Row,
    prompt_executor: PromptExecutor | None,
    execution_stats: ExecutionStats | None = None,
) -> Row:
    updates = _execute_spec(node, row, prompt_executor, execution_stats)
    if not isinstance(updates, Mapping):
        raise MMDSValidationError("Map operations must return mapping-like field updates.")
    result = dict(row)
    result.update(dict(updates))
    return result
