from __future__ import annotations

from collections.abc import Iterable, Iterator

from ...model import DatasetExpr, GatherSpec, MMDSValidationError, Row


def _apply_gather(node: DatasetExpr, rows: Iterable[Row]) -> Iterator[Row]:
    spec = node.spec
    if not isinstance(spec, GatherSpec):
        raise MMDSValidationError("gather nodes require a GatherSpec.")
    fn = spec.fn.load()
    for row in rows:
        context = fn(row)
        enriched = dict(row)
        enriched[spec.output_field] = context
        yield enriched
