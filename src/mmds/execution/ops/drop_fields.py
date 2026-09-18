from __future__ import annotations

from ...model import DatasetExpr, DropFieldsSpec, MMDSValidationError, Row


def _apply_drop_fields(node: DatasetExpr, row: Row) -> Row:
    spec = node.spec
    if not isinstance(spec, DropFieldsSpec):
        raise MMDSValidationError("DropFields requires a DropFieldsSpec.")

    missing = [field for field in spec.fields if field not in row]
    if missing:
        raise MMDSValidationError(
            f"DropFields cannot remove missing fields: {missing!r}."
        )

    result = dict(row)
    for field in spec.fields:
        del result[field]
    return result
