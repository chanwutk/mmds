from __future__ import annotations

from collections.abc import Iterable, Iterator

from ...join.hash_join import hash_join, nested_loop_join, one_to_one_hash_join
from ...model import DatasetExpr, JoinSpec, MMDSValidationError, Row


def _apply_join(
    node: DatasetExpr,
    left_rows: Iterable[Row],
    right_rows: Iterable[Row],
) -> Iterator[Row]:
    """
    Apply a Join node based on type of join to the given left and right rows.
    For one-to-one joins, we use a hash join with a score function to determine the best match.
    For multi-to-one joins, we use a hash join with the join keys to determine the best match.
    For nested loop joins, we use a nested loop join with the join keys to determine the best match.
    """
    spec = node.spec
    if not isinstance(spec, JoinSpec):
        raise MMDSValidationError("Join nodes require a JoinSpec.")

    predicate = spec.load_predicate()
    if spec.one_to_one:
        score = spec.load_score()
        if score is None:
            raise MMDSValidationError("Join one_to_one=True requires a score= UDF.")
        yield from one_to_one_hash_join(
            left_rows,
            right_rows,
            keys=spec.keys,
            predicate=predicate,
            score_fn=score,
            left_key=spec.left_key,
            right_key=spec.right_key,
            min_score=spec.min_score,
        )
    elif spec.keys:
        yield from hash_join(left_rows, right_rows, spec.keys, predicate)
    elif predicate is not None:
        yield from nested_loop_join(left_rows, right_rows, predicate)
    else:
        raise MMDSValidationError("Join requires on= keys and/or a UDF predicate.")
