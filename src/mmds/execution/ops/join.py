from __future__ import annotations

from collections.abc import Iterable

from ...join.hash_join import hash_join, nested_loop_join, one_to_one_hash_join
from ...model import DatasetExpr, JoinSpec, MMDSValidationError, Row


def _apply_join(
    node: DatasetExpr,
    left_rows: Iterable[Row],
    right_rows: Iterable[Row],
) -> Iterable[Row]:
    """
    Executes a join operation between two sets of rows according to the
    join specification (`JoinSpec`) attached to the provided `DatasetExpr` node.

    Supported join types:
      - Hash Join: Uses one or more keys (`keys`) to efficiently join left and right
        rows. Used for many-to-many joins where join keys are available.
      - One-to-One Hash Join: When `one_to_one=True` is set, performs a greedy 
        one-to-one join between left and right using keys and a required scoring
        function (`score`). Only the best scored pairs (above, or equal to, an 
        optional `min_score`) are joined, ensuring that each left/right pair is
        matched at most once.
      - Nested Loop Join: Performs a cartesian join, testing all pairs using the
        provided predicate, when no keys are provided (fallback, more expensive).

    Raises:
        MMDSValidationError: If the node does not have a `JoinSpec`, or if a required 
        predicate, keys, or score function is missing.

    Arguments:
        node: DatasetExpr
            The join operator node specifying the join parameters.
        left_rows: Iterable[Row]
            Iterable of input rows from the left dataset.
        right_rows: Iterable[Row]
            Iterable of input rows from the right dataset.

    Yields:
        Row: Joined rows as specified by the type of join.
    """
    spec = node.spec
    if not isinstance(spec, JoinSpec):
        raise MMDSValidationError("Join nodes require a JoinSpec.")

    predicate = spec.load_predicate()
    right_list = list(right_rows)
    left_list = list(left_rows)

    if spec.one_to_one:
        score_fn = spec.load_score()
        if score_fn is None:
            raise MMDSValidationError("Join one_to_one=True requires a score= UDF.")
        yield from one_to_one_hash_join(
            left_list,
            right_list,
            keys=spec.keys,
            predicate=predicate,
            score_fn=score_fn,
            left_key=spec.left_key,
            right_key=spec.right_key,
            min_score=spec.min_score,
        )
        return

    if spec.keys:
        yield from hash_join(left_list, right_list, spec.keys, predicate)
        return

    if predicate is None:
        raise MMDSValidationError("Join requires on= keys and/or a UDF predicate.")

    yield from nested_loop_join(left_list, right_list, predicate)
