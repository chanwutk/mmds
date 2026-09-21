from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from numbers import Real
from typing import Any

from ..model import MMDSValidationError, Row

JoinPredicate = Callable[[Row, Row], bool]
JoinScore = Callable[[Row, Row], float]


def join_hash_key(row: Row, keys: tuple[str, ...]) -> tuple[Any, ...] | None:
    """Return a hashable key, or None for missing/unhashable fields."""
    values: list[Any] = []
    for key in keys:
        if key not in row:
            return None
        values.append(row[key])
    result = tuple(values)
    try:
        hash(result)
    except TypeError:
        return None
    return result


def build_hash_index(
    rows: Iterable[Row],
    keys: tuple[str, ...],
) -> dict[tuple[Any, ...], list[Row]]:
    index: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for row in rows:
        key = join_hash_key(row, keys)
        if key is not None:
            index[key].append(row)
    return dict(index)


def emit_join_pair(
    left: Row,
    right: Row,
    *,
    match_score: float | None = None,
) -> Row:
    output: Row = {"left": dict(left), "right": dict(right)}
    if match_score is not None:
        output["match_score"] = match_score
    return output


def hash_join(
    left_rows: Iterable[Row],
    right_rows: Iterable[Row],
    keys: tuple[str, ...],
    predicate: JoinPredicate | None = None,
) -> Iterator[Row]:
    index = build_hash_index(right_rows, keys)
    for left in left_rows:
        key = join_hash_key(left, keys)
        if key is None:
            continue
        for right in index.get(key, ()):
            if predicate is None or predicate_matches(predicate, left, right):
                yield emit_join_pair(left, right)


def nested_loop_join(
    left_rows: Iterable[Row],
    right_rows: Iterable[Row],
    predicate: JoinPredicate,
) -> Iterator[Row]:
    right_list = list(right_rows)
    for left in left_rows:
        for right in right_list:
            if predicate_matches(predicate, left, right):
                yield emit_join_pair(left, right)


def one_to_one_hash_join(
    left_rows: Iterable[Row],
    right_rows: Iterable[Row],
    *,
    keys: tuple[str, ...],
    predicate: JoinPredicate | None,
    score_fn: JoinScore,
    left_key: tuple[str, ...],
    right_key: tuple[str, ...],
    min_score: float | None = None,
) -> Iterator[Row]:
    """Greedily select highest-scoring pairs with unique side identities."""
    left_list = list(left_rows)
    right_list = list(right_rows)
    candidates: list[tuple[float, Row, Row, tuple[Any, ...], tuple[Any, ...]]] = []

    if keys:
        index = build_hash_index(right_list, keys)
        for left in left_list:
            key = join_hash_key(left, keys)
            if key is None:
                continue
            for right in index.get(key, ()):
                _maybe_add_candidate(
                    candidates,
                    left,
                    right,
                    predicate,
                    score_fn,
                    min_score,
                    left_key,
                    right_key,
                )
    else:
        if predicate is None:
            raise MMDSValidationError(
                "One-to-one Join without on= keys requires a predicate."
            )
        for left in left_list:
            for right in right_list:
                _maybe_add_candidate(
                    candidates,
                    left,
                    right,
                    predicate,
                    score_fn,
                    min_score,
                    left_key,
                    right_key,
                )

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    used_left: set[tuple[Any, ...]] = set()
    used_right: set[tuple[Any, ...]] = set()
    for match_score, left, right, left_identity, right_identity in candidates:
        if left_identity in used_left or right_identity in used_right:
            continue
        used_left.add(left_identity)
        used_right.add(right_identity)
        yield emit_join_pair(left, right, match_score=match_score)


def predicate_matches(predicate: JoinPredicate, left: Row, right: Row) -> bool:
    try:
        return bool(predicate(left, right))
    except TypeError as exc:
        raise MMDSValidationError(
            "Join predicates must be binary callables: predicate(left, right)."
        ) from exc


def _maybe_add_candidate(
    candidates: list[tuple[float, Row, Row, tuple[Any, ...], tuple[Any, ...]]],
    left: Row,
    right: Row,
    predicate: JoinPredicate | None,
    score_fn: JoinScore,
    min_score: float | None,
    left_key: tuple[str, ...],
    right_key: tuple[str, ...],
) -> None:
    left_identity = join_hash_key(left, left_key)
    right_identity = join_hash_key(right, right_key)
    if left_identity is None or right_identity is None:
        return
    if predicate is not None and not predicate_matches(predicate, left, right):
        return
    try:
        raw_score = score_fn(left, right)
    except TypeError as exc:
        raise MMDSValidationError(
            "Join score UDFs must be binary callables: score(left, right) -> number."
        ) from exc
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, Real)
        or not math.isfinite(float(raw_score))
    ):
        raise MMDSValidationError(
            "Join score UDFs must return a finite real number."
        )
    score = float(raw_score)
    if min_score is not None and score < min_score:
        return
    candidates.append(
        (score, dict(left), dict(right), left_identity, right_identity)
    )
