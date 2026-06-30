from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..model import MMDSValidationError, Row

# Hash key for cross-camera vehicle appearance matching
VEHICLE_APPEARANCE_KEYS: tuple[str, ...] = ("vehicle_class", "color", "subtype")

JoinPredicate = Callable[[Row, Row], bool]
JoinScore = Callable[[Row, Row], float]


def vehicle_appearance_hash_key(row: Row) -> tuple[Any, ...] | None:
    """Hash bucket key for cross-camera vehicle appearance matching."""
    return join_hash_key(row, VEHICLE_APPEARANCE_KEYS)


def join_hash_key(row: Row, keys: tuple[str, ...]) -> tuple[Any, ...] | None:
    """Build a hashable join key from any ``keys`` present on ``row``."""
    values: list[Any] = []
    for key in keys:
        if key not in row:
            return None
        values.append(row[key])
    try:
        hash(tuple(values))
    except TypeError:
        return None
    return tuple(values)


def build_hash_index(
    rows: Iterable[Row],
    keys: tuple[str, ...],
) -> dict[tuple[Any, ...], list[Row]]:
    """Index rows by ``keys``; rows with missing or unhashable keys are skipped."""
    # e.g. keys = ("sedan", "white", "compact") -> rows = [track_a, track_b, ...]
    index: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for row in rows:
        key = join_hash_key(row, keys)
        if key is not None:
            index[key].append(row)
    return index


def emit_join_pair(
    left: Row,
    right: Row,
    *,
    match_score: float | None = None,
) -> Row:
    """Emit a join pair with the left and right rows and optional match score."""
    output: Row = {"left": dict(left), "right": dict(right)}
    if match_score is not None:
        output["match_score"] = match_score
    return output

# Executor picks the appropriate join function based on the join type
# - hash_join: when keys are provided and many-to-many join is needed
# - nested_loop_join: when keys are not provided and predicate is provided
# - one_to_one_hash_join: when keys are provided and one_to_one

def hash_join(
    left_rows: Iterable[Row],
    right_rows: Iterable[Row],
    keys: tuple[str, ...],
    predicate: JoinPredicate | None = None,
) -> Iterator[Row]:
    """
    Probe a hash index built on ``right_rows`` with each ``left_rows`` key.

    Only rows sharing the same ``keys`` tuple are compared.
    Complexity is ``O(|left| + |right|)``
    """
    right_list = list(right_rows)
    index = build_hash_index(right_list, keys)

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
    """
    Compare every left/right pair with ``predicate``.
    Complexity is ``O(|left| * |right|)``
    """
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
    """
    Greedy one-to-one join over hash buckets.
    Complexity is O(|left| + |right| + n log n), where n is the number of candidate pairs in all matching buckets.
    
    Candidate pairs inside each bucket must pass ``predicate`` (when set) and
    ``score_fn``. Pairs are sorted by descending score; each ``left_key`` and
    ``right_key`` identity is used at most once and only highest scoring pair is kept.
    """
    left_list = list(left_rows)
    right_list = list(right_rows)
    candidates: list[tuple[float, Row, Row]] = []

    if keys:
        # Build hash index on the right rows
        index = build_hash_index(right_list, keys)
        for left in left_list:
            # Probe the hash index with the left row
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
                )
    else:
        # No keys provided, nested loop join is needed
        if predicate is None:
            raise MMDSValidationError(
                "one_to_one_hash_join without keys requires a predicate."
            )
        for left in left_list:
            for right in right_list:
                # Add candidate pair if it passes the predicate and score threshold
                _maybe_add_candidate(
                    candidates,
                    left,
                    right,
                    predicate,
                    score_fn,
                    min_score,
                )

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_left: set[tuple[Any, ...]] = set()
    used_right: set[tuple[Any, ...]] = set()
    for match_score, left, right in candidates:
        # Create identity keys for the left and right rows
        left_identity = join_hash_key(left, left_key)
        right_identity = join_hash_key(right, right_key)
        if left_identity is None or right_identity is None:
            continue
        if left_identity in used_left or right_identity in used_right:
            continue
        # Mark the left and right rows as used (skip duplicates)
        used_left.add(left_identity)
        used_right.add(right_identity)
        # Emit the join pair with the match score
        yield emit_join_pair(left, right, match_score=match_score)


def predicate_matches(predicate: JoinPredicate, left: Row, right: Row) -> bool:
    """Check if the predicate matches the left and right rows."""
    try:
        return bool(predicate(left, right))
    except TypeError as exc:
        raise MMDSValidationError(
            "Join predicates must be binary callables: predicate(left, right)."
        ) from exc


def _maybe_add_candidate(
    candidates: list[tuple[float, Row, Row]],
    left: Row,
    right: Row,
    predicate: JoinPredicate | None,
    score_fn: JoinScore,
    min_score: float | None,
) -> None:
    """If the predicate matches the left and right rows, add the candidate score, left, and right rows to the list."""
    if predicate is not None and not predicate_matches(predicate, left, right):
        return
    try:
        raw_score = score_fn(left, right)
    except TypeError as exc:
        raise MMDSValidationError(
            "Join score UDFs must be binary callables: score(left, right) -> number."
        ) from exc
    if not isinstance(raw_score, (int, float)):
        return
    score = float(raw_score)
    if min_score is not None and score < float(min_score):
        return
    candidates.append((score, dict(left), dict(right)))
