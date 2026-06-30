from __future__ import annotations

from typing import Any

from mmds.join.predicates import same_vehicle as same_vehicle_predicate


def same_vehicle(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Join/Filter predicate: cross-camera track pair passes all relational checks."""
    return same_vehicle_predicate(left, right)


def vehicle_match_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Join score UDF: mean track confidence for greedy one-to-one matching."""
    left_conf = float(left.get("confidence", 0.0))
    right_conf = float(right.get("confidence", 0.0))
    return (left_conf + right_conf) / 2.0
