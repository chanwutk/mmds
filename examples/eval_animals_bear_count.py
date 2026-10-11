"""Evaluate the exact bear count on the full Swan Valley compilation.

Scores ``examples/animals_bear_count.py`` (YOLOE tracks, then a code Map; no
Gemini call) against ``data/animals_bear_count_ground_truth.json``.
``--compare-semantic`` also runs ``examples/animals_bear_count_map.py``.

Each labeled clip is scored on its integer ``bear_count``: exact match, signed
error (predicted minus truth), mean absolute error, and relative error (total
absolute error over the total true count). A clip with no prediction row is
scored as ``bear_count=0``.

Examples::

  uv run python examples/eval_animals_bear_count.py
  uv run python examples/eval_animals_bear_count.py --compare-semantic
  uv run python examples/eval_animals_bear_count.py --pred-json /tmp/count.json --semantic-json /tmp/map.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

# Wrapper evaluators (e.g. eval_vehicle_count.py) override these before main().
COUNT_FIELD = "bear_count"
DETECT_LABEL = "Detect count"
DESCRIPTION = __doc__
DEFAULT_COUNT_QUERY = _ROOT / "examples" / "animals_bear_count.py"
DEFAULT_SEMANTIC_QUERY = _ROOT / "examples" / "animals_bear_count_map.py"
DEFAULT_GROUND_TRUTH = _ROOT / "data" / "animals_bear_count_ground_truth.json"


def _load_bear_eval() -> ModuleType:
    path = Path(__file__).resolve().parent / "eval_animals_bear.py"
    spec = importlib.util.spec_from_file_location("eval_animals_bear", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_bear_eval = _load_bear_eval()


def as_count(value: Any, *, field: str) -> int:
    """Accept a non-negative int, or a float with no fractional part."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    return value


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips") if isinstance(payload, Mapping) else payload
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"{path} must contain a non-empty 'clips' list.")
    labels: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise ValueError(f"Ground-truth clip must be an object, got {clip!r}.")
        key = _bear_eval.row_key(clip)
        if key in seen:
            raise ValueError(f"Duplicate ground-truth clip {key}.")
        seen.add(key)
        labels.append(
            {"key": key, COUNT_FIELD: as_count(clip.get(COUNT_FIELD), field=COUNT_FIELD)}
        )
    return labels


def evaluate_bear_count(
    pred_rows: list[Any],
    ground_truth: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score predicted counts per clip. A missing row counts as 0."""
    predicted: dict[tuple[Any, ...], int] = {}
    for row in pred_rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Prediction row must be an object, got {row!r}.")
        key = _bear_eval.row_key(row)
        if key in predicted:
            raise ValueError(f"Duplicate prediction for clip {key}.")
        predicted[key] = as_count(row.get(COUNT_FIELD), field=COUNT_FIELD)

    labeled = {_bear_eval._ground_truth_key(clip) for clip in ground_truth}
    _bear_eval.check_prediction_coverage(set(predicted), labeled)

    details: list[dict[str, Any]] = []
    kinds = {"exact": 0, "over": 0, "under": 0}
    total_abs_error = 0
    total_truth = 0
    for clip in ground_truth:
        key = _bear_eval._ground_truth_key(clip)
        truth = as_count(clip[COUNT_FIELD], field=COUNT_FIELD)
        emitted = key in predicted
        pred = predicted[key] if emitted else 0
        if pred == truth:
            kind = "exact"
        elif pred > truth:
            kind = "over"
        else:
            kind = "under"
        kinds[kind] += 1
        total_abs_error += abs(pred - truth)
        total_truth += truth
        details.append(
            {
                "label": _bear_eval._format_key(key),
                "truth": truth,
                "predicted": pred,
                "error": pred - truth,
                "emitted": emitted,
                "kind": kind,
            }
        )

    unlabeled = [
        {"label": _bear_eval._format_key(key), "predicted": value}
        for key, value in sorted(
            predicted.items(), key=lambda item: _bear_eval._format_key(item[0])
        )
        if key not in labeled
    ]
    n_ref = len(ground_truth)
    return {
        "n_pred": len(predicted),
        "n_ref": n_ref,
        "exact_matches": kinds["exact"],
        "over_counts": kinds["over"],
        "under_counts": kinds["under"],
        "accuracy": kinds["exact"] / n_ref if n_ref else 0.0,
        "total_truth": total_truth,
        "total_abs_error": total_abs_error,
        "mae": total_abs_error / n_ref if n_ref else 0.0,
        # None when every label is 0, since any error is then unbounded.
        "relative_error": total_abs_error / total_truth if total_truth else None,
        "details": details,
        "unlabeled_predictions": unlabeled,
    }


def _format_relative(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_report(
    pred_rows: list[Any],
    ground_truth: list[Mapping[str, Any]],
    *,
    label: str,
    cost: Any = None,
) -> dict[str, Any]:
    report = evaluate_bear_count(pred_rows, ground_truth)
    print("=" * 72)
    print(f"{label.upper()} — GROUND-TRUTH EVALUATION")
    print("=" * 72)
    if cost is not None:
        print(
            f"wall time: {cost.wall_time_sec:.3f}s   "
            f"prompt calls: {cost.prompt_calls}   "
            f"total tokens: {cost.total_tokens}"
        )
    print(f"predictions: {report['n_pred']}   ground-truth clips: {report['n_ref']}")
    print()
    print(f"  exact-match accuracy : {report['accuracy']:.3f}")
    print(f"  mean absolute error  : {report['mae']:.3f}")
    print(f"  relative error       : {_format_relative(report['relative_error'])}")
    print(
        "  exact / over / under : "
        f"{report['exact_matches']} / {report['over_counts']} / "
        f"{report['under_counts']}"
    )
    print()
    print("-" * 72)
    print(f"PER CLIP ({len(report['details'])})")
    print("-" * 72)
    for item in report["details"]:
        emitted = "" if item["emitted"] else "  absent (scored 0)"
        print(
            f"  {item['label']}  truth={item['truth']}  "
            f"predicted={item['predicted']}  error={item['error']:+d}  "
            f"{item['kind']}{emitted}"
        )
    if report["unlabeled_predictions"]:
        print()
        print("-" * 72)
        print(f"UNLABELED PREDICTIONS ({len(report['unlabeled_predictions'])})")
        print("-" * 72)
        for item in report["unlabeled_predictions"]:
            print(f"  {item['label']}  predicted={item['predicted']}")
    return report


def _predicted_total(report: Mapping[str, Any]) -> str:
    return str(sum(item["predicted"] for item in report["details"]))


COMPARISON_ROWS: tuple[tuple[str, Any], ...] = (
    ("predictions", lambda report, cost: str(report["n_pred"])),
    ("true count", lambda report, cost: str(report["total_truth"])),
    ("predicted count", lambda report, cost: _predicted_total(report)),
    ("exact matches", lambda report, cost: str(report["exact_matches"])),
    ("over-counted clips", lambda report, cost: str(report["over_counts"])),
    ("under-counted clips", lambda report, cost: str(report["under_counts"])),
    ("MAE", lambda report, cost: f"{report['mae']:.3f}"),
    ("relative error", lambda report, cost: _format_relative(report["relative_error"])),
    ("wall time (s)", lambda report, cost: f"{cost.wall_time_sec:.3f}"),
    ("prompt calls", lambda report, cost: str(cost.prompt_calls)),
    ("prompt tokens", lambda report, cost: str(cost.prompt_tokens)),
    ("candidates tokens", lambda report, cost: str(cost.candidates_tokens)),
    ("total tokens", lambda report, cost: str(cost.total_tokens)),
)


def main() -> None:
    _bear_eval.FLAG_FIELD = COUNT_FIELD
    _bear_eval.PRESENCE_LABEL = DETECT_LABEL
    _bear_eval.DESCRIPTION = DESCRIPTION
    _bear_eval.DEFAULT_PRESENCE_QUERY = DEFAULT_COUNT_QUERY
    _bear_eval.DEFAULT_SEMANTIC_QUERY = DEFAULT_SEMANTIC_QUERY
    _bear_eval.DEFAULT_GROUND_TRUTH = DEFAULT_GROUND_TRUTH
    _bear_eval.load_ground_truth = load_ground_truth
    _bear_eval._print_report = print_report
    _bear_eval._COMPARISON_ROWS = COMPARISON_ROWS
    _bear_eval.main()


if __name__ == "__main__":
    main()
