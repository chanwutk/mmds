"""Evaluate the cross-camera vehicle join against hand-labeled ground truth.

Runs ``examples/join_cross_camera_vehicle.py`` (or loads a saved prediction
dump), scores its cross-camera trajectories against
``data/i24v_traffic_highway2_highway3_5s_ground_truth.json`` using the shared
:func:`mmds.join.eval_trajectories.evaluate_trajectories`, and prints
precision / recall / F1, attribute agreement, and per-match detail — including
the specific false positives and false negatives — so join-quality changes can
be measured before/after. Each run also reports its wall time and (for
prompt-backed pipelines) Gemini token usage.

With ``--compare-semantic`` it additionally runs
``examples/semantic_join_cross_camera_vehicle.py`` (requires Gemini), scores it
against the **same** ground truth, and prints a side-by-side comparison of the
accuracy metrics and cost (wall time / prompt calls / tokens).

Examples::

  # Run the UDF pipeline and evaluate against ground truth
  PYTHONPATH=src:. python examples/eval_cross_camera_join.py

  # Evaluate a previously saved prediction dump (no Detect/tracking run)
  PYTHONPATH=src:. python examples/eval_cross_camera_join.py \\
      --pred-json /tmp/join_out.json

  # Compare the UDF join to the semantic (Gemini) join, both vs ground truth
  PYTHONPATH=src:. python examples/eval_cross_camera_join.py --compare-semantic

  # Persist the prediction dump and the metrics report
  PYTHONPATH=src:. python examples/eval_cross_camera_join.py \\
      --save-pred /tmp/baseline_pred.json --save-report /tmp/baseline_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import GeminiPromptExecutor, execute  # noqa: E402
from mmds.join.eval_trajectories import (  # noqa: E402
    CostReport,
    evaluate_trajectories,
    normalize_trajectory_rows,
)

DEFAULT_QUERY = ROOT / "examples" / "join_cross_camera_vehicle.py"
DEFAULT_SEMANTIC_QUERY = (
    ROOT / "examples" / "semantic_join_cross_camera_vehicle.py"
)
DEFAULT_GROUND_TRUTH = (
    ROOT / "data" / "i24v_traffic_highway2_highway3_5s_ground_truth.json"
)


def _load_query_output(query_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(query_path.stem, query_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load query module from {str(query_path)!r}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "output"):
        raise RuntimeError(f"{query_path} does not define an 'output' expression.")
    return module.output


def _load_json_array(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return payload


def _run_pipeline(
    query_path: Path,
    *,
    label: str,
    use_gemini: bool,
) -> tuple[list[Any], CostReport]:
    """Execute a query's ``output`` expression, timing it and metering tokens.

    ``use_gemini`` constructs a :class:`GeminiPromptExecutor` (prompt-backed
    pipelines such as the semantic join); pure-UDF pipelines pass ``None`` and
    report 0 prompt calls / tokens.
    """
    output = _load_query_output(query_path)
    prompt_executor = GeminiPromptExecutor() if use_gemini else None
    if prompt_executor is not None:
        prompt_executor.reset_usage()
    started = time.perf_counter()
    rows = list(execute(output, prompt_executor=prompt_executor))
    elapsed = time.perf_counter() - started
    usage = prompt_executor.usage_snapshot() if prompt_executor is not None else {}
    cost = CostReport(
        label=label,
        wall_time_sec=elapsed,
        prompt_calls=usage.get("prompt_calls", 0),
        n_trajectories=len(normalize_trajectory_rows(rows)),
        prompt_tokens=usage.get("prompt_tokens", 0),
        candidates_tokens=usage.get("candidates_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )
    return rows, cost


def _loaded_cost(label: str, rows: list[Any]) -> CostReport:
    """Cost record for predictions loaded from disk (no execution)."""
    return CostReport(
        label=label,
        wall_time_sec=0.0,
        prompt_calls=0,
        n_trajectories=len(normalize_trajectory_rows(rows)),
    )


def _fmt_attrs(traj: dict[str, Any]) -> str:
    attrs = traj.get("attributes", {})
    return (
        f"class={attrs.get('class', '')!r} "
        f"color={attrs.get('color', '')!r} "
        f"subtype={attrs.get('subtype', '')!r}"
    )


def _fmt_timeline(traj: dict[str, Any]) -> str:
    segs = traj.get("timeline", []) or []
    parts = [
        f"{s.get('camera_id', '?')}[{s.get('entered', '?')}..{s.get('exited', '?')}]"
        for s in segs
        if isinstance(s, dict)
    ]
    return ", ".join(parts) if parts else "(no timeline)"


def _print_report(
    pred_rows: list[Any],
    ref_rows: list[Any],
    min_score: float,
    *,
    label: str = "UDF join",
    cost: CostReport | None = None,
    max_endpoint_delta_sec: float | None = 1.0,
) -> dict:
    preds = normalize_trajectory_rows(pred_rows)
    refs = normalize_trajectory_rows(ref_rows)
    report = evaluate_trajectories(
        pred_rows,
        ref_rows,
        min_score=min_score,
        max_endpoint_delta_sec=max_endpoint_delta_sec,
    )

    matched_pred = {m.pred_index for m in report.matches}
    matched_ref = {m.ref_index for m in report.matches}

    print("=" * 72)
    print(f"{label.upper()} — GROUND-TRUTH EVALUATION")
    print("=" * 72)
    if cost is not None:
        print(
            f"wall time: {cost.wall_time_sec:.3f}s   "
            f"prompt calls: {cost.prompt_calls}   "
            f"total tokens: {cost.total_tokens}"
        )
    print(f"min_score (match acceptance): {min_score}")
    delta_label = (
        "disabled"
        if max_endpoint_delta_sec is None
        else f"|Δentered|≤{max_endpoint_delta_sec}s OR |Δexited|≤{max_endpoint_delta_sec}s"
    )
    print(f"endpoint gate (per shared camera): {delta_label}")
    print(f"predictions: {report.n_pred}   ground-truth: {report.n_ref}")
    print()
    print(f"  precision : {report.precision:.3f}")
    print(f"  recall    : {report.recall:.3f}")
    print(f"  F1        : {report.f1:.3f}")
    print(f"  TP / FP / FN : {report.true_positives} / "
          f"{report.false_positives} / {report.false_negatives}")
    print()
    print("  attribute agreement (over matched pairs):")
    print(f"    mean timeline IoU   : {report.mean_timeline_score:.3f}")
    print(f"    mean attribute exact: {report.mean_attribute_exact:.3f}")
    print(f"    mean attribute soft : {report.mean_attribute_soft:.3f}")

    print()
    print("-" * 72)
    print(f"TRUE POSITIVES ({len(report.matches)})")
    print("-" * 72)
    for m in sorted(report.matches, key=lambda x: -x.score):
        pred = preds[m.pred_index]
        ref = refs[m.ref_index]
        print(f"  score={m.score:.3f} "
              f"(timeline={m.timeline_score:.3f}, "
              f"attr_exact={m.attribute_exact:.3f}, attr_soft={m.attribute_soft:.3f})")
        print(f"    pred  {pred.get('vehicle_id', '')!r}: {_fmt_attrs(pred)}")
        print(f"          timeline: {_fmt_timeline(pred)}")
        print(f"    truth {ref.get('vehicle_id', '')!r}: {_fmt_attrs(ref)}")
        print(f"          timeline: {_fmt_timeline(ref)}")

    print()
    print("-" * 72)
    fps = [i for i in range(len(preds)) if i not in matched_pred]
    print(f"FALSE POSITIVES ({len(fps)}) — predicted trajectories with no GT match")
    print("-" * 72)
    for i in fps:
        pred = preds[i]
        print(f"  pred {pred.get('vehicle_id', '')!r}: {_fmt_attrs(pred)}")
        print(f"       timeline: {_fmt_timeline(pred)}")

    print()
    print("-" * 72)
    fns = [i for i in range(len(refs)) if i not in matched_ref]
    print(f"FALSE NEGATIVES ({len(fns)}) — GT trajectories not recovered")
    print("-" * 72)
    for i in fns:
        ref = refs[i]
        n_cams = len({s.get("camera_id") for s in (ref.get("timeline") or [])})
        note = " [single-camera GT: a cross-camera join cannot match this]" if n_cams < 2 else ""
        print(f"  truth {ref.get('vehicle_id', '')!r}: {_fmt_attrs(ref)}{note}")
        print(f"        timeline: {_fmt_timeline(ref)}")

    return report.to_dict()


# (row label, value extractor) pairs for the side-by-side comparison table.
_COMPARISON_ROWS: tuple[tuple[str, Any], ...] = (
    ("predictions", lambda r, c: str(r["n_pred"])),
    ("true positives", lambda r, c: str(r["true_positives"])),
    ("false positives", lambda r, c: str(r["false_positives"])),
    ("false negatives", lambda r, c: str(r["false_negatives"])),
    ("precision", lambda r, c: f"{r['precision']:.3f}"),
    ("recall", lambda r, c: f"{r['recall']:.3f}"),
    ("F1", lambda r, c: f"{r['f1']:.3f}"),
    ("mean timeline IoU", lambda r, c: f"{r['mean_timeline_score']:.3f}"),
    ("mean attr exact", lambda r, c: f"{r['mean_attribute_exact']:.3f}"),
    ("mean attr soft", lambda r, c: f"{r['mean_attribute_soft']:.3f}"),
    ("wall time (s)", lambda r, c: f"{c.wall_time_sec:.3f}"),
    ("prompt calls", lambda r, c: str(c.prompt_calls)),
    ("prompt tokens", lambda r, c: str(c.prompt_tokens)),
    ("candidates tokens", lambda r, c: str(c.candidates_tokens)),
    ("total tokens", lambda r, c: str(c.total_tokens)),
)


def _print_comparison(results: list[tuple[str, dict, CostReport]]) -> None:
    """Print a metric x pipeline table, both scored against the same GT."""
    labels = [label for label, _report, _cost in results]
    metric_width = max(len(name) for name, _fmt in _COMPARISON_ROWS)
    col_width = max(14, *(len(label) for label in labels))

    print()
    print("=" * 72)
    print("COMPARISON (all pipelines vs the same ground truth)")
    print("=" * 72)
    header = f"{'metric':<{metric_width}}" + "".join(
        f"  {label:>{col_width}}" for label in labels
    )
    print(header)
    print("-" * len(header))
    for name, fmt in _COMPARISON_ROWS:
        line = f"{name:<{metric_width}}"
        for _label, report, cost in results:
            line += f"  {fmt(report, cost):>{col_width}}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-json",
        type=Path,
        help="Load predictions from JSON instead of running the pipeline.",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=DEFAULT_QUERY,
        help="Prediction pipeline query file.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Ground-truth trajectory JSON.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Minimum pair score to count as a true positive (default: 0.25).",
    )
    parser.add_argument(
        "--max-endpoint-delta",
        type=float,
        default=1.0,
        help=(
            "Per shared camera, require |Δentered|≤δ OR |Δexited|≤δ seconds "
            "before interval IoU contributes (default: 1.0). "
            "Pass a negative value to disable the gate."
        ),
    )
    parser.add_argument("--save-pred", type=Path, help="Write UDF prediction JSON here.")
    parser.add_argument("--save-report", type=Path, help="Write metrics JSON here.")
    parser.add_argument(
        "--compare-semantic",
        action="store_true",
        help="Also run the semantic (Gemini) join and compare it vs the same GT.",
    )
    parser.add_argument(
        "--semantic-query",
        type=Path,
        default=DEFAULT_SEMANTIC_QUERY,
        help="Semantic pipeline query file (default: semantic_join_cross_camera_vehicle.py).",
    )
    parser.add_argument(
        "--semantic-json",
        type=Path,
        help="Load semantic predictions from JSON instead of running Gemini (implies compare).",
    )
    parser.add_argument(
        "--save-semantic",
        type=Path,
        help="Write semantic prediction JSON here.",
    )
    args = parser.parse_args()

    # Resolve relative video / data paths against the repo root regardless of
    # the caller's working directory (matches compare_cross_camera_joins.py).
    os.chdir(ROOT)

    ref_rows = _load_json_array(args.ground_truth)

    if args.pred_json is not None:
        pred_rows = _load_json_array(args.pred_json)
        udf_cost = _loaded_cost(f"{args.pred_json.stem} (loaded)", pred_rows)
    else:
        pred_rows, udf_cost = _run_pipeline(
            args.query, label="UDF join", use_gemini=False
        )

    if args.save_pred is not None:
        args.save_pred.write_text(
            json.dumps(pred_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Optionally bring in the semantic pipeline as a second predictor.
    semantic_rows: list[Any] | None = None
    semantic_cost: CostReport | None = None
    if args.semantic_json is not None:
        semantic_rows = _load_json_array(args.semantic_json)
        semantic_cost = _loaded_cost(f"{args.semantic_json.stem} (loaded)", semantic_rows)
    elif args.compare_semantic:
        semantic_rows, semantic_cost = _run_pipeline(
            args.semantic_query, label="Semantic join", use_gemini=True
        )
        if args.save_semantic is not None:
            args.save_semantic.write_text(
                json.dumps(semantic_rows, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    results: list[tuple[str, dict, CostReport]] = []
    endpoint_delta = (
        None if args.max_endpoint_delta < 0 else float(args.max_endpoint_delta)
    )
    udf_report = _print_report(
        pred_rows,
        ref_rows,
        args.min_score,
        label="UDF join",
        cost=udf_cost,
        max_endpoint_delta_sec=endpoint_delta,
    )
    results.append(("UDF join", udf_report, udf_cost))

    if semantic_rows is not None and semantic_cost is not None:
        print()
        semantic_report = _print_report(
            semantic_rows,
            ref_rows,
            args.min_score,
            label="Semantic join",
            cost=semantic_cost,
            max_endpoint_delta_sec=endpoint_delta,
        )
        results.append(("Semantic join", semantic_report, semantic_cost))
        _print_comparison(results)

    if args.save_report is not None:
        report_payload = {
            label: {"accuracy": report, "cost": cost.to_dict()}
            for label, report, cost in results
        }
        args.save_report.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved metrics report -> {args.save_report}")


if __name__ == "__main__":
    main()
