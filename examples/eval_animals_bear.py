"""Evaluate bear presence on the full Swan Valley compilation.

The input video is the 19:35 YouTube file, not the short windows in
data/animals.jsonl. Scores one pipeline at a time against
data/animals_bear_ground_truth.json, and optionally compares detection
presence with the semantic Map:

- default — ``examples/animals_bear_detect_presence.py`` (YOLOE, then a code Map;
  no Gemini call)
- ``--compare-semantic`` — also ``examples/animals_bear_map.py``

A clip with no detection is dropped and scored as ``bear_present=false``. Precision,
recall, and F1 treat ``true`` as the positive class.

Examples::

  uv run python examples/eval_animals_bear.py
  uv run python examples/eval_animals_bear.py --compare-semantic
  uv run python examples/eval_animals_bear.py --pred-json /tmp/presence.json --semantic-json /tmp/map.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import GeminiPromptExecutor, execute  # noqa: E402
from mmds.model import (  # noqa: E402
    DatasetExpr,
    MMDSValidationError,
    PromptSpec,
    ResolvedPrompt,
    VideoMapSpec,
)

DEFAULT_PRESENCE_QUERY = ROOT / "examples" / "animals_bear_detect_presence.py"
DEFAULT_SEMANTIC_QUERY = ROOT / "examples" / "animals_bear_map.py"
DEFAULT_GROUND_TRUTH = ROOT / "data" / "animals_bear_ground_truth.json"


@dataclass(frozen=True)
class CostReport:
    label: str
    wall_time_sec: float
    prompt_calls: int
    n_rows: int
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _CountingExecutor(GeminiPromptExecutor):
    """Gemini executor that records call and token counts for one pipeline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prompt_calls = 0
        self.prompt_tokens = 0
        self.candidates_tokens = 0

    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any:
        self.prompt_calls += 1
        client = self._get_client()
        types = self._get_types()
        parts = self._build_parts(resolved_prompt.parts, client, types)
        contents = types.Content(parts=parts)
        config = self._build_config(op_type, prompt)
        response = client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        text = getattr(response, "text", None) or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        self.prompt_tokens += prompt_tokens
        self.candidates_tokens += candidate_tokens
        if not text:
            raise MMDSValidationError("Gemini returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MMDSValidationError(f"Gemini returned invalid JSON: {text!r}") from exc

    def cost_fields(self) -> dict[str, int]:
        return {
            "prompt_calls": self.prompt_calls,
            "prompt_tokens": self.prompt_tokens,
            "candidates_tokens": self.candidates_tokens,
            "total_tokens": self.prompt_tokens + self.candidates_tokens,
        }


def clip_key(start: Any, end: Any) -> tuple[float, float]:
    """Identity of one timed window inside a video."""
    return (round(float(start), 3), round(float(end), 3))


def _media_source(value: Mapping[str, Any]) -> str:
    for key in ("source", "uri", "path"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    video = value.get("video")
    if isinstance(video, Mapping):
        return _media_source(video)
    raise ValueError("A full-video row needs video.source, video.uri, or video.path.")


def row_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Match a prediction or ground-truth clip.

    Timed windows match on start/end. A full video, with no start/end, matches
    on its source URL.
    """
    video = value.get("video")
    payload = video if isinstance(video, Mapping) else value
    if "start" in payload or "end" in payload:
        return ("window", *clip_key(payload["start"], payload["end"]))
    return ("video", _media_source(payload))


def _format_key(key: tuple[Any, ...]) -> str:
    if key[0] == "window":
        return f"{key[1]:.0f}-{key[2]:.0f}s"
    return str(key[1])


def _ground_truth_key(clip: Mapping[str, Any]) -> tuple[Any, ...]:
    stored = clip.get("key")
    if isinstance(stored, tuple):
        return stored
    return row_key(clip)


def _as_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean, got {value!r}.")
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
        key = row_key(clip)
        if key in seen:
            raise ValueError(f"Duplicate ground-truth clip {key}.")
        seen.add(key)
        labels.append(
            {
                "key": key,
                "bear_present": _as_bool(clip.get("bear_present"), field="bear_present"),
            }
        )
    return labels


def evaluate_bear_presence(
    pred_rows: list[Any],
    ground_truth: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score predicted bear presence. A missing row counts as false."""
    predicted: dict[tuple[Any, ...], bool] = {}
    for row in pred_rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Prediction row must be an object, got {row!r}.")
        key = row_key(row)
        if key in predicted:
            raise ValueError(f"Duplicate prediction for clip {key}.")
        predicted[key] = _as_bool(row.get("bear_present"), field="bear_present")

    details: list[dict[str, Any]] = []
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    labeled = {_ground_truth_key(clip) for clip in ground_truth}
    for clip in ground_truth:
        key = _ground_truth_key(clip)
        truth = _as_bool(clip["bear_present"], field="bear_present")
        present = key in predicted
        pred = predicted[key] if present else False
        if truth and pred:
            kind = "tp"
        elif pred and not truth:
            kind = "fp"
        elif truth and not pred:
            kind = "fn"
        else:
            kind = "tn"
        counts[kind] += 1
        details.append(
            {
                "label": _format_key(key),
                "truth": truth,
                "predicted": pred,
                "emitted": present,
                "kind": kind,
            }
        )

    unlabeled = [
        {"label": _format_key(key), "predicted": value}
        for key, value in sorted(predicted.items(), key=lambda item: _format_key(item[0]))
        if key not in labeled
    ]
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_pred": len(predicted),
        "n_ref": len(ground_truth),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": counts["tn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "details": details,
        "unlabeled_predictions": unlabeled,
    }


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
        raise ValueError(f"Expected a JSON array in {path}.")
    return payload


def _require_detect_runtime() -> None:
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError as exc:
        venv_python = ROOT / ".venv" / "bin" / "python"
        raise SystemExit(
            "OpenCV (cv2) is not installed for this Python interpreter.\n"
            "The detect-presence pipeline needs the project virtualenv.\n\n"
            "Run:\n"
            "  uv run python examples/eval_animals_bear.py [args...]\n"
            "or:\n"
            f"  PYTHONPATH=src:. {venv_python} examples/eval_animals_bear.py [args...]\n\n"
            f"Current interpreter: {sys.executable}"
        ) from exc


def _plan_uses_prompt(plan: DatasetExpr) -> bool:
    for node in plan.walk_postorder():
        spec = node.spec
        if isinstance(spec, PromptSpec):
            return True
        if isinstance(spec, VideoMapSpec) and isinstance(spec.map_spec, PromptSpec):
            return True
    return False


def _require_prompt_key(plan: DatasetExpr) -> None:
    if not _plan_uses_prompt(plan):
        return
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    raise SystemExit("Set GEMINI_API_KEY before running a prompt-backed pipeline.")


def _run_pipeline(query_path: Path, *, label: str) -> tuple[list[Any], CostReport]:
    if query_path.name == DEFAULT_PRESENCE_QUERY.name:
        _require_detect_runtime()
    output = _load_query_output(query_path)
    _require_prompt_key(output)
    executor = _CountingExecutor()
    started = time.perf_counter()
    rows = list(execute(output, prompt_executor=executor))
    elapsed = time.perf_counter() - started
    usage = executor.cost_fields()
    cost = CostReport(
        label=label,
        wall_time_sec=elapsed,
        n_rows=len(rows),
        **usage,
    )
    return rows, cost


def _loaded_cost(label: str, rows: list[Any]) -> CostReport:
    return CostReport(
        label=label,
        wall_time_sec=0.0,
        prompt_calls=0,
        n_rows=len(rows),
    )


def _print_report(
    pred_rows: list[Any],
    ground_truth: list[Mapping[str, Any]],
    *,
    label: str,
    cost: CostReport | None = None,
) -> dict[str, Any]:
    report = evaluate_bear_presence(pred_rows, ground_truth)
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
    print(f"  precision : {report['precision']:.3f}")
    print(f"  recall    : {report['recall']:.3f}")
    print(f"  F1        : {report['f1']:.3f}")
    print(
        "  TP / FP / FN / TN : "
        f"{report['true_positives']} / {report['false_positives']} / "
        f"{report['false_negatives']} / {report['true_negatives']}"
    )

    def _section(title: str, kinds: set[str]) -> None:
        rows = [item for item in report["details"] if item["kind"] in kinds]
        print()
        print("-" * 72)
        print(f"{title} ({len(rows)})")
        print("-" * 72)
        for item in rows:
            emitted = "emitted" if item["emitted"] else "absent (scored false)"
            print(
                f"  {item['label']}  "
                f"truth={item['truth']}  predicted={item['predicted']}  {emitted}"
            )

    _section("TRUE POSITIVES", {"tp"})
    _section("FALSE POSITIVES", {"fp"})
    _section("FALSE NEGATIVES", {"fn"})
    if report["unlabeled_predictions"]:
        print()
        print("-" * 72)
        print(f"UNLABELED PREDICTIONS ({len(report['unlabeled_predictions'])})")
        print("-" * 72)
        for item in report["unlabeled_predictions"]:
            print(f"  {item['label']}  predicted={item['predicted']}")
    return report


_COMPARISON_ROWS: tuple[tuple[str, Any], ...] = (
    ("predictions", lambda report, cost: str(report["n_pred"])),
    ("true positives", lambda report, cost: str(report["true_positives"])),
    ("false positives", lambda report, cost: str(report["false_positives"])),
    ("false negatives", lambda report, cost: str(report["false_negatives"])),
    ("true negatives", lambda report, cost: str(report["true_negatives"])),
    ("precision", lambda report, cost: f"{report['precision']:.3f}"),
    ("recall", lambda report, cost: f"{report['recall']:.3f}"),
    ("F1", lambda report, cost: f"{report['f1']:.3f}"),
    ("wall time (s)", lambda report, cost: f"{cost.wall_time_sec:.3f}"),
    ("prompt calls", lambda report, cost: str(cost.prompt_calls)),
    ("prompt tokens", lambda report, cost: str(cost.prompt_tokens)),
    ("candidates tokens", lambda report, cost: str(cost.candidates_tokens)),
    ("total tokens", lambda report, cost: str(cost.total_tokens)),
)


def _print_comparison(results: list[tuple[str, dict[str, Any], CostReport]]) -> None:
    labels = [label for label, _report, _cost in results]
    metric_width = max(len(name) for name, _fmt in _COMPARISON_ROWS)
    col_width = max(16, *(len(label) for label in labels))
    print()
    print("=" * 72)
    print("COMPARISON (both pipelines vs the same ground truth)")
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
    parser.add_argument("--pred-json", type=Path, help="Load detect-presence rows instead of running it.")
    parser.add_argument(
        "--query",
        type=Path,
        default=DEFAULT_PRESENCE_QUERY,
        help="Detect-presence query file.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Ground-truth JSON.",
    )
    parser.add_argument("--save-pred", type=Path, help="Write detect-presence prediction JSON here.")
    parser.add_argument("--save-report", type=Path, help="Write metrics JSON here.")
    parser.add_argument(
        "--compare-semantic",
        action="store_true",
        help="Also run the semantic Map and compare it against the same ground truth.",
    )
    parser.add_argument(
        "--semantic-query",
        type=Path,
        default=DEFAULT_SEMANTIC_QUERY,
        help="Semantic Map query file.",
    )
    parser.add_argument(
        "--semantic-json",
        type=Path,
        help="Load semantic predictions from JSON instead of running Gemini.",
    )
    parser.add_argument("--save-semantic", type=Path, help="Write semantic prediction JSON here.")
    args = parser.parse_args()

    os.chdir(ROOT)
    ground_truth = load_ground_truth(args.ground_truth)

    if args.pred_json is not None:
        pred_rows = _load_json_array(args.pred_json)
        presence_cost = _loaded_cost(f"{args.pred_json.stem} (loaded)", pred_rows)
    else:
        pred_rows, presence_cost = _run_pipeline(args.query, label="Detect presence")

    if args.save_pred is not None:
        args.save_pred.write_text(
            json.dumps(pred_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved detect-presence predictions -> {args.save_pred}")

    semantic_rows: list[Any] | None = None
    semantic_cost: CostReport | None = None
    if args.semantic_json is not None:
        semantic_rows = _load_json_array(args.semantic_json)
        semantic_cost = _loaded_cost(f"{args.semantic_json.stem} (loaded)", semantic_rows)
    elif args.compare_semantic:
        semantic_rows, semantic_cost = _run_pipeline(
            args.semantic_query, label="Semantic map"
        )
        if args.save_semantic is not None:
            args.save_semantic.write_text(
                json.dumps(semantic_rows, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Saved semantic predictions -> {args.save_semantic}")

    results: list[tuple[str, dict[str, Any], CostReport]] = []
    presence_report = _print_report(
        pred_rows,
        ground_truth,
        label="Detect presence",
        cost=presence_cost,
    )
    results.append(("Detect presence", presence_report, presence_cost))
    if semantic_rows is not None and semantic_cost is not None:
        print()
        semantic_report = _print_report(
            semantic_rows,
            ground_truth,
            label="Semantic map",
            cost=semantic_cost,
        )
        results.append(("Semantic map", semantic_report, semantic_cost))
        _print_comparison(results)

    if args.save_report is not None:
        payload = {
            label: {"accuracy": report, "cost": cost.to_dict()}
            for label, report, cost in results
        }
        args.save_report.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved metrics report -> {args.save_report}")


if __name__ == "__main__":
    main()
