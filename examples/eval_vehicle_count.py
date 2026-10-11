"""Evaluate the exact vehicle count on the first 5 seconds of I24V highway2.

Scores ``examples/vehicle_count.py`` (YOLOE tracks, then a code Map; no Gemini
call) against ``data/vehicle_count_ground_truth.json`` (20 vehicles).
``--compare-semantic`` also runs ``examples/vehicle_count_map.py``.

Scoring is the same as examples/eval_animals_bear_count.py: exact match,
signed error, mean absolute error, relative error, and cost. A clip with no
prediction row is scored as ``vehicle_count=0``.

Needs data/i24v_traffic/videos/highway2.mp4 (see data/i24v_traffic/README.md).

Examples::

  uv run python examples/eval_vehicle_count.py
  uv run python examples/eval_vehicle_count.py --compare-semantic
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_count_eval() -> ModuleType:
    path = Path(__file__).resolve().parent / "eval_animals_bear_count.py"
    spec = importlib.util.spec_from_file_location("eval_animals_bear_count", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(evaluator: ModuleType) -> ModuleType:
    """Point the shared count evaluator at the vehicle example."""
    root = Path(__file__).resolve().parents[1]
    evaluator.COUNT_FIELD = "vehicle_count"
    evaluator.DETECT_LABEL = "Detect count"
    evaluator.DESCRIPTION = __doc__
    evaluator.DEFAULT_COUNT_QUERY = root / "examples" / "vehicle_count.py"
    evaluator.DEFAULT_SEMANTIC_QUERY = root / "examples" / "vehicle_count_map.py"
    evaluator.DEFAULT_GROUND_TRUTH = root / "data" / "vehicle_count_ground_truth.json"
    return evaluator


def main() -> None:
    configure(_load_count_eval()).main()


if __name__ == "__main__":
    main()
