"""Evaluate sedan presence on the Swan Valley wildlife compilation.

The expected answer is false. Scores
``examples/animals_false_detect_presence.py`` against
``data/animals_false_ground_truth.json``. ``--compare-semantic`` also runs
``examples/animals_false_map.py``.

A clip with no sedan track is dropped and scored as ``sedan_present=false``.

Examples::

  uv run python examples/eval_animals_false.py
  uv run python examples/eval_animals_false.py --compare-semantic
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_bear_eval():
    path = Path(__file__).resolve().parent / "eval_animals_bear.py"
    spec = importlib.util.spec_from_file_location("eval_animals_bear", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    evaluator = _load_bear_eval()
    root = Path(__file__).resolve().parents[1]
    evaluator.FLAG_FIELD = "sedan_present"
    evaluator.PRESENCE_LABEL = "Detect sedan"
    evaluator.DESCRIPTION = __doc__
    evaluator.DEFAULT_PRESENCE_QUERY = (
        root / "examples" / "animals_false_detect_presence.py"
    )
    evaluator.DEFAULT_SEMANTIC_QUERY = root / "examples" / "animals_false_map.py"
    evaluator.DEFAULT_GROUND_TRUTH = root / "data" / "animals_false_ground_truth.json"
    evaluator.main()


if __name__ == "__main__":
    main()
