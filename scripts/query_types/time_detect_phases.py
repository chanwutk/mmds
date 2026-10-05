"""Time a detect-presence query phase by phase: startup vs detection.

Runs one of Charisse's wildlife queries (branch ``detect-directive-queries``)
unchanged, from a checkout of that branch, and times each phase separately:

- ``imports``: importing torch and ultralytics
- ``device``: choosing the device (on a GPU machine: CUDA initialisation)
- ``model_load``: loading the YOLOE weights onto the device
- ``text_encoder``: the first text embedding, which loads the text encoder
- ``detect_run_1`` .. ``detect_run_N``: the query's ``execute()`` with the
  model already loaded (video open + decode + inference)

"Including startup" is the sum of every phase; "excluding startup" is a
detect run. Run each query in a fresh process so startup is paid once::

    PYTHONPATH=src:. python -m scripts.query_types.time_detect_phases \\
        --checkout /path/to/mmds-charisse \\
        --query examples/animals_bear_detect_presence.py \\
        --out bear.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_query_output(query_path: Path) -> Any:
    """Import a query file as a module and return its ``output`` plan."""
    spec = importlib.util.spec_from_file_location(f"_timed_query_{query_path.stem}", query_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import query file {query_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "output"):
        raise ValueError(f"{query_path} does not define an 'output' plan.")
    return module.output


def detect_models(plan: Any) -> list[str]:
    """Return the distinct YOLOE model names used by Detect nodes in ``plan``."""
    models = sorted({node.spec.model for node in plan.walk_postorder() if node.kind == "detect"})
    if not models:
        raise ValueError("The query has no Detect node to time.")
    return models


def detect_classes(plan: Any) -> list[str]:
    """Return the first Detect node's class list (used to warm the text encoder)."""
    for node in plan.walk_postorder():
        if node.kind == "detect":
            return list(node.spec.classes)
    raise ValueError("The query has no Detect node to time.")


def timed(phases: dict[str, float], name: str, fn: Callable[[], Any], sync: Callable[[], None]) -> Any:
    """Run ``fn``, wait for queued GPU work, and record the elapsed seconds."""
    started = time.perf_counter()
    result = fn()
    sync()
    phases[name] = time.perf_counter() - started
    return result


def summarize(phases: dict[str, float]) -> dict[str, float]:
    """Split recorded phases into startup, first detect run, and the total."""
    detect = [seconds for name, seconds in phases.items() if name.startswith("detect_run_")]
    if not detect:
        raise ValueError("No detect runs were recorded.")
    startup = sum(seconds for name, seconds in phases.items() if not name.startswith("detect_run_"))
    return {
        "startup_seconds": startup,
        "detect_first_seconds": detect[0],
        "detect_warm_seconds": detect[-1],
        "including_startup_seconds": startup + detect[0],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkout", type=Path, required=True, help="checkout of the branch whose code is timed")
    parser.add_argument("--query", type=Path, required=True, help="query file, relative to the checkout")
    parser.add_argument("--repeats", type=int, default=2, help="detect runs after startup")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    checkout = args.checkout.resolve()
    out = args.out.resolve()
    # Time the checkout's own code: its src/ and udfs/ win over anything installed.
    sys.path[:0] = [str(checkout / "src"), str(checkout)]
    os.chdir(checkout)

    phases: dict[str, float] = {}
    no_sync: Callable[[], None] = lambda: None
    timed(phases, "imports", lambda: (__import__("torch"), __import__("ultralytics")), no_sync)

    import torch

    from mmds import execute
    from mmds.execution.ops import detect as detect_ops

    if not str(Path(detect_ops.__file__).resolve()).startswith(str(checkout)):
        raise SystemExit(f"Imported mmds from {detect_ops.__file__}, not from {checkout}.")

    device = timed(phases, "device", detect_ops._get_device, no_sync)
    sync = torch.cuda.synchronize if device == "cuda" else no_sync

    plan = load_query_output(checkout / args.query)
    models = [timed(phases, f"model_load:{name}", lambda name=name: detect_ops._get_model(name), sync)
              for name in detect_models(plan)]
    classes = detect_classes(plan)
    timed(phases, "text_encoder", lambda: models[0].get_text_pe(classes), sync)

    rows_seen = []
    for run in range(1, args.repeats + 1):
        rows = timed(phases, f"detect_run_{run}", lambda: list(execute(plan)), sync)
        rows_seen.append(len(rows))

    report = {
        "query": str(args.query),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "phases": phases,
        "rows_per_run": rows_seen,
        **summarize(phases),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
