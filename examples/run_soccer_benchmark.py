"""Run baseline vs. optimized soccer goal-detection pipelines and compare.

Usage:
    GEMINI_API_KEY=<key> PYTHONPATH=src:. .venv/bin/python examples/run_soccer_benchmark.py

Produces a side-by-side table of LLM call counts, precision, recall, and F1
against SoccerNet ground-truth labels stored in data/soccer_goal_clips.jsonl.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

from mmds import GeminiPromptExecutor, execute  # noqa: E402
from mmds.model import PromptSpec, ResolvedPrompt  # noqa: E402

DATA_FILE = ROOT / "data" / "soccer_goal_clips.jsonl"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# Gemini 2.5 Flash pricing (USD per 1M tokens)
_INPUT_COST_PER_M = 0.15
_OUTPUT_COST_PER_M = 0.60


class _CountingExecutor(GeminiPromptExecutor):
    """Counts LLM calls and logs per-call tokens, cost, and response."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_count = 0
        self.call_log: list[dict] = []
        self._pipeline: str = "unknown"

    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any:
        self.call_count += 1
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
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        cost = (in_tok * _INPUT_COST_PER_M + out_tok * _OUTPUT_COST_PER_M) / 1_000_000

        clip_id = payload.get("clip_id", "?") if isinstance(payload, Mapping) else "?"
        game = payload.get("game", "?") if isinstance(payload, Mapping) else "?"
        true_label = payload.get("true_label") if isinstance(payload, Mapping) else None

        self.call_log.append({
            "pipeline": self._pipeline,
            "call_num": self.call_count,
            "clip_id": clip_id,
            "game": game,
            "true_label": true_label,
            "op_type": op_type,
            "response": text.strip(),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(cost, 6),
        })

        import json as _json
        try:
            return _json.loads(text)
        except Exception:
            from mmds.model import MMDSValidationError
            raise MMDSValidationError(f"Gemini returned invalid JSON: {text!r}")


def _load_true_labels() -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with open(DATA_FILE) as f:
        for line in f:
            row = json.loads(line.strip())
            if row:
                labels[row["clip_id"]] = bool(row["true_label"])
    return labels


def _load_query(path: str) -> Any:
    spec = importlib.util.spec_from_file_location("_query", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load query from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _metrics(true_labels: dict[str, bool], result_rows: list[dict]) -> dict[str, float]:
    predicted = {r["clip_id"] for r in result_rows}
    tp = sum(1 for cid in predicted if true_labels.get(cid, False))
    fp = sum(1 for cid in predicted if not true_labels.get(cid, True))
    fn = sum(1 for cid, v in true_labels.items() if v and cid not in predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


URI_CACHE_FILE = ROOT / "data" / "gemini_uri_cache.json"


def _load_uri_cache(executor: _CountingExecutor) -> None:
    """Warm the executor's upload cache from disk (avoids re-uploading across runs)."""
    if not URI_CACHE_FILE.exists():
        return
    cache = json.loads(URI_CACHE_FILE.read_text())
    executor._uploaded_files.update({k: tuple(v) for k, v in cache.items()})


def _save_uri_cache(executor: _CountingExecutor) -> None:
    URI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    URI_CACHE_FILE.write_text(json.dumps(executor._uploaded_files, indent=2))


def _prewarm_uploads(executor: _CountingExecutor) -> None:
    """Upload any clips not yet in Gemini's Files API, reusing cached URIs from disk."""
    _load_uri_cache(executor)
    client = executor._get_client()
    rows = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]
    sources = list(dict.fromkeys(
        r["video"]["source"] for r in rows if r.get("video", {}).get("source")
    ))
    already = sum(1 for s in sources if s in executor._uploaded_files)
    to_upload = [s for s in sources if s not in executor._uploaded_files]
    if already:
        print(f"  Reusing {already} cached file URIs from previous run.")
    if to_upload:
        print(f"  Uploading {len(to_upload)} new clips to Gemini Files API…")
        for i, source in enumerate(to_upload, 1):
            name = Path(source).name
            print(f"    [{i:>2}/{len(to_upload)}] {name}", flush=True)
            executor._upload_video_file(source, client)
        _save_uri_cache(executor)
    print(f"  All uploads ready.\n")


def _run(
    query_path: str,
    label: str,
    shared_executor: _CountingExecutor,
) -> tuple[list[dict], int, float]:
    shared_executor.call_count = 0
    shared_executor._pipeline = label.strip()
    mod = _load_query(query_path)
    t0 = time.time()
    rows = execute(mod.output, prompt_executor=shared_executor)
    elapsed = time.time() - t0
    total_cost = sum(e["cost_usd"] for e in shared_executor.call_log
                     if e["pipeline"] == label.strip())
    print(
        f"  [{label}] {len(rows)} clips passed | "
        f"{shared_executor.call_count} LLM calls | "
        f"~${total_cost:.4f} | {elapsed:.1f}s"
    )
    return rows, shared_executor.call_count, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results as JSON to this path (e.g. examples/soccer_results.json)")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        sys.exit("Set GEMINI_API_KEY before running the benchmark.")
    if not DATA_FILE.exists():
        sys.exit(
            f"{DATA_FILE} not found.\n"
            "Run: PYTHONPATH=src:. .venv/bin/python scripts/prepare_soccer_data.py"
        )

    true_labels = _load_true_labels()
    total = len(true_labels)
    total_goals = sum(true_labels.values())
    print(f"Dataset: {total} clips | {total_goals} goals | {total - total_goals} non-goals\n")

    # One shared executor so both pipelines reuse the same upload cache.
    executor = _CountingExecutor(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
        file_ready_timeout_seconds=600,
    )
    print("Uploading clips…")
    _prewarm_uploads(executor)

    print("Running pipelines…")
    base_rows, base_calls, base_time = _run(
        "examples/soccer_highlights_baseline.py", "Baseline ", executor
    )
    opt_rows, opt_calls, opt_time = _run(
        "examples/soccer_highlights_optimized.py", "Optimized", executor
    )

    bm = _metrics(true_labels, base_rows)
    om = _metrics(true_labels, opt_rows)
    savings = (base_calls - opt_calls) / base_calls * 100 if base_calls else 0.0

    W = 52
    def row(label: str, bv: Any, ov: Any, fmt: str = "") -> str:
        bstr = format(bv, fmt) if fmt else str(bv)
        ostr = format(ov, fmt) if fmt else str(ov)
        return f"║ {label:<28} {bstr:>10} {ostr:>10} ║"

    print(f"\n╔{'═' * W}╗")
    print(f"║{'SOCCER GOAL DETECTION BENCHMARK':^{W}}║")
    print(f"╠{'═' * W}╣")
    print(f"║ {'Metric':<28} {'Baseline':>10} {'Optimized':>10} ║")
    print(f"╠{'═' * W}╣")
    print(row("LLM calls", base_calls, opt_calls))
    print(row("LLM call savings", "—", f"{savings:.1f}%"))
    print(row("Clips passed to output", len(base_rows), len(opt_rows)))
    print(f"╠{'═' * W}╣")
    print(row("Precision", bm["precision"], om["precision"], ".3f"))
    print(row("Recall",    bm["recall"],    om["recall"],    ".3f"))
    print(row("F1",        bm["f1"],        om["f1"],        ".3f"))
    print(row("TP / FP / FN",
              f"{int(bm['tp'])}/{int(bm['fp'])}/{int(bm['fn'])}",
              f"{int(om['tp'])}/{int(om['fp'])}/{int(om['fn'])}"))
    print(f"╠{'═' * W}╣")
    print(row("Wall time (s)", f"{base_time:.1f}", f"{opt_time:.1f}"))
    print(f"╚{'═' * W}╝")

    if args.output:
        base_cost = sum(e["cost_usd"] for e in executor.call_log if e["pipeline"] == "Baseline")
        opt_cost  = sum(e["cost_usd"] for e in executor.call_log if e["pipeline"] == "Optimized")
        result = {
            "dataset": {"total": total, "goals": total_goals, "non_goals": total - total_goals},
            "model": GEMINI_MODEL,
            "baseline": {"llm_calls": base_calls, "wall_time_s": round(base_time, 2),
                         "clips_passed": len(base_rows), "cost_usd": round(base_cost, 6), **bm},
            "optimized": {"llm_calls": opt_calls, "wall_time_s": round(opt_time, 2),
                          "clips_passed": len(opt_rows), "cost_usd": round(opt_cost, 6), **om},
            "llm_call_savings_pct": round(savings, 2),
            "cost_savings_pct": round((base_cost - opt_cost) / base_cost * 100, 2) if base_cost else 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved → {args.output}")

        log_path = args.output.with_stem(args.output.stem + "_calls")
        log_path.write_text(json.dumps(executor.call_log, indent=2))
        print(f"Call log saved → {log_path}")


if __name__ == "__main__":
    main()
