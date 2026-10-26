"""Running Gemini over (film, animal) pairs: the loop both WLD experiments share.

``run_pairs`` takes an experiment's per-pair task and, for every pair not yet
done, gives it a fresh Gemini executor, runs it, and writes the task's
result together with every call's token usage. A finished pair is never run
again, so a stopped run resumes where it left off.

A pair that hits a temporary Gemini error (HTTP 500, 503, or a 429 rate
limit) is retried from the start after a wait (``RETRY_WAITS``), so its
measured time never includes the wait. Any other error, including the 429
for a full file-storage quota, stops the run. Tokens billed by failed
attempts are kept in ``_failed_usage.jsonl``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds import execute
from mmds.model import DatasetExpr
from scripts.query_types.common import (
    append_failed_usage,
    record_run_info,
    require_gemini_key,
    usage_collector,
    write_json_atomic,
    write_jsonl,
)
from scripts.query_types.wld import Pair

MODEL = "gemini-3.1-flash-lite"
RETRY_WAITS = (60, 120, 240, 480, 960)  # seconds before retries 1-5

UsageSink = Callable[[dict[str, Any]], None]
ExecutorFactory = Callable[[UsageSink], Any]
Task = Callable[[Pair, Any, Path], dict[str, Any]]  # (pair, executor, scratch) -> result


def gemini_executor_factory() -> ExecutorFactory:
    """Executors for ``MODEL`` that report each call's usage to a sink."""
    require_gemini_key()
    from mmds import GeminiPromptExecutor

    return lambda sink: GeminiPromptExecutor(model=MODEL, usage_sink=sink)


def is_transient(error: BaseException) -> bool:
    """A Gemini error worth retrying: a server error or a rate limit, not a full quota."""
    code = getattr(error, "code", None)
    if code in (500, 503):
        return True
    return code == 429 and "file_storage" not in str(error)


def ask_row(plan: Callable[[str], DatasetExpr], row: Mapping[str, Any], input_path: Path, executor: Any) -> dict[str, Any]:
    """Run a one-row plan: write ``row`` to ``input_path`` and return the single output row."""
    rows = execute(plan(str(write_jsonl(input_path, [row]))), executor)
    if len(rows) != 1:
        raise ValueError(f"{row.get('pair_id')}: expected one answer, got {len(rows)}.")
    return rows[0]


def run_pairs(
    pairs: Sequence[Pair],
    result_dir: Path,
    task: Task,
    *,
    scratch: Path,
    executor_factory: ExecutorFactory | None = None,
    waits: Sequence[float] = RETRY_WAITS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run ``task`` on every pair without a result file in ``result_dir``."""
    pending = [pair for pair in pairs if not (result_dir / f"{pair.file_stem}.json").exists()]
    print(f"{result_dir}: {len(pairs) - len(pending)} pairs done, {len(pending)} to run", flush=True)
    if not pending:
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    record_run_info(result_dir)
    factory = executor_factory or gemini_executor_factory()
    for pair in pending:
        print(f"  {pair.pair_id} ...", flush=True)
        for attempt in range(1, len(waits) + 2):
            records, sink = usage_collector()
            try:
                result = task(pair, factory(sink), scratch)
                break
            except Exception as error:
                label = {"pair_id": pair.pair_id, "attempt": attempt, "error": f"{type(error).__name__}: {error}"[:300]}
                append_failed_usage(result_dir / "_failed_usage.jsonl", label, records)
                if attempt > len(waits) or not is_transient(error):
                    raise
                print(f"    {label['error'][:120]}; retrying in {waits[attempt - 1]:.0f} s", flush=True)
                sleep(waits[attempt - 1])
        write_json_atomic(
            result_dir / f"{pair.file_stem}.json",
            {"pair_id": pair.pair_id, "model": MODEL, **result, "attempts": attempt, "usage": list(records)},
        )
