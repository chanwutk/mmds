from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from threading import Lock
from time import monotonic
from typing import Any

from ..model import PromptSpec, ResolvedPrompt
from ._spec import PromptExecutor


@dataclass(frozen=True)
class ExecutionMetrics:
    elapsed_seconds: float
    prompt_calls: int
    prompt_seconds: float
    video_seconds: float
    input_rows: int
    output_rows: int


@dataclass(frozen=True)
class MeasuredExecution:
    rows: list[dict[str, Any]]
    metrics: ExecutionMetrics


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self.prompt_calls = 0
        self.prompt_seconds = 0.0
        self.video_seconds = 0.0
        self.input_rows = 0

    def observe_input_rows(self, count: int) -> None:
        with self._lock:
            self.input_rows += count

    def observe_prompt(self, elapsed_seconds: float, video_seconds: float) -> None:
        with self._lock:
            self.prompt_calls += 1
            self.prompt_seconds += elapsed_seconds
            self.video_seconds += video_seconds

    def snapshot(
        self,
        *,
        elapsed_seconds: float,
        output_rows: int,
    ) -> ExecutionMetrics:
        with self._lock:
            return ExecutionMetrics(
                elapsed_seconds=elapsed_seconds,
                prompt_calls=self.prompt_calls,
                prompt_seconds=self.prompt_seconds,
                video_seconds=self.video_seconds,
                input_rows=self.input_rows,
                output_rows=output_rows,
            )


class MeasuringPromptExecutor:
    def __init__(
        self,
        delegate: PromptExecutor,
        collector: MetricsCollector,
    ) -> None:
        self._delegate = delegate
        self._collector = collector

    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any:
        video_seconds = sum(
            _video_seconds(value) for value in resolved_prompt.parts
        )
        started = monotonic()
        try:
            return self._delegate.execute(
                op_type,
                prompt,
                resolved_prompt,
                payload,
                context,
            )
        finally:
            self._collector.observe_prompt(
                monotonic() - started,
                video_seconds,
            )


def _video_seconds(value: Any) -> float:
    if isinstance(value, Mapping):
        media_type = value.get("type")
        if isinstance(media_type, str) and media_type.casefold() == "videoview":
            start = value.get("start")
            end = value.get("end")
            if (
                isinstance(start, Real)
                and not isinstance(start, bool)
                and isinstance(end, Real)
                and not isinstance(end, bool)
                and end > start
            ):
                return float(end - start)
        return sum(_video_seconds(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_video_seconds(item) for item in value)
    return 0.0
