"""Execution controls and structured runtime statistics."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..model import MMDSValidationError


@dataclass(frozen=True)
class MaterializedClip:
    path: Path
    size_bytes: int
    sha256: str
    duration_seconds: float


class MediaMaterializer(Protocol):
    def materialize(
        self, source: Path, destination: Path, start_seconds: float, end_seconds: float
    ) -> MaterializedClip: ...


@dataclass(frozen=True)
class OperatorExecutionStat:
    kind: str
    name: str | None
    input_rows: int
    output_rows: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ViewExecutionStat:
    name: str | None
    source: str
    start_seconds: float
    end_seconds: float
    output_path: str | None
    output_bytes: int
    reused: bool


@dataclass(frozen=True)
class ProviderExecutionStat:
    provider: str
    model: str
    operator_kind: str
    operator_name: str | None
    elapsed_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass
class ExecutionStats:
    """Thread-safe execution facts; evaluation and pricing remain external."""

    _operators: list[OperatorExecutionStat] = field(default_factory=list, init=False)
    _views: list[ViewExecutionStat] = field(default_factory=list, init=False)
    _provider_calls: list[ProviderExecutionStat] = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def operators(self) -> tuple[OperatorExecutionStat, ...]:
        with self._lock:
            return tuple(self._operators)

    @property
    def views(self) -> tuple[ViewExecutionStat, ...]:
        with self._lock:
            return tuple(self._views)

    @property
    def provider_calls(self) -> tuple[ProviderExecutionStat, ...]:
        with self._lock:
            return tuple(self._provider_calls)

    def record_operator(self, value: OperatorExecutionStat) -> None:
        with self._lock:
            self._operators.append(value)

    def record_view(self, value: ViewExecutionStat) -> None:
        with self._lock:
            self._views.append(value)

    def record_provider_call(self, value: ProviderExecutionStat) -> None:
        with self._lock:
            self._provider_calls.append(value)


@dataclass
class ExecutionContext:
    """Explicit resources and policies for one query execution."""

    workspace: Path | None = None
    max_workers: int = 1
    materializer: MediaMaterializer | None = None
    stats: ExecutionStats = field(default_factory=ExecutionStats)
    _view_cache: dict[str, MaterializedClip] = field(default_factory=dict, init=False)
    _view_locks: dict[str, threading.Lock] = field(default_factory=dict, init=False)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int):
            raise TypeError("ExecutionContext max_workers must be an integer.")
        if self.max_workers <= 0:
            raise MMDSValidationError("ExecutionContext max_workers must be positive.")
        if self.workspace is not None:
            self.workspace = Path(self.workspace).resolve()

    def cached_view(self, key: str) -> MaterializedClip | None:
        with self._cache_lock:
            return self._view_cache.get(key)

    def cache_view(self, key: str, clip: MaterializedClip) -> None:
        with self._cache_lock:
            self._view_cache[key] = clip

    def view_lock(self, key: str) -> threading.Lock:
        with self._cache_lock:
            return self._view_locks.setdefault(key, threading.Lock())


__all__ = [
    "ExecutionContext",
    "ExecutionStats",
    "MaterializedClip",
    "MediaMaterializer",
    "OperatorExecutionStat",
    "ProviderExecutionStat",
    "ViewExecutionStat",
]
