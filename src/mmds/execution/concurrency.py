"""Small bounded-concurrency helpers used by streaming unary operators."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def bounded_map(
    function: Callable[[InputT], OutputT],
    values: Iterable[InputT],
    *,
    max_workers: int,
) -> Iterator[OutputT]:
    """Apply in input order with at most ``max_workers`` submitted calls."""
    if max_workers == 1:
        for value in values:
            yield function(value)
        return

    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: deque[Future[OutputT]] = deque()
        for _ in range(max_workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


__all__ = ["bounded_map"]
