"""SoccerNet compatibility wrappers over shared experiment Gemini runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.experiments.gemini_runtime import (
    ApiCallRecorder,
    CachingGeminiPromptExecutor as _SharedCachingGeminiPromptExecutor,
    create_experiment_executor as _create_shared_executor,
)

from .common import SoccerNetDataError


EXPERIMENT_MODEL = "gemini-3.1-flash-lite"


class CachingGeminiPromptExecutor(_SharedCachingGeminiPromptExecutor):
    """Preserve the SoccerNet error type while sharing runtime behavior."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("error_type", SoccerNetDataError)
        super().__init__(**kwargs)


def create_experiment_executor(
    *,
    stage_directory: Path,
    model: str = EXPERIMENT_MODEL,
    env_file: Path = Path(".env"),
) -> _SharedCachingGeminiPromptExecutor:
    return _create_shared_executor(
        stage_directory=stage_directory,
        model=model,
        env_file=env_file,
        error_type=SoccerNetDataError,
    )


__all__ = [
    "ApiCallRecorder",
    "CachingGeminiPromptExecutor",
    "EXPERIMENT_MODEL",
    "create_experiment_executor",
]
