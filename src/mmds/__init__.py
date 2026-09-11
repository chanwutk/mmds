from typing import TYPE_CHECKING

from .dsl import (
    Detect,
    Filter,
    ForEach,
    Input,
    Map,
    PadInterval,
    ReconcileIntervals,
    Reduce,
    Resolve,
    Unnest,
    View,
)
from .execution import (
    ExecutionContext,
    ExecutionStats,
    PromptExecutor,
    StaticPromptExecutor,
    execute,
)
from .execution.llm.gemini import GeminiPromptExecutor
from .optimizers.rewriter.agent import LLMClient, StaticLLMClient
from .optimizers.cross_modal import CrossModalTemporalPushdown, ModalitySubstitution
from .model import (
    Assignment,
    DatasetExpr,
    DetectSpec,
    ForEachPrompt,
    JsonValue,
    MMDSExecutionError,
    MMDSValidationError,
    PadIntervalSpec,
    PromptSpec,
    QueryProgram,
    Record,
    RecordPath,
    ReconcileIntervalsSpec,
    ResolveSpec,
    ResolvedPrompt,
    Row,
    UdfSpec,
    ViewSpec,
)
from .parser import load_query, parse_query
from .render import program_from_plan, render_query
from .optimizers.rewriter.rule import canonicalize, optimize
from .udf_catalog import UdfCatalog, UdfEntry, discover_udfs

if TYPE_CHECKING:
    # Importing the video utility pulls in OpenCV/NumPy. Keep it out of the
    # eager import path so `import mmds` works without the heavy CV stack;
    # `VideoView` is loaded lazily via __getattr__ below.
    from .utilities.video import VideoView

__all__ = [
    "Assignment",
    "CrossModalTemporalPushdown",
    "DatasetExpr",
    "Detect",
    "DetectSpec",
    "ExecutionContext",
    "ExecutionStats",
    "Filter",
    "ForEach",
    "ForEachPrompt",
    "GeminiPromptExecutor",
    "Input",
    "JsonValue",
    "LLMClient",
    "MMDSValidationError",
    "MMDSExecutionError",
    "ModalitySubstitution",
    "Map",
    "PadInterval",
    "PadIntervalSpec",
    "PromptExecutor",
    "PromptSpec",
    "QueryProgram",
    "Record",
    "RecordPath",
    "Reduce",
    "ReconcileIntervals",
    "ReconcileIntervalsSpec",
    "Resolve",
    "ResolveSpec",
    "ResolvedPrompt",
    "Row",
    "StaticLLMClient",
    "StaticPromptExecutor",
    "UdfCatalog",
    "UdfEntry",
    "UdfSpec",
    "Unnest",
    "View",
    "ViewSpec",
    "VideoView",
    "canonicalize",
    "discover_udfs",
    "execute",
    "load_query",
    "optimize",
    "parse_query",
    "program_from_plan",
    "render_query",
]


def __getattr__(name: str):
    # Lazy public exports that would otherwise force heavy optional imports
    # (OpenCV/NumPy) at `import mmds` time. See PEP 562.
    if name == "VideoView":
        from .utilities.video import VideoView

        return VideoView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    print(
        "MMDS exposes a Python DSL. See README.md for the operator catalog "
        "and runnable examples."
    )
