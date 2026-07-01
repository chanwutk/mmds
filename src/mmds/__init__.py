from typing import TYPE_CHECKING

from .dsl import Detect, Filter, ForEach, Gather, Input, Map, Reduce, Unnest
from .execution import PromptExecutor, StaticPromptExecutor, execute
from .execution.llm.gemini import GeminiPromptExecutor
from .optimizers.rewriter.agent import LLMClient, StaticLLMClient
from .model import (
    Assignment,
    DatasetExpr,
    DetectSpec,
    ForEachPrompt,
    GatherSpec,
    JsonValue,
    MMDSValidationError,
    PromptSpec,
    QueryProgram,
    Record,
    RecordPath,
    ResolvedPrompt,
    Row,
    UdfSpec,
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
    "DatasetExpr",
    "Detect",
    "DetectSpec",
    "Filter",
    "ForEach",
    "ForEachPrompt",
    "Gather",
    "GatherSpec",
    "GeminiPromptExecutor",
    "Input",
    "JsonValue",
    "LLMClient",
    "MMDSValidationError",
    "Map",
    "PromptExecutor",
    "PromptSpec",
    "QueryProgram",
    "Record",
    "RecordPath",
    "Reduce",
    "ResolvedPrompt",
    "Row",
    "StaticLLMClient",
    "StaticPromptExecutor",
    "UdfCatalog",
    "UdfEntry",
    "UdfSpec",
    "Unnest",
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
        "MMDS exposes a Python DSL. Import Input, Map, Filter, Reduce, Unnest, Record, and ForEach from mmds."
    )
