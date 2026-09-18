from typing import TYPE_CHECKING
from .dsl import (
    Coalesce,
    Detect,
    DropFields,
    Filter,
    ForEach,
    Input,
    Map,
    Reduce,
    Unnest,
    VideoMap,
    VideoMapEach,
    Window,
)
from .execution import (
    ExecutionMetrics,
    MeasuredExecution,
    PromptExecutor,
    StaticPromptExecutor,
    execute,
    execute_measured,
)
from .execution.llm.gemini import GeminiPromptExecutor
from .optimizers.rewriter.agent import LLMClient, StaticLLMClient
from .optimizers.rewriter import (
    JointTemporalPushdown,
    MMDSRewriteError,
    ModelRewriteAgent,
    ModalitySubstitution,
    PerViewTemporalPushdown,
    ProjectionBeforeMap,
    RewriteSelection,
    RewriteModelClient,
    StaticRewriteAgent,
    rewrite_once,
    search_rewrites,
)
from .model import (
    Assignment,
    DatasetExpr,
    DetectSpec,
    DropFieldsSpec,
    WindowSpec,
    ForEachPrompt,
    JsonValue,
    MMDSValidationError,
    PromptSpec,
    QueryProgram,
    Record,
    RecordPath,
    ResolvedPrompt,
    Row,
    UdfSpec,
    VideoMapSpec,
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
    "Coalesce",
    "DatasetExpr",
    "Detect",
    "DetectSpec",
    "DropFields",
    "DropFieldsSpec",
    "ExecutionMetrics",
    "WindowSpec",
    "Window",
    "Filter",
    "ForEach",
    "ForEachPrompt",
    "GeminiPromptExecutor",
    "Input",
    "JsonValue",
    "JointTemporalPushdown",
    "LLMClient",
    "MMDSValidationError",
    "MMDSRewriteError",
    "ModelRewriteAgent",
    "Map",
    "MeasuredExecution",
    "ModalitySubstitution",
    "PerViewTemporalPushdown",
    "PromptExecutor",
    "PromptSpec",
    "ProjectionBeforeMap",
    "QueryProgram",
    "Record",
    "RecordPath",
    "Reduce",
    "ResolvedPrompt",
    "RewriteSelection",
    "RewriteModelClient",
    "Row",
    "StaticLLMClient",
    "StaticRewriteAgent",
    "StaticPromptExecutor",
    "UdfCatalog",
    "UdfEntry",
    "UdfSpec",
    "Unnest",
    "VideoMap",
    "VideoMapEach",
    "VideoMapSpec",
    "VideoView",
    "canonicalize",
    "discover_udfs",
    "execute",
    "execute_measured",
    "load_query",
    "optimize",
    "parse_query",
    "program_from_plan",
    "render_query",
    "rewrite_once",
    "search_rewrites",
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
        "MMDS exposes a Python DSL and directive rewriter. "
        "Import operators and rewrite_once from mmds."
    )
