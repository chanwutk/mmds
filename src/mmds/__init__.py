from .dsl import Filter, ForEach, Input, Map, Reduce, Unnest
from .execution import PromptExecutor, StaticPromptExecutor, execute
from .gemini_executor import GeminiPromptExecutor
from .llm_optimizer import LLMClient, StaticLLMClient
from .model import (
    Assignment,
    DatasetExpr,
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
)
from .parser import load_query, parse_query
from .render import program_from_plan, render_query
from .rule_optimizer import canonicalize, optimize
from .udf_catalog import UdfCatalog, UdfEntry, discover_udfs

__all__ = [
    "Assignment",
    "DatasetExpr",
    "Filter",
    "ForEach",
    "ForEachPrompt",
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
    "canonicalize",
    "discover_udfs",
    "execute",
    "load_query",
    "optimize",
    "parse_query",
    "program_from_plan",
    "render_query",
]


def main() -> None:
    print("MMDS exposes a Python DSL. Import Input, Map, Filter, Reduce, Unnest, Record, and ForEach from mmds.")
