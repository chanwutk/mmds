from .dsl import Filter, Input, Map, Reduce, Unnest
from .execution import PromptExecutor, StaticPromptExecutor, execute
from .llm_optimizer import LLMClient, StaticLLMClient
from .model import Assignment, DatasetExpr, MMDSValidationError, PromptSpec, QueryProgram, Row, UdfSpec
from .parser import load_query, parse_query
from .render import program_from_plan, render_query
from .rule_optimizer import canonicalize, optimize
from .udf_catalog import UdfCatalog, UdfEntry, discover_udfs

__all__ = [
    "Assignment",
    "DatasetExpr",
    "Filter",
    "Input",
    "LLMClient",
    "MMDSValidationError",
    "Map",
    "PromptExecutor",
    "PromptSpec",
    "QueryProgram",
    "Reduce",
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
    print("MMDS exposes a Python DSL. Import Input, Map, Filter, Reduce, and Unnest from mmds.")
