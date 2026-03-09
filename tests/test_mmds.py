from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    Filter,
    Input,
    MMDSValidationError,
    Map,
    Reduce,
    StaticLLMClient,
    StaticPromptExecutor,
    Unnest,
    discover_udfs,
    execute,
    load_query,
    optimize,
    program_from_plan,
    render_query,
)
from mmds.llm_optimizer import build_rewrite_prompt, rewrite  # noqa: E402
from udfs.test_ops import add_bucket, annotate, keep_large, summarize_group  # noqa: E402


class QueryRoundTripTests(unittest.TestCase):
    def test_parse_render_parse_round_trip(self) -> None:
        query = """
from mmds import Input, Map, Filter, Reduce, Unnest
from udfs.test_ops import add_bucket, summarize_group

docs = Input("docs")
mapped = Map(docs, add_bucket)
filtered = Filter(mapped, "keep bucketed rows")
expanded = Unnest(filtered, "tags", keep_empty=True)
output = Reduce(expanded, ["bucket"], summarize_group)
"""
        program = load_query(query)
        rendered = render_query(program)
        reparsed = load_query(rendered)

        self.assertEqual(program.input_names(), ("docs",))
        self.assertEqual(rendered, render_query(reparsed))
        self.assertEqual(reparsed.output_name, "output")

    def test_program_from_runtime_plan_renders_normalized_python(self) -> None:
        plan = Filter(Map(Input("docs"), annotate), "keep labels")
        program = program_from_plan(plan)
        rendered = render_query(program)

        self.assertIn("from udfs.test_ops import annotate", rendered)
        self.assertIn('output = Filter(step_1, "keep labels")', rendered)


class ExecutionTests(unittest.TestCase):
    def test_execute_udf_pipeline(self) -> None:
        rows = [
            {"value": 1, "tags": ["a", "b"]},
            {"value": 2, "tags": ["c"]},
            {"value": 4, "tags": []},
        ]
        plan = Reduce(
            Unnest(Filter(Map(Input("docs"), add_bucket), keep_large), "tags", keep_empty=True),
            ["bucket"],
            summarize_group,
        )

        result = execute(plan, {"docs": rows})
        self.assertEqual(
            sorted(result, key=lambda row: row["bucket"]),
            [
                {"bucket": 1, "count": 1, "total": 2},
                {"bucket": 2, "count": 1, "total": 4},
            ],
        )

    def test_execute_prompt_pipeline_with_static_executor(self) -> None:
        rows = [{"value": 1}, {"value": 3}, {"value": 5}]
        plan = Reduce(
            Filter(Map(Input("docs"), "label rows"), "keep large rows"),
            "_all",
            "sum values",
        )
        executor = StaticPromptExecutor(
            {
                ("map", "label rows"): lambda row, ctx: {"label": f"row-{row['value']}"},
                ("filter", "keep large rows"): lambda row, ctx: row["value"] >= 3,
                ("reduce", "sum values"): lambda rows, ctx: {
                    "total": sum(row["value"] for row in rows),
                    "count": len(rows),
                },
            }
        )

        result = execute(plan, {"docs": rows}, prompt_executor=executor)
        self.assertEqual(result, [{"count": 2, "total": 8}])

    def test_unnest_scalar_and_empty_behavior(self) -> None:
        rows = [
            {"value": 1, "tags": "solo"},
            {"value": 2, "tags": []},
            {"value": 3},
        ]
        plan = Unnest(Input("docs"), "tags", keep_empty=True)
        result = execute(plan, {"docs": rows})

        self.assertEqual(
            result,
            [
                {"value": 1, "tags": "solo"},
                {"value": 2, "tags": None},
                {"value": 3, "tags": None},
            ],
        )


class ValidationTests(unittest.TestCase):
    def test_lambda_specs_are_rejected(self) -> None:
        with self.assertRaises(MMDSValidationError):
            Map(Input("docs"), lambda row: row)

    def test_parser_rejects_unsupported_python(self) -> None:
        query = """
from mmds import Input, Map

docs = Input("docs")
for row in []:
    docs = Map(docs, "noop")
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_parser_rejects_non_udf_callable_specs(self) -> None:
        query = """
from mmds import Input, Map
from math import ceil

docs = Input("docs")
output = Map(docs, ceil)
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)


class OptimizerTests(unittest.TestCase):
    def test_rule_optimizer_preserves_results(self) -> None:
        rows = [{"value": 2}, {"value": 4}]
        plan = Map(Input("docs"), annotate)

        original = execute(plan, {"docs": rows})
        optimized = execute(optimize(plan), {"docs": rows})

        self.assertEqual(original, optimized)

    def test_llm_optimizer_rewrites_and_validates_inputs(self) -> None:
        query = """
from mmds import Input, Map, Filter
from udfs.test_ops import annotate

docs = Input("docs")
tagged = Map(docs, annotate)
output = Filter(tagged, "keep everything")
"""
        client = StaticLLMClient(
            """
```python
from mmds import Input, Map, Filter
from udfs.test_ops import annotate

docs = Input("docs")
mapped = Map(docs, annotate)
output = Filter(mapped, "keep everything")
```
"""
        )

        rewritten = rewrite(query, client, objective="latency")
        self.assertIn("output = Filter(mapped, \"keep everything\")", rewritten)
        self.assertIn("Optimization objective: latency.", build_rewrite_prompt(query, objective="latency"))

    def test_llm_optimizer_rejects_changed_inputs(self) -> None:
        query = """
from mmds import Input, Map
from udfs.test_ops import annotate

docs = Input("docs")
output = Map(docs, annotate)
"""
        client = StaticLLMClient(
            """
from mmds import Input, Map
from udfs.test_ops import annotate

other = Input("other")
output = Map(other, annotate)
"""
        )
        with self.assertRaises(MMDSValidationError):
            rewrite(query, client)


class UdfCatalogTests(unittest.TestCase):
    def test_discover_udfs_finds_python_and_stub_entries(self) -> None:
        catalog = discover_udfs(ROOT / "udfs")
        implemented = catalog.get("udfs.test_ops", "annotate")
        planned = catalog.get("udfs.planned_only", "synthesize_summary")

        self.assertIsNotNone(implemented)
        self.assertTrue(implemented.implemented)
        self.assertIsNotNone(planned)
        self.assertFalse(planned.implemented)
        self.assertIn("def synthesize_summary", planned.signature)


if __name__ == "__main__":
    unittest.main()
