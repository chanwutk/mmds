from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    FieldPredicateSpec,
    Filter,
    Input,
    Map,
    PromptSpec,
    Record,
    StaticPromptExecutor,
    execute,
    parse_query,
    render_query,
)
from mmds.optimizers.rewriter import (  # noqa: E402
    BooleanMapCodeFilter,
    MMDSRewriteError,
    PlanIndex,
    apply_rewrite,
)


MAP_SCHEMA = {"summary": "string", "contains_dog": "boolean"}


def _map_then_filter_program(path: str = "clips.jsonl"):
    return parse_query(
        f'''
from mmds import Filter, Input, Map, Record
clips = Input({path!r})
mapped = Map(
    clips,
    [
        "Watch this video clip.\\n",
        "Video content: ",
        Record["video"],
        "\\nReturn a short summary and whether a dog is clearly visible.",
    ],
    schema={MAP_SCHEMA!r},
)
output = Filter(
    mapped,
    [
        "Keep this row only when the clip clearly contains a dog.\\n",
        "Model summary: ",
        Record["summary"],
        "\\ncontains_dog flag: ",
        Record["contains_dog"],
    ],
)
'''
    )


class FieldPredicateFilterTests(unittest.TestCase):
    def test_parse_render_execute_record_field_predicate(self) -> None:
        program = parse_query(
            '''
from mmds import Filter, Input, Map, Record
rows = Input("clips.jsonl")
mapped = Map(
    rows,
    ["Inspect ", Record["video"]],
    schema={"contains_dog": "boolean"},
)
output = Filter(mapped, Record["contains_dog"])
'''
        )
        self.assertIsInstance(program.output_expr.spec, FieldPredicateSpec)
        self.assertEqual(program.output_expr.spec.field, "contains_dog")

        rendered = render_query(program)
        self.assertIn('Filter(mapped, Record["contains_dog"])', rendered)
        reparsed = parse_query(rendered)
        self.assertEqual(reparsed.output_expr, program.output_expr)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clips.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"video": "a.mp4", "keep": True},
                        {"video": "b.mp4", "keep": False},
                    )
                ),
                encoding="utf-8",
            )
            runtime = Filter(
                Map(
                    Input(str(path)),
                    ["x"],
                    schema={"contains_dog": "boolean"},
                ),
                Record["contains_dog"],
            )
            responses = iter(
                (
                    {"contains_dog": True},
                    {"contains_dog": False},
                )
            )
            result = execute(
                runtime,
                prompt_executor=StaticPromptExecutor(
                    {("map", "x"): lambda *_args: next(responses)}
                ),
            )
        self.assertEqual(
            result,
            [{"video": "a.mp4", "keep": True, "contains_dog": True}],
        )


class BooleanMapCodeFilterTests(unittest.TestCase):
    def test_rewrites_map_then_llm_filter_to_field_predicate(self) -> None:
        original = _map_then_filter_program()
        rewritten = apply_rewrite(
            original,
            directive=BooleanMapCodeFilter(),
            match=BooleanMapCodeFilter().find_matches(
                PlanIndex.build(original.output_expr)
            )[0],
            params={
                "flag_field": "contains_dog",
                "rewritten_map_prompt": (
                    "Watch the clip and return a short summary plus whether a "
                    "dog is clearly visible."
                ),
                "map_schema": MAP_SCHEMA,
            },
        )

        self.assertEqual(rewritten.output_expr.kind, "filter")
        self.assertEqual(
            rewritten.output_expr.spec,
            FieldPredicateSpec(field="contains_dog"),
        )
        mapped = rewritten.output_expr.source
        self.assertEqual(mapped.kind, "map")
        self.assertIsInstance(mapped.spec, PromptSpec)
        self.assertEqual(mapped.spec.output_schema, MAP_SCHEMA)
        self.assertEqual(
            mapped.spec.parts[0],
            (
                "Watch the clip and return a short summary plus whether a "
                "dog is clearly visible."
            ),
        )

    def test_inserts_map_when_filter_source_is_not_a_prompt_map(self) -> None:
        program = parse_query(
            '''
from mmds import Filter, Input, Record
clips = Input("clips.jsonl")
output = Filter(
    clips,
    ["Keep only dog clips from ", Record["video"]],
)
'''
        )
        rewritten = apply_rewrite(
            program,
            directive=BooleanMapCodeFilter(),
            match=BooleanMapCodeFilter().find_matches(
                PlanIndex.build(program.output_expr)
            )[0],
            params={
                "flag_field": "contains_dog",
                "rewritten_map_prompt": "Decide whether a dog is clearly visible.",
                "map_schema": {"contains_dog": "boolean"},
            },
        )

        mapped = rewritten.output_expr.source
        self.assertEqual(mapped.kind, "map")
        self.assertEqual(mapped.name, "rewrite_materialize_keep_flag")
        self.assertEqual(
            rewritten.output_expr.spec,
            FieldPredicateSpec(field="contains_dog"),
        )

    def test_rejects_schema_drift_on_existing_map(self) -> None:
        program = _map_then_filter_program()
        match = BooleanMapCodeFilter().find_matches(
            PlanIndex.build(program.output_expr)
        )[0]
        with self.assertRaisesRegex(MMDSRewriteError, "preserve the upstream"):
            apply_rewrite(
                program,
                directive=BooleanMapCodeFilter(),
                match=match,
                params={
                    "flag_field": "contains_dog",
                    "rewritten_map_prompt": "Decide.",
                    "map_schema": {"contains_dog": "boolean"},
                },
            )

    def test_parameters_require_boolean_flag_in_schema(self) -> None:
        program = _map_then_filter_program()
        match = BooleanMapCodeFilter().find_matches(
            PlanIndex.build(program.output_expr)
        )[0]
        invalid = (
            {
                "flag_field": "contains_dog",
                "rewritten_map_prompt": "Decide.",
                "map_schema": {"contains_dog": "string"},
            },
            {
                "flag_field": "missing",
                "rewritten_map_prompt": "Decide.",
                "map_schema": MAP_SCHEMA,
            },
            {
                "flag_field": "contains_dog",
                "rewritten_map_prompt": "   ",
                "map_schema": MAP_SCHEMA,
            },
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
                    apply_rewrite(
                        program,
                        directive=BooleanMapCodeFilter(),
                        match=match,
                        params=params,
                    )


if __name__ == "__main__":
    unittest.main()
