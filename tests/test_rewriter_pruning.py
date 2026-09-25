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

from mmds import PromptSpec, RecordPath, parse_query  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    MMDSRewriteError,
    PlanIndex,
    PromptFieldPruning,
    apply_rewrite,
)


def _program():
    return parse_query(
        '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(
    rows,
    [
        "Answer ",
        Record["query"],
        " using ",
        Record["title"],
        " and ",
        Record["video"],
    ],
    schema={"answer": "string"},
    name="answer_question",
)
'''
    )


class PromptFieldPruningTests(unittest.TestCase):
    def test_drops_selected_top_level_fields_and_rewrites_prompt(self) -> None:
        original = _program()
        directive = PromptFieldPruning()
        match = directive.find_matches(PlanIndex.build(original.output_expr))[0]

        rewritten = apply_rewrite(
            original,
            directive=directive,
            match=match,
            params={
                "drop_fields": ["video", "title"],
                "rewritten_prompt": "Answer the query from text context only.",
            },
        )

        self.assertIsInstance(rewritten.output_expr.spec, PromptSpec)
        self.assertEqual(
            rewritten.output_expr.spec.parts,
            (
                "Answer the query from text context only.",
                "\nquery:\n",
                RecordPath(("query",)),
            ),
        )
        self.assertEqual(
            rewritten.output_expr.spec.output_schema,
            original.output_expr.spec.output_schema,
        )
        self.assertEqual(rewritten.output_expr.name, "answer_question")

    def test_finds_each_prompt_backed_map(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
first = Map(
    rows,
    ["Inspect ", Record["video"], " and ", Record["title"]],
    schema={"notes": "string"},
)
output = Map(first, ["Summarize ", Record["notes"]], schema={"answer": "string"})
'''
        )

        matches = PromptFieldPruning().find_matches(
            PlanIndex.build(program.output_expr)
        )

        self.assertEqual(
            [str(match.path) for match in matches],
            ["output", "output.source"],
        )
        self.assertIn("notes", matches[0].summary)
        self.assertIn("title", matches[1].summary)
        self.assertIn("video", matches[1].summary)

    def test_rejects_missing_nested_or_total_pruning(self) -> None:
        nested = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(
    rows,
    [Record["query"], Record["metadata"]["term"]],
    schema={"answer": "string"},
)
'''
        )
        cases = (
            (
                _program(),
                {"drop_fields": ["missing"], "rewritten_prompt": "Answer."},
                "drop field",
            ),
            (
                nested,
                {"drop_fields": ["metadata"], "rewritten_prompt": "Answer."},
                "drop field",
            ),
            (
                parse_query(
                    '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(
    rows,
    [Record["query"], Record["metadata"], Record["metadata"]["term"]],
    schema={"answer": "string"},
)
'''
                ),
                {"drop_fields": ["metadata"], "rewritten_prompt": "Answer."},
                "top-level drop",
            ),
            (
                _program(),
                {
                    "drop_fields": ["query", "title", "video"],
                    "rewritten_prompt": "Answer.",
                },
                "at least one Record",
            ),
        )

        for program, params, error in cases:
            with self.subTest(error=error):
                directive = PromptFieldPruning()
                match = directive.find_matches(
                    PlanIndex.build(program.output_expr)
                )[0]
                with self.assertRaisesRegex(MMDSRewriteError, error):
                    apply_rewrite(
                        program,
                        directive=directive,
                        match=match,
                        params=params,
                    )

    def test_parameters_are_strict_nonblank_and_unique(self) -> None:
        program = _program()
        directive = PromptFieldPruning()
        match = directive.find_matches(PlanIndex.build(program.output_expr))[0]
        invalid = (
            {"drop_fields": [], "rewritten_prompt": "Answer."},
            {"drop_fields": ["video", "video"], "rewritten_prompt": "Answer."},
            {"drop_fields": ["  "], "rewritten_prompt": "Answer."},
            {"drop_fields": ["video"], "rewritten_prompt": "   "},
            {"drop_fields": [7], "rewritten_prompt": "Answer."},
        )

        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
                    apply_rewrite(
                        program,
                        directive=directive,
                        match=match,
                        params=params,
                    )


if __name__ == "__main__":
    unittest.main()
