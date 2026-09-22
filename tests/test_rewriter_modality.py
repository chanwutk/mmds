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
    ModalitySubstitution,
    PlanIndex,
    apply_rewrite,
)


def _program():
    return parse_query(
        '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(
    rows,
    ["Answer from ", Record["video"], " for ", Record["question"]],
    schema={"answer": "string"},
    name="answer_question",
)
'''
    )


class ModalitySubstitutionTests(unittest.TestCase):
    def test_replaces_only_the_selected_video_reference(self) -> None:
        original = _program()
        directive = ModalitySubstitution()
        match = directive.find_matches(PlanIndex.build(original.output_expr))[0]

        rewritten = apply_rewrite(
            original,
            directive=directive,
            match=match,
            params={
                "video_field": "video",
                "transcript_field": "transcript",
                "rewritten_prompt": "Answer using only the transcript.",
            },
        )

        self.assertIsInstance(rewritten.output_expr.spec, PromptSpec)
        self.assertEqual(
            rewritten.output_expr.spec.parts,
            (
                "Answer using only the transcript.",
                "\ntranscript:\n",
                RecordPath(("transcript",)),
                "\nquestion:\n",
                RecordPath(("question",)),
            ),
        )
        self.assertEqual(
            rewritten.output_expr.spec.output_schema,
            original.output_expr.spec.output_schema,
        )
        self.assertEqual(
            original.output_expr.spec.parts[1],
            RecordPath(("video",)),
        )

    def test_finds_each_prompt_backed_map_at_its_node_path(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
first = Map(rows, ["Inspect ", Record["video"]], schema={"notes": "string"})
output = Map(first, ["Summarize ", Record["notes"]], schema={"answer": "string"})
'''
        )

        matches = ModalitySubstitution().find_matches(
            PlanIndex.build(program.output_expr)
        )

        self.assertEqual(
            [str(match.path) for match in matches],
            ["output", "output.source"],
        )
        self.assertIn("notes", matches[0].summary)
        self.assertIn("video", matches[1].summary)

    def test_rejects_missing_or_nested_video_references(self) -> None:
        programs = (
            _program(),
            parse_query(
                '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(rows, ["Inspect ", Record["media"]["video"]], schema={"answer": "string"})
'''
            ),
        )
        fields = ("missing", "media")

        for program, video_field in zip(programs, fields, strict=True):
            with self.subTest(video_field=video_field):
                directive = ModalitySubstitution()
                match = directive.find_matches(
                    PlanIndex.build(program.output_expr)
                )[0]
                with self.assertRaisesRegex(MMDSRewriteError, "directly reference"):
                    apply_rewrite(
                        program,
                        directive=directive,
                        match=match,
                        params={
                            "video_field": video_field,
                            "transcript_field": "transcript",
                            "rewritten_prompt": "Answer using the transcript.",
                        },
                    )

    def test_parameters_are_strict_nonblank_and_distinct(self) -> None:
        program = _program()
        directive = ModalitySubstitution()
        match = directive.find_matches(PlanIndex.build(program.output_expr))[0]
        invalid = (
            {
                "video_field": "video",
                "transcript_field": "video",
                "rewritten_prompt": "Answer.",
            },
            {
                "video_field": "video",
                "transcript_field": "   ",
                "rewritten_prompt": "Answer.",
            },
            {
                "video_field": "video",
                "transcript_field": 7,
                "rewritten_prompt": "Answer.",
            },
            {
                "video_field": "video",
                "transcript_field": "transcript",
                "rewritten_prompt": "   ",
            },
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
