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

from mmds import parse_query  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    MMDSRewriteError,
    PlanIndex,
    build_rewrite_context,
)


def _program(query_path: Path):
    return parse_query(
        '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(
    rows,
    ["Find ", Record["query_text"], " in ", Record["video"]],
    schema={"events": {"type": "array", "items": {"type": "object"}}},
)
''',
        path=query_path,
    )


class RewriteContextTests(unittest.TestCase):
    def test_summarizes_plan_and_field_roles_without_row_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = directory / "query.py"
            row = {
                "lecture_id": "SECRET ID",
                "query_text": "SECRET QUERY",
                "video": {"type": "Video", "source": "SECRET URL"},
                "transcript": [
                    {"start": 1, "end": 2, "text": "SECRET TRANSCRIPT"}
                ],
                "ground_truth_events": [{"start": 1, "end": 2}],
                "annotation_policy": "SECRET POLICY",
            }
            (directory / "lectures.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

            context = build_rewrite_context(_program(query_path))

        self.assertEqual(
            [(node["path"], node["kind"]) for node in context["plan"]],
            [("output", "map"), ("output.source", "input")],
        )
        dataset = context["datasets"][0]
        self.assertTrue(dataset["available"])
        self.assertEqual(dataset["rows_profiled"], 1)
        self.assertEqual(dataset["evaluation_fields_excluded"], 2)
        fields = dataset["fields"]
        self.assertEqual(fields["lecture_id"]["role"], "identifier")
        self.assertEqual(fields["query_text"]["role"], "query")
        self.assertEqual(fields["video"]["role"], "video")
        self.assertEqual(
            fields["transcript"]["role"],
            "timestamped_transcript",
        )
        self.assertNotIn("ground_truth_events", fields)
        self.assertNotIn("annotation_policy", fields)

        serialized = json.dumps(context)
        for secret in (
            "SECRET ID",
            "SECRET QUERY",
            "SECRET URL",
            "SECRET TRANSCRIPT",
            "SECRET POLICY",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, serialized)

    def test_referenced_evaluation_field_is_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = directory / "query.py"
            (directory / "lectures.jsonl").write_text(
                json.dumps({"video": "v.mp4", "label": "category"}) + "\n",
                encoding="utf-8",
            )
            program = parse_query(
                '''
from mmds import Input, Map, Record
rows = Input("lectures.jsonl")
output = Map(rows, [Record["video"], Record["label"]], schema={"answer": "string"})
''',
                path=query_path,
            )

            context = build_rewrite_context(program)

        self.assertIn("label", context["datasets"][0]["fields"])

    def test_missing_file_is_nonblocking_but_malformed_file_fails(self) -> None:
        missing = parse_query(
            '''
from mmds import Input
output = Input("missing.jsonl")
'''
        )
        self.assertFalse(build_rewrite_context(missing)["datasets"][0]["available"])

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "lectures.jsonl").write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(MMDSRewriteError, "invalid JSON"):
                build_rewrite_context(_program(directory / "query.py"))

    def test_context_uses_at_most_eight_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            rows = [
                {"video": "v.mp4", "query_text": "q", "lecture_id": index}
                for index in range(12)
            ]
            (directory / "lectures.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            context = build_rewrite_context(_program(directory / "query.py"))

        self.assertEqual(context["datasets"][0]["rows_profiled"], 8)

    def test_supplied_index_must_describe_the_program(self) -> None:
        program = _program(Path("query.py"))
        other = parse_query(
            '''
from mmds import Input
output = Input("other.jsonl")
'''
        )

        with self.assertRaisesRegex(MMDSRewriteError, "output plan"):
            build_rewrite_context(
                program,
                index=PlanIndex.build(other.output_expr),
            )


if __name__ == "__main__":
    unittest.main()
