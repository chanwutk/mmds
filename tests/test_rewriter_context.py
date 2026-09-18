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

from mmds import MMDSRewriteError, parse_query  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    DEFAULT_REWRITE_POLICY,
    build_rewrite_context,
)


def _query(path: Path):
    return parse_query(
        '''
from mmds import Input, Map, Record
rows = Input("rows.jsonl")
output = Map(
    rows,
    ["Find ", Record["question"], " in ", Record["video"]],
    schema={"events": {"type": "array", "items": {"type": "object"}}},
)
''',
        path=path,
    )


class RewriteContextTests(unittest.TestCase):
    def test_context_contains_query_semantics_and_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = directory / "query.py"
            row = {
                "item_id": "lecture-1",
                "question": "SECRET QUESTION VALUE",
                "video": {"type": "Video", "source": "SECRET VIDEO URL"},
                "transcript": [
                    {"start": 1, "end": 2, "text": "SECRET TRANSCRIPT VALUE"}
                ],
                "ground_truth_events": [{"start": 1, "end": 2}],
                "annotation_policy": "SECRET ANNOTATION VALUE",
            }
            (directory / "rows.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

            context = build_rewrite_context(_query(query_path))

        self.assertEqual(context.policy, DEFAULT_REWRITE_POLICY)
        map_summary = context.query_plan[0]
        self.assertEqual(map_summary["kind"], "map")
        self.assertEqual(map_summary["reads"], ["question", "video"])
        self.assertEqual(map_summary["writes"], ["events"])
        self.assertIn("Find ", map_summary["prompt"])

        dataset = context.datasets[0]
        self.assertTrue(dataset["available"])
        self.assertEqual(dataset["rows_profiled"], 1)
        self.assertEqual(dataset["evaluation_fields_excluded"], 2)
        fields = dataset["fields"]
        self.assertEqual(fields["video"]["role"], "video")
        self.assertEqual(fields["transcript"]["role"], "timestamped_transcript")
        self.assertEqual(fields["question"]["role"], "query")
        self.assertEqual(fields["item_id"]["role"], "identifier")
        self.assertNotIn("ground_truth_events", fields)
        self.assertNotIn("annotation_policy", fields)

        serialized = json.dumps(context.as_dict())
        self.assertNotIn("SECRET QUESTION VALUE", serialized)
        self.assertNotIn("SECRET VIDEO URL", serialized)
        self.assertNotIn("SECRET TRANSCRIPT VALUE", serialized)
        self.assertNotIn("SECRET ANNOTATION VALUE", serialized)

    def test_referenced_evaluation_named_field_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = directory / "query.py"
            (directory / "rows.jsonl").write_text(
                json.dumps({"video": "v", "label": "user-visible category"})
                + "\n",
                encoding="utf-8",
            )
            program = parse_query(
                '''
from mmds import Input, Map, Record
rows = Input("rows.jsonl")
output = Map(rows, [Record["video"], Record["label"]], schema={"answer": "string"})
''',
                path=query_path,
            )

            context = build_rewrite_context(program)

        self.assertIn("label", context.datasets[0]["fields"])

    def test_missing_dataset_is_reported_without_blocking_rewrite(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("missing.jsonl")
output = Map(rows, [Record["video"]], schema={"answer": "string"})
'''
        )

        context = build_rewrite_context(program)

        self.assertFalse(context.datasets[0]["available"])
        self.assertEqual(context.datasets[0]["fields"], {})

    def test_existing_invalid_dataset_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = directory / "query.py"
            (directory / "rows.jsonl").write_text("not-json\n", encoding="utf-8")

            with self.assertRaisesRegex(MMDSRewriteError, "invalid JSON"):
                build_rewrite_context(_query(query_path))

    def test_mixed_field_types_are_reported_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            query_path = directory / "query.py"
            rows = [
                {"video": "v", "question": "q", "metadata": 1},
                {"video": "v", "question": "q", "metadata": "text"},
                {"video": "v", "question": "q", "metadata": None},
            ]
            (directory / "rows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            context = build_rewrite_context(_query(query_path))

        self.assertEqual(
            context.datasets[0]["fields"]["metadata"]["types"],
            ["null", "number", "string"],
        )


if __name__ == "__main__":
    unittest.main()
