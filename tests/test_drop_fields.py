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
    DropFields,
    DropFieldsSpec,
    Coalesce,
    Input,
    MMDSValidationError,
    execute,
    parse_query,
    render_query,
    Window,
)
from mmds.operator_catalog import SOURCE_OPERATOR_NAMES  # noqa: E402
from mmds.execution.ops.drop_fields import _apply_drop_fields  # noqa: E402


class DropFieldsTests(unittest.TestCase):
    def test_constructs_immutable_spec(self) -> None:
        node = DropFields(Input("data.jsonl"), ["temporary", "debug"])

        self.assertEqual(node.kind, "drop_fields")
        self.assertEqual(
            node.spec,
            DropFieldsSpec(fields=("temporary", "debug")),
        )

    def test_rejects_empty_duplicate_or_invalid_fields(self) -> None:
        for fields in ([], ["value", "value"], [""]):
            with self.subTest(fields=fields):
                with self.assertRaises(MMDSValidationError):
                    DropFields(Input("data.jsonl"), fields)

    def test_removes_fields_without_mutating_input(self) -> None:
        node = DropFields(Input("data.jsonl"), "temporary")
        row = {"id": 1, "temporary": "value"}

        result = _apply_drop_fields(node, row)

        self.assertEqual(result, {"id": 1})
        self.assertEqual(row, {"id": 1, "temporary": "value"})

    def test_missing_field_raises(self) -> None:
        node = DropFields(Input("data.jsonl"), "temporary")

        with self.assertRaises(MMDSValidationError):
            _apply_drop_fields(node, {"id": 1})

    def test_executes_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(
                json.dumps({"id": 1, "temporary": "value"}) + "\n",
                encoding="utf-8",
            )
            plan = DropFields(Input(str(path)), "temporary", name="cleanup")

            result = execute(plan)
            rendered = render_query(plan)
            reparsed = parse_query(rendered)

        self.assertEqual(result, [{"id": 1}])
        self.assertEqual(reparsed.output_expr, plan)

    def test_window_and_coalesce_are_source_round_trip_operators(self) -> None:
        plan = Coalesce(
            Window(
                Input("data.jsonl"),
                "video",
                "candidate",
                "clip",
                5,
            ),
            "source_id",
            "clip",
        )

        reparsed = parse_query(render_query(plan))

        self.assertEqual(reparsed.output_expr, plan)

    def test_operator_catalog_distinguishes_source_and_internal_operators(self) -> None:
        self.assertIn("Window", SOURCE_OPERATOR_NAMES)
        self.assertIn("Coalesce", SOURCE_OPERATOR_NAMES)
        self.assertIn("DropFields", SOURCE_OPERATOR_NAMES)
        self.assertIn("VideoMap", SOURCE_OPERATOR_NAMES)
        self.assertNotIn("Detect", SOURCE_OPERATOR_NAMES)
        self.assertNotIn("ViewBudget", SOURCE_OPERATOR_NAMES)


if __name__ == "__main__":
    unittest.main()
