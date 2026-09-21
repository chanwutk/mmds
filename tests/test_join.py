from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mmds.execution as execution_module  # noqa: E402
from mmds import Input, Join, Map, execute, optimize, parse_query, render_query  # noqa: E402
from mmds.model import DatasetExpr, JoinSpec, MMDSValidationError  # noqa: E402
from udfs.test_ops import (  # noqa: E402
    empty_update,
    join_cross_camera_test_bucket,
    join_rows_share_incident_id,
    join_test_match_score,
)


def _write_jsonl(directory: Path, name: str, rows: list[dict]) -> str:
    path = directory / name
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return str(path)


class JoinValidationTests(unittest.TestCase):
    def test_model_and_dsl_validation(self) -> None:
        left = Input("left.jsonl")
        right = Input("right.jsonl")
        invalid_calls = [
            lambda: Join(left, right),
            lambda: Join(left, right, "not-a-callable"),  # type: ignore[arg-type]
            lambda: Join(left, right, on=[]),
            lambda: Join(left, right, on="id", one_to_one=True),
            lambda: Join(left, right, on="id", score=join_test_match_score),
            lambda: Map(left, empty_update, replace="False"),  # type: ignore[arg-type]
            lambda: DatasetExpr(
                kind="join",
                source=left,
                spec=JoinSpec(keys=("id",)),
            ),
        ]
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, MMDSValidationError)):
                    invalid_call()


class JoinExecutionTests(unittest.TestCase):
    def test_keyed_and_predicate_join_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left_path = _write_jsonl(
                root,
                "left.jsonl",
                [
                    {"incident_id": "a", "camera_id": "cam-1"},
                    {"incident_id": "b", "camera_id": "cam-1"},
                    {"camera_id": "cam-1"},
                ],
            )
            right_path = _write_jsonl(
                root,
                "right.jsonl",
                [
                    {"incident_id": "a", "camera_id": "cam-2"},
                    {"incident_id": "b", "camera_id": "cam-1"},
                    {"incident_id": "c", "camera_id": "cam-2"},
                ],
            )
            keyed = execute(Join(Input(left_path), Input(right_path), on="incident_id"))
            constrained = execute(
                Join(
                    Input(left_path),
                    Input(right_path),
                    join_cross_camera_test_bucket,
                    on="incident_id",
                )
            )
            predicate_only = execute(
                Join(
                    Input(left_path),
                    Input(right_path),
                    join_rows_share_incident_id,
                )
            )

        self.assertEqual(
            [row["left"]["incident_id"] for row in keyed],
            ["a", "b"],
        )
        self.assertEqual(
            [(row["left"]["incident_id"], row["right"]["incident_id"]) for row in constrained],
            [("a", "a")],
        )
        self.assertEqual(len(predicate_only), 2)

    def test_map_replace_discards_join_envelope_even_for_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left_path = _write_jsonl(root, "left.jsonl", [{"incident_id": "a"}])
            right_path = _write_jsonl(root, "right.jsonl", [{"incident_id": "a"}])
            result = execute(
                Map(
                    Join(Input(left_path), Input(right_path), on="incident_id"),
                    empty_update,
                    replace=True,
                )
            )

        self.assertEqual(result, [{}])

    def test_self_join_shared_upstream_is_evaluated_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_jsonl(
                Path(temp_dir),
                "tracks.jsonl",
                [
                    {"incident_id": "a", "camera_id": "left"},
                    {"incident_id": "a", "camera_id": "right"},
                ],
            )
            source = Input(path)
            plan = Join(source, source, join_cross_camera_test_bucket, on="incident_id")
            original_loader = execution_module._load_input_rows
            with patch.object(
                execution_module,
                "_load_input_rows",
                wraps=original_loader,
            ) as loader:
                result = execute(plan)

        self.assertEqual(loader.call_count, 1)
        self.assertEqual(len(result), 2)

    def test_distinct_equal_join_sources_stay_distinct_through_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_jsonl(
                Path(temp_dir),
                "tracks.jsonl",
                [{"incident_id": "a", "camera_id": "left"}],
            )
            plan = Join(Input(path), Input(path), on="incident_id")
            rendered = render_query(plan)
            program = parse_query(rendered)
            join = program.output_expr

        self.assertIsNot(plan.source, plan.right_source)
        self.assertEqual(plan.source, plan.right_source)
        self.assertIsNot(join.source, join.right_source)
        self.assertIn("Join(source_tracks, source_tracks_2, on=\"incident_id\")", rendered)


class JoinParseRenderTests(unittest.TestCase):
    def test_parse_render_preserves_full_join_and_map_replace(self) -> None:
        source = """
from mmds import Input, Join, Map
from udfs.test_ops import empty_update, join_cross_camera_test_bucket, join_test_match_score

tracks = Input("tracks.jsonl")
matches = Join(
    tracks,
    tracks,
    join_cross_camera_test_bucket,
    one_to_one=True,
    score=join_test_match_score,
    min_score=0.5,
    left_key=("camera_id", "track_id"),
    right_key=("camera_id", "track_id"),
    name="matches",
)
output = Map(matches, empty_update, replace=True, name="project")
"""
        program = parse_query(source)
        rendered = render_query(program)
        reparsed = parse_query(rendered)

        self.assertEqual(rendered, render_query(reparsed))
        self.assertIn("one_to_one=True", rendered)
        self.assertIn("min_score=0.5", rendered)
        self.assertIn("replace=True", rendered)
        self.assertIs(
            reparsed.assignments[1].expr.source,
            reparsed.assignments[1].expr.right_source,
        )


class JoinOptimizerTests(unittest.TestCase):
    def test_rule_optimizer_preserves_shared_source_identity(self) -> None:
        source = Map(Input("tracks.jsonl"), empty_update)
        plan = Join(source, source, join_cross_camera_test_bucket)

        optimized = optimize(plan)

        self.assertIs(optimized.source, optimized.right_source)
        self.assertIsNot(optimized.source, source)
        self.assertEqual(optimized, plan)


if __name__ == "__main__":
    unittest.main()
