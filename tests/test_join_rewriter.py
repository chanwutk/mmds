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

from examples.join_cross_camera_vehicle import (  # noqa: E402
    DEFAULT_MANIFEST as JOIN_DEFAULT_MANIFEST,
)
from examples.join_cross_camera_vehicle import build_query as build_join_query  # noqa: E402
from examples.semantic_join_cross_camera_vehicle import (  # noqa: E402
    DEFAULT_MANIFEST as SEMANTIC_DEFAULT_MANIFEST,
)
from examples.semantic_join_cross_camera_vehicle import (  # noqa: E402
    build_query as build_semantic_query,
)
from mmds import StaticLLMClient, StaticPromptExecutor, execute, parse_query, render_query  # noqa: E402
from mmds.model import JoinSpec, MMDSValidationError, PromptSpec  # noqa: E402
from mmds.optimizers.rewriter.agent import build_rewrite_prompt, rewrite  # noqa: E402


def _feed_row(camera_id: str, title: str, path: str) -> dict:
    return {
        "camera_id": camera_id,
        "title": title,
        "video": {"type": "Video", "path": path},
    }


def _trajectory_vehicle(
    vehicle_id: str,
    *,
    color: str,
    upstream_entered: float,
    upstream_exited: float,
    downstream_entered: float,
    downstream_exited: float,
    match_score: float,
) -> dict:
    return {
        "vehicle_id": vehicle_id,
        "attributes": {"class": "sedan", "color": color, "subtype": "sedan"},
        "timeline": [
            {
                "camera_id": "cam-i24v-highway2",
                "entered": upstream_entered,
                "exited": upstream_exited,
            },
            {
                "camera_id": "cam-i24v-highway3",
                "entered": downstream_entered,
                "exited": downstream_exited,
            },
        ],
        "match_score": match_score,
    }


class SemanticToJoinRewriteTests(unittest.TestCase):
    def test_both_builders_default_to_the_same_manifest(self) -> None:
        self.assertEqual(SEMANTIC_DEFAULT_MANIFEST, JOIN_DEFAULT_MANIFEST)
        self.assertEqual(
            SEMANTIC_DEFAULT_MANIFEST,
            "data/i24v_traffic_highway2_highway3_5s.jsonl",
        )

    def test_semantic_baseline_has_reduce_unnest_rewrite_stages(self) -> None:
        plan = build_semantic_query("feeds.jsonl")
        self.assertEqual(
            [(node.kind, node.name) for node in plan.walk_postorder()],
            [
                ("input", None),
                ("reduce", "stitch_cross_camera_vehicles"),
                ("unnest", "one_vehicle_per_row"),
                ("map", "promote_vehicle_trajectory"),
            ],
        )
        reduce_node = plan.source.source
        self.assertIsNotNone(reduce_node)
        self.assertIsInstance(reduce_node.spec, PromptSpec)

    def test_rewrite_semantic_baseline_to_detect_track_join(self) -> None:
        baseline = render_query(build_semantic_query("feeds.jsonl"))
        target = render_query(build_join_query("feeds.jsonl"))

        rewritten = rewrite(
            baseline,
            StaticLLMClient(f"```python\n{target}\n```"),
            objective="replace semantic stitch with detect-track-join",
        )
        program = parse_query(rewritten)

        self.assertEqual(program.input_paths(), ("feeds.jsonl",))
        self.assertEqual(render_query(program), rewritten)
        self.assertEqual(
            [(node.kind, node.name) for node in program.output_expr.walk_postorder()],
            [
                ("input", None),
                ("detect", "detect_vehicles"),
                ("map", "suppress_duplicate_detections"),
                ("map", "build_frame_detections"),
                ("map", "track_vehicles"),
                ("map", "embed_track_summaries"),
                ("unnest", "one_track_per_row"),
                ("map", "promote_track_summary"),
                ("map", "project_track_fields"),
                ("join", "match_cross_camera_tracks"),
                ("map", "build_vehicle_trajectories"),
            ],
        )
        join = program.output_expr.source
        self.assertIsNotNone(join)
        self.assertEqual(join.kind, "join")
        self.assertIs(join.source, join.right_source)
        self.assertIsInstance(join.spec, JoinSpec)
        assert isinstance(join.spec, JoinSpec)
        self.assertTrue(join.spec.one_to_one)
        self.assertEqual(join.spec.predicate.name, "same_vehicle")
        self.assertEqual(join.spec.score.name, "appearance_match_score")

    def test_semantic_baseline_emits_rewrite_compatible_schema(self) -> None:
        rows = [
            _feed_row(
                "cam-i24v-highway2",
                "I-24 MOTION — highway2 (first 5s)",
                "/tmp/highway2.mp4",
            ),
            _feed_row(
                "cam-i24v-highway3",
                "I-24 MOTION — highway3 (first 5s)",
                "/tmp/highway3.mp4",
            ),
        ]
        expected_vehicles = [
            _trajectory_vehicle(
                "v1",
                color="white",
                upstream_entered=0.0,
                upstream_exited=1.5,
                downstream_entered=1.7,
                downstream_exited=3.0,
                match_score=0.91,
            ),
            _trajectory_vehicle(
                "v2",
                color="black",
                upstream_entered=2.0,
                upstream_exited=4.0,
                downstream_entered=2.2,
                downstream_exited=4.5,
                match_score=0.88,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "feeds.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            plan = build_semantic_query(str(path))
            reduce_node = plan.source.source
            self.assertIsNotNone(reduce_node)
            reduce_spec = reduce_node.spec
            self.assertIsInstance(reduce_spec, PromptSpec)
            assert isinstance(reduce_spec, PromptSpec)

            seen_videos: list[object] = []

            def stitch_handler(resolved_prompt, payload, context):
                seen_videos.extend(
                    part for part in resolved_prompt.parts if isinstance(part, dict)
                )
                self.assertEqual(len(payload), 2)
                return {"vehicles": expected_vehicles}

            result = execute(
                plan,
                prompt_executor=StaticPromptExecutor(
                    {("reduce", reduce_spec.cache_key()): stitch_handler}
                ),
            )

        self.assertEqual(
            [row["video"]["path"] for row in rows],
            [video["path"] for video in seen_videos],
        )
        self.assertEqual(result, expected_vehicles)
        for row in result:
            self.assertEqual(
                set(row),
                {"vehicle_id", "attributes", "timeline", "match_score"},
            )

    def test_prompt_names_supported_operators_and_udf_import_contract(self) -> None:
        prompt = build_rewrite_prompt(render_query(build_semantic_query("feeds.jsonl")))

        for operator in ("Join", "Detect", "Window", "Coalesce"):
            with self.subTest(operator=operator):
                self.assertIn(operator, prompt)
        self.assertIn("Import every referenced operator from mmds", prompt)
        self.assertIn("every referenced UDF explicitly from its udfs.* module", prompt)
        self.assertIn("must pass MMDS parser validation", prompt)

    def test_rewrite_rejects_malformed_output(self) -> None:
        baseline = render_query(build_semantic_query("feeds.jsonl"))

        with self.assertRaises((SyntaxError, MMDSValidationError)):
            rewrite(
                baseline,
                StaticLLMClient(
                    "from mmds import Input\noutput = Input('feeds.jsonl'\n"
                ),
            )

    def test_rewrite_rejects_changed_input(self) -> None:
        baseline = render_query(build_semantic_query("feeds.jsonl"))
        changed = render_query(build_join_query("different.jsonl"))

        with self.assertRaisesRegex(MMDSValidationError, "preserve"):
            rewrite(baseline, StaticLLMClient(changed))

    def test_parser_requires_imported_join_and_record_helpers(self) -> None:
        with self.assertRaisesRegex(MMDSValidationError, "Join must be imported"):
            parse_query(
                """
from mmds import Input, Map
from udfs.trajectory_ops import join_match_to_trajectory

tracks = Input("tracks.jsonl")
output = Join(tracks, tracks, on="camera_id")
"""
            )
        with self.assertRaisesRegex(MMDSValidationError, "Record must be imported"):
            parse_query(
                """
from mmds import Input, Map

docs = Input("docs.jsonl")
output = Map(docs, ["keep ", Record["title"]], schema={"title": "string"})
"""
            )


if __name__ == "__main__":
    unittest.main()
