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

from examples.join_cross_camera_vehicle import build_query  # noqa: E402
from mmds import MMDSValidationError, parse_query, render_query  # noqa: E402


class ProgrammaticOperatorRoundTripTests(unittest.TestCase):
    def test_detect_window_coalesce_round_trip_with_extended_options(self) -> None:
        source = """
from mmds import Input, Detect, Window, Coalesce

feeds = Input("feeds.jsonl")
detected = Detect(
    feeds,
    "video",
    ["sedan", "truck"],
    model="custom.pt",
    output_field="objects",
    frame_stride=3,
    conf=0.25,
    imgsz=960,
    name="detect_vehicles",
)
windowed = Window(
    detected,
    "video",
    "candidate",
    "clip",
    5,
    name="pad_candidates",
)
output = Coalesce(
    windowed,
    ("camera_id", "source_id"),
    "clip",
    name="merge_clips",
)
"""
        program = parse_query(source)
        rendered = render_query(program)
        reparsed = parse_query(rendered)

        self.assertEqual(program.output_expr, reparsed.output_expr)
        self.assertEqual(rendered, render_query(reparsed))
        self.assertIn(
            "from mmds import Input, Map, Filter, Reduce, Unnest, Join, Detect, Window, Coalesce, Record, ForEach",
            rendered,
        )
        self.assertIn('model="custom.pt"', rendered)
        self.assertIn('output_field="objects"', rendered)
        self.assertIn("frame_stride=3", rendered)
        self.assertIn("conf=0.25", rendered)
        self.assertIn("imgsz=960", rendered)
        self.assertIn(
            'Window(detected, "video", "candidate", "clip", 5.0, name="pad_candidates")',
            rendered,
        )
        self.assertIn(
            'Coalesce(windowed, ["camera_id", "source_id"], "clip", name="merge_clips")',
            rendered,
        )

    def test_detect_defaults_render_normalized_and_reparse(self) -> None:
        program = parse_query(
            """
from mmds import Input, Detect

feeds = Input("feeds.jsonl")
output = Detect(feeds, "video", ("car",), conf=None, imgsz=None)
"""
        )
        rendered = render_query(program)

        self.assertIn('output = Detect(feeds, "video", ["car"])', rendered)
        self.assertNotIn("conf=", rendered)
        self.assertNotIn("imgsz=", rendered)
        self.assertEqual(program.output_expr, parse_query(rendered).output_expr)

    def test_full_cross_camera_plan_renders_and_reparses_equivalently(self) -> None:
        plan = build_query("tracks.jsonl", min_score=0.6)

        rendered = render_query(plan)
        reparsed = parse_query(rendered).output_expr

        self.assertEqual(reparsed, plan)
        self.assertEqual(
            [(node.kind, node.name) for node in reparsed.walk_postorder()],
            [(node.kind, node.name) for node in plan.walk_postorder()],
        )
        join = reparsed.source
        self.assertIsNotNone(join)
        self.assertEqual(join.kind, "join")
        self.assertIs(join.source, join.right_source)
        self.assertIn("Detect(", rendered)
        self.assertIn("Join(", rendered)


class ProgrammaticOperatorValidationTests(unittest.TestCase):
    def test_detect_parser_rejects_invalid_literals_and_configuration(self) -> None:
        invalid_calls = (
            'Detect(feeds, "video", [])',
            'Detect(feeds, "video", ["car", ""])',
            'Detect(feeds, "video", ["car"], model="")',
            'Detect(feeds, "video", ["car"], output_field="")',
            'Detect(feeds, "video", ["car"], frame_stride=0)',
            'Detect(feeds, "video", ["car"], frame_stride=True)',
            'Detect(feeds, "video", ["car"], conf=1.1)',
            'Detect(feeds, "video", ["car"], imgsz=1.5)',
            'Detect(feeds, "video", ["car"], unknown=True)',
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(MMDSValidationError):
                    parse_query(
                        "from mmds import Input, Detect\n"
                        'feeds = Input("feeds.jsonl")\n'
                        f"output = {call}\n"
                    )

    def test_window_and_coalesce_parser_reject_invalid_configuration(self) -> None:
        invalid_programs = (
            (
                "Window",
                "from mmds import Input, Window\n"
                'feeds = Input("feeds.jsonl")\n'
                'output = Window(feeds, "", "candidate", "clip", 5)\n',
            ),
            (
                "Window padding",
                "from mmds import Input, Window\n"
                'feeds = Input("feeds.jsonl")\n'
                'output = Window(feeds, "video", "candidate", "clip", -1)\n',
            ),
            (
                "Coalesce group",
                "from mmds import Input, Coalesce\n"
                'feeds = Input("feeds.jsonl")\n'
                'output = Coalesce(feeds, [], "clip")\n',
            ),
            (
                "Coalesce field",
                "from mmds import Input, Coalesce\n"
                'feeds = Input("feeds.jsonl")\n'
                'output = Coalesce(feeds, "source_id", "")\n',
            ),
        )
        for label, source in invalid_programs:
            with self.subTest(label=label):
                with self.assertRaises(MMDSValidationError):
                    parse_query(source)


if __name__ == "__main__":
    unittest.main()
