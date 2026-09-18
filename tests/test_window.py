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
    DatasetExpr,
    Input,
    MMDSValidationError,
    Window,
    WindowSpec,
    execute,
)
from mmds.execution.ops.window import _apply_window  # noqa: E402


def _source() -> DatasetExpr:
    return DatasetExpr(kind="input", input_path="data.jsonl")


def _window_node(*, padding_time: float = 5.0) -> DatasetExpr:
    return Window(
        _source(),
        "video",
        "candidate",
        "clip",
        padding_time,
    )


class WindowSpecTests(unittest.TestCase):
    def test_valid_spec_normalizes_padding_to_float(self) -> None:
        spec = WindowSpec(
            video_field="video",
            candidate_field="candidate",
            output_field="clip",
            padding_time=5,
        )

        self.assertEqual(spec.video_field, "video")
        self.assertEqual(spec.candidate_field, "candidate")
        self.assertEqual(spec.output_field, "clip")
        self.assertEqual(spec.padding_time, 5.0)

    def test_default_output_field_and_padding(self) -> None:
        spec = WindowSpec(video_field="video", candidate_field="candidate")

        self.assertEqual(spec.output_field, "clip")
        self.assertEqual(spec.padding_time, 0.0)

    def test_empty_field_names_raise(self) -> None:
        invalid_values = (
            {"video_field": "", "candidate_field": "candidate", "output_field": "clip"},
            {"video_field": "video", "candidate_field": "", "output_field": "clip"},
            {"video_field": "video", "candidate_field": "candidate", "output_field": ""},
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(MMDSValidationError):
                    WindowSpec(**values)

    def test_negative_padding_raises(self) -> None:
        for padding in (-1, float("inf"), float("nan"), True):
            with self.subTest(padding=padding):
                with self.assertRaises(MMDSValidationError):
                    WindowSpec(
                        video_field="video",
                        candidate_field="candidate",
                        padding_time=padding,
                    )


class WindowDSLTests(unittest.TestCase):
    def test_constructs_window_node(self) -> None:
        node = Window(
            _source(),
            "video",
            "candidate",
            "clip",
            10,
            name="candidate_window",
        )

        self.assertEqual(node.kind, "window")
        self.assertEqual(node.source, _source())
        self.assertEqual(node.name, "candidate_window")
        self.assertEqual(
            node.spec,
            WindowSpec(
                video_field="video",
                candidate_field="candidate",
                output_field="clip",
                padding_time=10,
            ),
        )

    def test_invalid_configuration_raises(self) -> None:
        cases = (
            ("", "candidate", "clip", 5),
            ("video", "", "clip", 5),
            ("video", "candidate", "", 5),
            ("video", "candidate", "clip", -1),
            ("video", "candidate", "clip", float("inf")),
            ("video", "candidate", "clip", float("nan")),
            ("video", "candidate", "clip", True),
        )

        for video_field, candidate_field, output_field, padding_time in cases:
            with self.subTest(
                video_field=video_field,
                candidate_field=candidate_field,
                output_field=output_field,
                padding_time=padding_time,
            ):
                with self.assertRaises(TypeError):
                    Window(
                        _source(),
                        video_field,
                        candidate_field,
                        output_field,
                        padding_time,
                    )

    def test_non_dataset_source_raises(self) -> None:
        with self.assertRaises(TypeError):
            Window(
                "not-a-plan",  # type: ignore[arg-type]
                "video",
                "candidate",
                "clip",
                5,
            )


class ApplyWindowTests(unittest.TestCase):
    def test_pads_candidate_and_preserves_row(self) -> None:
        row = {
            "source_id": "video-1",
            "video": {"type": "Video", "source": "video.mp4"},
            "candidate": {"start": 20, "end": 30},
        }

        result = _apply_window(_window_node(), row)

        self.assertEqual(
            result,
            {
                "source_id": "video-1",
                "video": {"type": "Video", "source": "video.mp4"},
                "candidate": {"start": 20, "end": 30},
                "clip": {
                    "type": "VideoView",
                    "source": "video.mp4",
                    "start": 15.0,
                    "end": 35.0,
                },
            },
        )

    def test_clamps_padded_start_to_zero(self) -> None:
        row = {
            "video": {"type": "Video", "source": "video.mp4"},
            "candidate": {"start": 2, "end": 7},
        }

        result = _apply_window(_window_node(padding_time=5), row)

        self.assertEqual(result["clip"]["start"], 0)
        self.assertEqual(result["clip"]["end"], 12.0)

    def test_zero_padding_preserves_candidate_bounds(self) -> None:
        row = {
            "video": {"type": "Video", "source": "video.mp4"},
            "candidate": {"start": 2.5, "end": 7.5},
        }

        result = _apply_window(_window_node(padding_time=0), row)

        self.assertEqual(result["clip"]["start"], 2.5)
        self.assertEqual(result["clip"]["end"], 7.5)

    def test_does_not_mutate_input_row_or_video(self) -> None:
        row = {
            "video": {"type": "Video", "source": "video.mp4"},
            "candidate": {"start": 20, "end": 30},
        }
        original = {
            "video": {"type": "Video", "source": "video.mp4"},
            "candidate": {"start": 20, "end": 30},
        }

        result = _apply_window(_window_node(), row)

        self.assertEqual(row, original)
        self.assertIsNot(result, row)
        self.assertIsNot(result["clip"], row["video"])

    def test_end_not_after_start_raises(self) -> None:
        for candidate in ({"start": 10, "end": 10}, {"start": 10, "end": 5}):
            with self.subTest(candidate=candidate):
                row = {
                    "video": {"type": "Video", "source": "video.mp4"},
                    "candidate": candidate,
                }
                with self.assertRaises(MMDSValidationError):
                    _apply_window(_window_node(), row)


class WindowExecutionTests(unittest.TestCase):
    def test_execute_dispatches_window_over_input_rows(self) -> None:
        rows = [
            {
                "source_id": "video-1",
                "video": {"type": "Video", "source": "video.mp4"},
                "candidate": {"start": 10, "end": 20},
            },
            {
                "source_id": "video-2",
                "video": {"type": "Video", "source": "other.mp4"},
                "candidate": {"start": 3, "end": 8},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "rows.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            plan = Window(
                Input(str(input_path)),
                "video",
                "candidate",
                "clip",
                5,
            )

            result = execute(plan)

        self.assertEqual(
            [row["clip"] for row in result],
            [
                {
                    "type": "VideoView",
                    "source": "video.mp4",
                    "start": 5.0,
                    "end": 25.0,
                },
                {
                    "type": "VideoView",
                    "source": "other.mp4",
                    "start": 0,
                    "end": 13.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
