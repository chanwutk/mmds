from __future__ import annotations

import math
import unittest

from udfs.lecture_paper_eval_ops import (
    candidate_segment_ranges_to_windows,
    normalize_candidate_video_events,
    normalize_full_video_events,
    normalize_transcript_events,
)


class LecturePaperEvalContractTests(unittest.TestCase):
    def test_empty_event_lists_are_valid_negatives(self) -> None:
        self.assertEqual(
            normalize_full_video_events({"duration_seconds": 100, "events": []}),
            {"events": []},
        )
        self.assertEqual(
            normalize_transcript_events(
                {
                    "duration_seconds": 100,
                    "transcript_segments": _segments(),
                    "transcript_event_ranges": [],
                }
            ),
            {"events": []},
        )

    def test_structurally_invalid_provider_output_fails_closed(self) -> None:
        for events in (None, {}, ["not an object"]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                normalize_full_video_events(
                    {"duration_seconds": 100, "events": events}
                )
        bad = _video_event(1, 2)
        bad["confidence"] = 1.1
        with self.assertRaises(ValueError):
            normalize_full_video_events(
                {"duration_seconds": 100, "events": [bad]}
            )

    def test_well_formed_bad_video_intervals_are_retained_not_repaired(self) -> None:
        result = normalize_full_video_events(
            {
                "duration_seconds": 10,
                "events": [_video_event(8, 7), _video_event(8, 12)],
            }
        )["events"]
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["interval_error"], "start_not_before_end")
        self.assertEqual(result[1]["interval_error"], "outside_supplied_video")
        self.assertEqual((result[1]["start_seconds"], result[1]["end_seconds"]), (8, 12))

    def test_candidate_clip_coordinates_translate_exactly_once(self) -> None:
        result = normalize_candidate_video_events(
            {
                "candidate_windows": {
                    "window_id": 3,
                    "start_seconds": 100,
                    "end_seconds": 130,
                },
                "events": [_video_event(2, 5)],
            }
        )["events"][0]
        self.assertEqual((result["start_seconds"], result["end_seconds"]), (102, 105))
        self.assertEqual(result["window_id"], 3)

    def test_transcript_unknown_and_reversed_ids_remain_invalid(self) -> None:
        result = normalize_transcript_events(
            {
                "duration_seconds": 100,
                "transcript_segments": _segments(),
                "transcript_event_ranges": [
                    _range(0, 1),
                    _range(99, 99),
                    _range(1, 0),
                ],
            }
        )["events"]
        self.assertTrue(result[0]["interval_valid"])
        self.assertEqual((result[0]["start_seconds"], result[0]["end_seconds"]), (10, 30))
        self.assertEqual(result[1]["interval_error"], "unknown_start_segment_id")
        self.assertIsNone(result[1]["start_seconds"])
        self.assertEqual(result[2]["interval_error"], "start_segment_after_end_segment")

    def test_candidates_pad_merge_audit_invalid_and_never_cap(self) -> None:
        result = candidate_segment_ranges_to_windows(
            {
                "duration_seconds": 100,
                "candidate_padding_seconds": 10,
                "transcript_segments": _segments(),
                "candidate_ranges": [
                    _range(0, 0),
                    _range(1, 1),
                    _range(0, 0),
                    _range(99, 99),
                ],
            }
        )
        self.assertEqual(len(result["candidate_windows"]), 1)
        self.assertEqual(
            (
                result["candidate_windows"][0]["start_seconds"],
                result["candidate_windows"][0]["end_seconds"],
            ),
            (0, 40),
        )
        self.assertEqual(result["raw_candidate_range_count"], 4)
        self.assertEqual(result["valid_candidate_range_count"], 2)
        self.assertEqual(result["duplicate_candidate_range_count"], 1)
        self.assertEqual(result["invalid_candidate_range_count"], 1)

    def test_nonfinite_and_backward_transcript_boundaries_fail(self) -> None:
        for segments in (
            [{"segment_id": 0, "start_seconds": math.nan, "end_seconds": 1, "text": "x"}],
            [
                {"segment_id": 0, "start_seconds": 10, "end_seconds": 20, "text": "x"},
                {"segment_id": 1, "start_seconds": 9, "end_seconds": 30, "text": "x"},
            ],
        ):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                candidate_segment_ranges_to_windows(
                    {
                        "duration_seconds": 100,
                        "candidate_padding_seconds": 0,
                        "transcript_segments": segments,
                        "candidate_ranges": [],
                    }
                )


def _segments() -> list[dict[str, object]]:
    return [
        {"segment_id": 0, "start_seconds": 10, "end_seconds": 20, "text": "first"},
        {"segment_id": 1, "start_seconds": 20, "end_seconds": 30, "text": "second"},
    ]


def _range(start: int, end: int) -> dict[str, object]:
    return {
        "start_segment_id": start,
        "end_segment_id": end,
        "confidence": 0.8,
        "evidence": "direct transcript evidence",
    }


def _video_event(start: float, end: float) -> dict[str, object]:
    return {
        "start_minute": 0,
        "start_second": start,
        "end_minute": 0,
        "end_second": end,
        "confidence": 0.9,
        "evidence": "direct audiovisual evidence",
    }


if __name__ == "__main__":
    unittest.main()

