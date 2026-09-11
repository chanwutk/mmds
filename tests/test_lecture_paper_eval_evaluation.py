from __future__ import annotations

import unittest

from scripts.experiments.common import ExperimentDataError
from scripts.lecture_paper_eval.catalog import LECTURES, VERIFIED_3PAIR_PROFILE
from scripts.lecture_paper_eval.evaluation import (
    build_ground_truth,
    evaluate_candidate_windows,
    evaluate_predictions,
)
from scripts.lectures.evaluation import evaluate_predictions as shared_evaluate_predictions


class LecturePaperEvalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = build_ground_truth(
            {
                lecture.lecture_id: {"duration_seconds": 5000.0}
                for lecture in LECTURES
            }
        )

    def test_ground_truth_has_four_positive_pairs_seven_events_and_twelve_negatives(self) -> None:
        self.assertEqual(self.truth["pair_count"], 16)
        self.assertEqual(self.truth["positive_pair_count"], 4)
        self.assertEqual(self.truth["negative_pair_count"], 12)
        self.assertEqual(self.truth["event_count"], 7)
        self.assertEqual(len({(p["lecture_id"], p["query_id"]) for p in self.truth["pairs"]}), 16)

    def test_verified_ground_truth_has_three_positive_pairs_and_four_events(self) -> None:
        profile = VERIFIED_3PAIR_PROFILE
        truth = build_ground_truth(
            {
                lecture.lecture_id: {"duration_seconds": 5000.0}
                for lecture in profile.lectures
            },
            profile=profile,
        )
        self.assertEqual(truth["experiment"], profile.experiment_name)
        self.assertEqual(truth["pair_count"], 3)
        self.assertEqual(truth["positive_pair_count"], 3)
        self.assertEqual(truth["negative_pair_count"], 0)
        self.assertEqual(truth["event_count"], 4)
        self.assertEqual(
            [(pair["lecture_id"], pair["query_id"]) for pair in truth["pairs"]],
            list(profile.pairs),
        )

    def test_primary_evaluation_enriches_boundary_and_presence_diagnostics(self) -> None:
        predictions = [
            _prediction(
                "mit_8_03sc_lecture_03",
                "tone_shatters_glass",
                4428,
                4433,
            ),
            {
                **_prediction(
                    "mit_8_03sc_lecture_07",
                    "heat_device_resonant_sound",
                    1,
                    2,
                ),
                "interval_valid": False,
                "start_seconds": None,
                "end_seconds": None,
            },
        ]
        result = evaluate_predictions(predictions, self.truth)
        primary = result["primary_metrics"]
        self.assertEqual(primary["true_positives"], 1)
        self.assertEqual(primary["false_positives"], 1)
        self.assertEqual(primary["false_negatives"], 6)
        pair = next(
            item
            for item in result["per_pair_by_tiou"]["0.3"]
            if item["lecture_id"] == "mit_8_03sc_lecture_03"
            and item["query_id"] == "tone_shatters_glass"
        )
        match = pair["matches"][0]
        self.assertEqual(match["start_error_seconds"], 1)
        self.assertEqual(match["end_error_seconds"], 1)
        self.assertEqual(match["boundary_mae_seconds"], 1)
        self.assertEqual(result["pair_presence"]["metrics"]["false_positives"], 1)

    def test_four_chladni_events_are_matched_one_to_one(self) -> None:
        predictions = [
            _prediction(
                "mit_8_03sc_lecture_15",
                "speaker_driven_chladni_formation",
                start,
                end,
            )
            for start, end in (
                (4195, 4225),
                (4289, 4300),
                (4338, 4360),
                (4375, 4395),
            )
        ]
        result = evaluate_predictions(predictions, self.truth)
        pair = next(
            item
            for item in result["per_pair_by_tiou"]["0.3"]
            if item["lecture_id"] == "mit_8_03sc_lecture_15"
            and item["query_id"] == "speaker_driven_chladni_formation"
        )
        self.assertEqual(pair["true_positives"], 4)
        self.assertEqual(len(pair["matches"]), 4)
        self.assertEqual(
            {item["truth_index"] for item in pair["matches"]},
            {0, 1, 2, 3},
        )

    def test_candidate_metrics_include_any_half_full_and_mean_coverage(self) -> None:
        rows = []
        for pair in self.truth["pairs"]:
            windows = []
            if (
                pair["lecture_id"] == "mit_8_03sc_lecture_03"
                and pair["query_id"] == "tone_shatters_glass"
            ):
                windows = [{"start_seconds": 4427, "end_seconds": 4429.5}]
            rows.append(
                {
                    "lecture_id": pair["lecture_id"],
                    "query_id": pair["query_id"],
                    "candidate_windows": windows,
                    "raw_candidate_range_count": len(windows),
                    "valid_candidate_range_count": len(windows),
                    "invalid_candidate_range_count": 0,
                    "duplicate_candidate_range_count": 0,
                }
            )
        result = evaluate_candidate_windows(rows, self.truth)
        self.assertEqual(result["candidate_recall_at_any_coverage"], 1 / 7)
        self.assertEqual(result["candidate_recall_at_50_percent_coverage"], 1 / 7)
        self.assertEqual(result["candidate_recall_at_full_coverage"], 0)
        self.assertAlmostEqual(result["mean_truth_event_coverage"], 0.5 / 7)

    def test_primary_threshold_must_be_in_threshold_set(self) -> None:
        with self.assertRaises(ExperimentDataError):
            shared_evaluate_predictions(
                [], self.truth, thresholds=(0.1, 0.5), primary_threshold=0.3
            )


def _prediction(
    lecture_id: str, query_id: str, start: float, end: float
) -> dict[str, object]:
    return {
        "lecture_id": lecture_id,
        "query_id": query_id,
        "method": "test",
        "start_seconds": start,
        "end_seconds": end,
        "interval_valid": True,
        "interval_error": None,
        "confidence": 0.9,
        "evidence": "direct evidence",
    }


if __name__ == "__main__":
    unittest.main()
