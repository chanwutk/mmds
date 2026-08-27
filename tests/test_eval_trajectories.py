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

from examples.support.trajectory_evaluation import (  # noqa: E402
    attribute_agreement,
    canonicalize_class,
    canonicalize_color,
    colors_soft_match,
    endpoints_within_or_tolerance,
    evaluate_trajectories,
    greedy_match_trajectories,
    interval_iou,
    normalize_trajectory_row,
    pair_score,
    timeline_overlap_score,
)


def _traj(
    *,
    vehicle_id: str,
    vehicle_class: str = "sedan",
    color: str = "white",
    subtype: str = "sedan",
    highway2: tuple[float, float] = (0.0, 5.0),
    highway3: tuple[float, float] = (0.0, 5.0),
) -> dict:
    return {
        "vehicle_id": vehicle_id,
        "attributes": {"class": vehicle_class, "color": color, "subtype": subtype},
        "timeline": [
            {
                "camera_id": "cam-i24v-highway2",
                "entered": highway2[0],
                "exited": highway2[1],
            },
            {
                "camera_id": "cam-i24v-highway3",
                "entered": highway3[0],
                "exited": highway3[1],
            },
        ],
        "match_score": 0.9,
    }


class NormalizeTrajectoryTests(unittest.TestCase):
    def test_unwraps_semantic_unnest_vehicles_field(self) -> None:
        row = {
            "camera_id": "cam-i24v-highway2",
            "vehicles": _traj(vehicle_id="v1"),
        }
        normalized = normalize_trajectory_row(row)
        assert normalized is not None
        self.assertEqual(normalized["vehicle_id"], "v1")
        self.assertEqual(len(normalized["timeline"]), 2)

    def test_accepts_bare_udf_trajectory(self) -> None:
        normalized = normalize_trajectory_row(_traj(vehicle_id="join_unique_abc"))
        assert normalized is not None
        self.assertEqual(normalized["vehicle_id"], "join_unique_abc")

    def test_rejects_empty_and_missing_timeline(self) -> None:
        self.assertIsNone(normalize_trajectory_row({}))
        self.assertIsNone(normalize_trajectory_row({"vehicle_id": "x", "timeline": []}))
        self.assertIsNone(normalize_trajectory_row(None))


class IntervalAndTimelineTests(unittest.TestCase):
    def test_interval_iou_identical_and_disjoint(self) -> None:
        self.assertAlmostEqual(interval_iou(0.0, 5.0, 0.0, 5.0), 1.0)
        self.assertAlmostEqual(interval_iou(0.0, 1.0, 2.0, 3.0), 0.0)
        self.assertAlmostEqual(interval_iou(0.0, 4.0, 2.0, 6.0), 2.0 / 6.0)

    def test_endpoints_or_tolerance(self) -> None:
        # enter within 1s, exit way off -> OR passes
        self.assertTrue(
            endpoints_within_or_tolerance(
                0.0, 5.0, 0.5, 3.0, max_endpoint_delta_sec=1.0
            )
        )
        # exit within 1s, enter way off -> OR passes
        self.assertTrue(
            endpoints_within_or_tolerance(
                0.0, 5.0, 2.5, 5.4, max_endpoint_delta_sec=1.0
            )
        )
        # both endpoints off by >1s -> fails
        self.assertFalse(
            endpoints_within_or_tolerance(
                0.0, 5.0, 1.5, 3.0, max_endpoint_delta_sec=1.0
            )
        )

    def test_timeline_overlap_penalizes_missing_camera(self) -> None:
        pred = _traj(vehicle_id="p")
        ref = {
            "vehicle_id": "r",
            "attributes": {"class": "sedan", "color": "white", "subtype": "sedan"},
            "timeline": [
                {
                    "camera_id": "cam-i24v-highway2",
                    "entered": 0.0,
                    "exited": 5.0,
                }
            ],
        }
        # One shared camera at IoU=1, one missing -> mean 0.5
        self.assertAlmostEqual(timeline_overlap_score(pred, ref), 0.5)

    def test_timeline_gate_allows_one_endpoint_within_tolerance(self) -> None:
        # Both cameras: enter within 1s, exit off by >1s -> OR passes; IoU still applied.
        pred = _traj(vehicle_id="p", highway2=(0.0, 5.0), highway3=(0.0, 5.0))
        ref = _traj(vehicle_id="r", highway2=(0.8, 3.0), highway3=(0.8, 3.0))
        score = timeline_overlap_score(pred, ref, max_endpoint_delta_sec=1.0)
        expected = interval_iou(0.0, 5.0, 0.8, 3.0)
        self.assertAlmostEqual(score, expected)

    def test_timeline_gate_rejects_when_both_endpoints_drift(self) -> None:
        pred = _traj(vehicle_id="p", highway2=(0.0, 5.0), highway3=(0.0, 5.0))
        ref = _traj(vehicle_id="r", highway2=(1.5, 3.0), highway3=(1.5, 3.0))
        self.assertAlmostEqual(
            timeline_overlap_score(pred, ref, max_endpoint_delta_sec=1.0), 0.0
        )
        # Disabled gate falls back to pure IoU (nonzero overlap).
        self.assertGreater(
            timeline_overlap_score(pred, ref, max_endpoint_delta_sec=None), 0.0
        )


class AttributeTests(unittest.TestCase):
    def test_canonicalize_class_aliases(self) -> None:
        self.assertEqual(canonicalize_class("Car"), "sedan")
        self.assertEqual(canonicalize_class("pickup truck"), "truck")
        self.assertEqual(canonicalize_class("SUV"), "suv")

    def test_canonicalize_color_families(self) -> None:
        self.assertEqual(canonicalize_color("Blue"), "cool_dark")
        self.assertEqual(canonicalize_color("grey"), "cool_dark")
        self.assertEqual(canonicalize_color("silver"), "cool_dark")
        self.assertEqual(canonicalize_color("black"), "cool_dark")
        self.assertEqual(canonicalize_color("red"), "warm_red")
        self.assertEqual(canonicalize_color("brown"), "warm_red")
        self.assertEqual(canonicalize_color("white"), "white")
        self.assertTrue(colors_soft_match("blue", "gray"))
        self.assertTrue(colors_soft_match("red", "brown"))
        self.assertFalse(colors_soft_match("white", "black"))
        self.assertFalse(colors_soft_match("red", "blue"))

    def test_attribute_agreement_exact_vs_soft(self) -> None:
        pred = _traj(vehicle_id="p", vehicle_class="car", color="White", subtype="sedan")
        ref = _traj(vehicle_id="r", vehicle_class="sedan", color="white", subtype="sedan")
        exact, soft = attribute_agreement(pred, ref)
        self.assertAlmostEqual(exact, 2.0 / 3.0)  # color + subtype; class string differs
        self.assertAlmostEqual(soft, 1.0)  # class alias + color + subtype

    def test_soft_color_family_counts_as_match(self) -> None:
        pred = _traj(vehicle_id="p", color="blue")
        ref = _traj(vehicle_id="r", color="silver")
        exact, soft = attribute_agreement(pred, ref)
        self.assertAlmostEqual(exact, 2.0 / 3.0)  # class + subtype only
        self.assertAlmostEqual(soft, 1.0)  # color family + class + subtype

    def test_soft_score_zero_when_color_not_close(self) -> None:
        pred = _traj(vehicle_id="p", color="white")
        ref = _traj(vehicle_id="r", color="red")
        exact, soft = attribute_agreement(pred, ref)
        self.assertAlmostEqual(exact, 2.0 / 3.0)  # class + subtype
        self.assertAlmostEqual(soft, 0.0)  # color gate fails entire soft score


class MatchingAndEvalTests(unittest.TestCase):
    def test_pair_score_perfect_match(self) -> None:
        traj = _traj(vehicle_id="a")
        combined, timeline, exact, soft = pair_score(traj, traj)
        self.assertAlmostEqual(timeline, 1.0)
        self.assertAlmostEqual(exact, 1.0)
        self.assertAlmostEqual(soft, 1.0)
        self.assertAlmostEqual(combined, 1.0)

    def test_greedy_match_is_one_to_one(self) -> None:
        predictions = [
            _traj(vehicle_id="p1", color="white", highway2=(0.0, 5.0)),
            _traj(vehicle_id="p2", color="black", highway2=(0.0, 5.0)),
        ]
        references = [
            _traj(vehicle_id="r1", color="white", highway2=(0.0, 5.0)),
            _traj(vehicle_id="r2", color="black", highway2=(0.0, 5.0)),
        ]
        matches = greedy_match_trajectories(predictions, references, min_score=0.2)
        self.assertEqual(len(matches), 2)
        paired = {(m.pred_index, m.ref_index) for m in matches}
        self.assertEqual(paired, {(0, 0), (1, 1)})

    def test_evaluate_perfect_alignment(self) -> None:
        rows = [
            _traj(vehicle_id="a", color="gray"),
            _traj(vehicle_id="b", color="black", highway2=(2.0, 5.0), highway3=(2.0, 5.0)),
        ]
        report = evaluate_trajectories(rows, rows, min_score=0.2)
        self.assertEqual(report.true_positives, 2)
        self.assertEqual(report.false_positives, 0)
        self.assertEqual(report.false_negatives, 0)
        self.assertAlmostEqual(report.precision, 1.0)
        self.assertAlmostEqual(report.recall, 1.0)
        self.assertAlmostEqual(report.f1, 1.0)

    def test_evaluate_extra_prediction_is_false_positive(self) -> None:
        predictions = [
            _traj(vehicle_id="p1", color="white"),
            _traj(vehicle_id="p2", color="red", highway2=(1.0, 2.0), highway3=(1.0, 2.0)),
        ]
        references = [_traj(vehicle_id="r1", color="white")]
        report = evaluate_trajectories(predictions, references, min_score=0.2)
        self.assertEqual(report.true_positives, 1)
        self.assertEqual(report.false_positives, 1)
        self.assertEqual(report.false_negatives, 0)
        self.assertAlmostEqual(report.precision, 0.5)
        self.assertAlmostEqual(report.recall, 1.0)

    def test_evaluate_missing_prediction_is_false_negative(self) -> None:
        predictions = [_traj(vehicle_id="p1", color="white")]
        references = [
            _traj(vehicle_id="r1", color="white"),
            _traj(vehicle_id="r2", color="black"),
        ]
        report = evaluate_trajectories(predictions, references, min_score=0.2)
        self.assertEqual(report.true_positives, 1)
        self.assertEqual(report.false_positives, 0)
        self.assertEqual(report.false_negatives, 1)

    def test_evaluate_unwraps_semantic_rows(self) -> None:
        pred = [_traj(vehicle_id="join_1", vehicle_class="sedan", color="silver")]
        ref = [{"vehicles": _traj(vehicle_id="v1", vehicle_class="car", color="silver")}]
        report = evaluate_trajectories(pred, ref, min_score=0.2)
        self.assertEqual(report.true_positives, 1)
        self.assertGreater(report.mean_attribute_soft, report.mean_attribute_exact)

    def test_empty_inputs(self) -> None:
        report = evaluate_trajectories([], [])
        self.assertEqual(report.true_positives, 0)
        self.assertAlmostEqual(report.precision, 0.0)
        self.assertAlmostEqual(report.recall, 0.0)
        self.assertAlmostEqual(report.f1, 0.0)

    def test_low_score_pairs_are_not_matched(self) -> None:
        predictions = [_traj(vehicle_id="p", color="white", highway2=(0.0, 1.0), highway3=(0.0, 1.0))]
        references = [
            _traj(vehicle_id="r", color="black", highway2=(4.0, 5.0), highway3=(4.0, 5.0))
        ]
        report = evaluate_trajectories(predictions, references, min_score=0.9)
        self.assertEqual(report.true_positives, 0)
        self.assertEqual(report.false_positives, 1)
        self.assertEqual(report.false_negatives, 1)


if __name__ == "__main__":
    unittest.main()
