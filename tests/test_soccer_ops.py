import unittest

from udfs.soccer_ops import has_ball_activity, is_goal_candidate


def _make_detections(ball_frame_count: int) -> list[dict]:
    """Build a detections list as produced by the MMDS Detect operator."""
    bboxes = [
        {"frame_idx": i * 10, "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.8}
        for i in range(ball_frame_count)
    ]
    result = []
    if ball_frame_count > 0:
        result.append({"type": "sports ball", "bboxes": bboxes})
    result.append({"type": "person", "bboxes": [{"frame_idx": 0, "bbox": [0, 0, 1, 1], "confidence": 0.9}]})
    return result


class IsGoalCandidateTests(unittest.TestCase):
    """Tests for the transcript-based text gate used in the optimized pipeline."""

    def test_goal_description_passes(self) -> None:
        row = {"transcript": "He unleashes a shot which goes into the top right corner, ricocheting in off the post."}
        self.assertTrue(is_goal_candidate(row))

    def test_into_net_passes(self) -> None:
        row = {"transcript": "The ball flies into the net. What a finish!"}
        self.assertTrue(is_goal_candidate(row))

    def test_scores_passes(self) -> None:
        row = {"transcript": "Ramsey scores to give Arsenal the lead."}
        self.assertTrue(is_goal_candidate(row))

    def test_equalizer_passes(self) -> None:
        row = {"transcript": "A stunning equalizer brings the teams level."}
        self.assertTrue(is_goal_candidate(row))

    def test_miss_description_fails(self) -> None:
        row = {"transcript": "He sends the ball high over the crossbar. Very poor attempt."}
        self.assertFalse(is_goal_candidate(row))

    def test_save_description_fails(self) -> None:
        row = {"transcript": "The goalkeeper stops it with a miraculous save. The shot was blocked."}
        self.assertFalse(is_goal_candidate(row))

    def test_tactical_commentary_fails(self) -> None:
        row = {"transcript": "A long ball fails to find a teammate across the pitch. Possession retained."}
        self.assertFalse(is_goal_candidate(row))

    def test_empty_transcript_passes(self) -> None:
        # No caption data → cannot rule out goal → must forward to Gemini
        self.assertTrue(is_goal_candidate({"transcript": ""}))

    def test_missing_transcript_passes(self) -> None:
        self.assertTrue(is_goal_candidate({}))

    def test_none_transcript_passes(self) -> None:
        self.assertTrue(is_goal_candidate({"transcript": None}))

    def test_case_insensitive(self) -> None:
        row = {"transcript": "INTO THE NET! GOAL!"}
        self.assertTrue(is_goal_candidate(row))


class HasBallActivityTests(unittest.TestCase):
    """Tests for the YOLOE-gate UDF used in the goal-detection pipeline."""

    # --- Detect operator output (detections field) ---

    def test_ball_in_two_frames_passes(self) -> None:
        row = {"detections": _make_detections(2)}
        self.assertTrue(has_ball_activity(row))

    def test_ball_in_five_frames_passes(self) -> None:
        row = {"detections": _make_detections(5)}
        self.assertTrue(has_ball_activity(row))

    def test_ball_in_one_frame_fails(self) -> None:
        row = {"detections": _make_detections(1)}
        self.assertFalse(has_ball_activity(row))

    def test_no_ball_class_fails(self) -> None:
        row = {"detections": [{"type": "person", "bboxes": [{"frame_idx": 0, "bbox": [], "confidence": 0.9}]}]}
        self.assertFalse(has_ball_activity(row))

    def test_empty_detections_fails(self) -> None:
        self.assertFalse(has_ball_activity({"detections": []}))

    def test_missing_detections_field_fails(self) -> None:
        self.assertFalse(has_ball_activity({}))

    # --- detect_ball_sampled UDF output (ball_detected field) ---

    def test_ball_detected_true_passes(self) -> None:
        row = {"ball_detected": True, "ball_frame_count": 3, "sampled_frame_count": 30}
        self.assertTrue(has_ball_activity(row))

    def test_ball_detected_false_fails(self) -> None:
        row = {"ball_detected": False, "ball_frame_count": 0, "sampled_frame_count": 30}
        self.assertFalse(has_ball_activity(row))

    def test_ball_detected_key_takes_priority_over_detections(self) -> None:
        # ball_detected=False should override a non-empty detections list
        row = {
            "ball_detected": False,
            "detections": _make_detections(5),
        }
        self.assertFalse(has_ball_activity(row))


if __name__ == "__main__":
    unittest.main()
