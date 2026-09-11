from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mmds import StaticPromptExecutor, render_query
from mmds.model import PromptSpec, ResolvedPrompt

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json_object,
    load_jsonl_objects,
    sha256_file,
)
from scripts.experiments.gemini_runtime import (
    ApiCallRecorder,
    CachingGeminiPromptExecutor,
    cache_key,
)
from scripts.experiments.usage import aggregate_api_usage
from scripts.experiments.whisper import (
    FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS,
    NORMALIZATION_CONTRACT_VERSION,
    normalize_transcription,
)
from scripts.lectures.catalog import (
    LECTURES,
    WHISPER_MODEL,
    LectureSource,
    source_catalog_payload,
)
from scripts.lectures.download import download_lectures
from scripts.lectures.evaluation import (
    evaluate_binary_decisions,
    evaluate_candidate_windows,
    evaluate_predictions,
    match_intervals,
    temporal_iou,
)
from scripts.lectures.experiment import (
    FORBIDDEN_INPUT_KEYS,
    V2_CONFIG_FILENAME,
    V3_CONFIG_FILENAME,
    V4_CONFIG_FILENAME,
    V5_CONFIG_FILENAME,
    compare_results,
    compare_v2_results,
    compare_v3_results,
    compare_v4_results,
    compare_v5_results,
    experiment_directory,
    freeze_ground_truth,
    initialize_ground_truth_review,
    materialize_clips,
    materialize_v3_proposal_clips,
    prepare_experiment,
    prepare_v2,
    prepare_v3,
    prepare_v4,
    prepare_v5,
    run_candidates,
    run_naive,
    run_naive_v2_binary,
    run_naive_v2_localization,
    run_optimized,
    run_optimized_materialized,
    run_optimized_v2_binary,
    run_optimized_v2_localization,
    run_transcript_only,
    run_v3_refinement,
    run_v3_transcript_proposals,
    run_v4_predicate_grounding,
    run_v4_query_conditions,
    run_v5_predicate_grounding,
    run_v5_query_conditions,
    status,
)
from scripts.lectures.materialize import (
    MATERIALIZATION_CONTRACT,
    ffmpeg_materialize_clip,
    materialize_candidate_clips,
    validate_materialized_clip_stage,
)
from scripts.lectures.plans import (
    BINARY_VERIFICATION_SCHEMA,
    BINARY_VERIFIER_PROMPT,
    EPISODE_LOCALIZATION_PROMPT,
    EPISODE_SCHEMA,
    TRANSCRIPT_EPISODE_PROPOSAL_PROMPT,
    TRANSCRIPT_GROUNDED_REFINEMENT_PROMPT,
    TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA,
    PREDICATE_GROUNDING_PROMPT,
    PREDICATE_GROUNDING_SCHEMA,
    QUERY_CONDITION_PROMPT,
    QUERY_CONDITION_SCHEMA,
    ROLE_AWARE_PREDICATE_GROUNDING_PROMPT,
    ROLE_AWARE_QUERY_CONDITION_PROMPT,
    ROLE_AWARE_QUERY_CONDITION_SCHEMA,
    VERIFIER_PROMPT,
    build_candidate_plan,
    build_materialized_v2_binary_plan,
    build_materialized_v2_localization_plan,
    build_materialized_verification_plan,
    build_naive_plan,
    build_naive_v2_binary_plan,
    build_naive_v2_localization_plan,
    build_transcript_only_plan,
    build_v3_refinement_plan,
    build_v3_transcript_proposal_plan,
    build_v4_predicate_grounding_plan,
    build_v4_query_condition_plan,
    build_v5_predicate_grounding_plan,
    build_v5_query_condition_plan,
    build_verification_plan,
)
from scripts.lectures.transcribe import transcribe_lectures
from udfs.lecture_ops import (
    attach_candidate_video_view,
    candidate_segment_ranges_to_windows,
    is_event_present,
    normalize_naive_episodes,
    normalize_transcript_events,
    normalize_transcript_grounded_refinement,
    normalize_condition_evidence_to_event,
    normalize_required_conditions,
    normalize_role_aware_condition_evidence_to_event,
    normalize_role_aware_required_conditions,
    normalize_verified_events,
    normalize_verified_episodes,
    transcript_episode_proposals_to_windows,
    validate_binary_verification,
)


POSITIVE_PAIRS = {
    (LECTURES[0].lecture_id, "tone_shatters_glass"),
    (LECTURES[1].lecture_id, "heat_device_resonant_sound"),
    (LECTURES[2].lecture_id, "measure_speed_of_sound"),
}


class LectureUdfTests(unittest.TestCase):
    def test_binary_verification_is_strict_and_never_coerces_truthiness(self) -> None:
        row = {
            "verification": {
                "event_present": True,
                "confidence": 0.75,
                "evidence": "visible outcome",
            }
        }
        self.assertEqual(validate_binary_verification(row), row)
        self.assertTrue(is_event_present(row))

        invalid = (
            {"event_present": "yes", "confidence": 1, "evidence": "x"},
            {"event_present": 1, "confidence": 1, "evidence": "x"},
            {"event_present": True, "confidence": math.nan, "evidence": "x"},
            {"event_present": True, "confidence": 1.1, "evidence": "x"},
            {"event_present": True, "confidence": 1, "evidence": "  "},
        )
        for verification in invalid:
            with self.subTest(verification=verification), self.assertRaises(
                ValueError
            ):
                validate_binary_verification({"verification": verification})

    def test_complete_episode_normalization_requires_a_nonempty_response(self) -> None:
        episode = {
            "start_minute": 0,
            "start_second": 2,
            "end_minute": 0,
            "end_second": 8,
            "confidence": 1,
            "evidence": "complete physical activity",
        }
        naive = normalize_naive_episodes(
            {"duration_seconds": 100, "episodes": [episode]}
        )
        self.assertEqual(
            (naive["events"][0]["start_seconds"], naive["events"][0]["end_seconds"]),
            (2.0, 8.0),
        )
        optimized = normalize_verified_episodes(
            {
                "candidate_windows": {
                    "window_id": 4,
                    "start_seconds": 20,
                    "end_seconds": 40,
                },
                "episodes": [episode],
            }
        )
        self.assertEqual(
            (
                optimized["events"][0]["start_seconds"],
                optimized["events"][0]["end_seconds"],
                optimized["events"][0]["window_id"],
            ),
            (22.0, 28.0, 4),
        )
        with self.assertRaises(ValueError):
            normalize_naive_episodes({"duration_seconds": 100, "episodes": []})

    def test_candidate_ranges_are_grounded_padded_merged_and_invalid_preserved(self) -> None:
        row = {
            "duration_seconds": 100.0,
            "candidate_padding_seconds": 10.0,
            "transcript_segments": _segments(),
            "candidate_ranges": [
                _range(0, 0, confidence=0.7),
                _range(1, 1, confidence=0.8),
                _range(1, 1, confidence=0.1),
                _range(True, 1),
                _range(9, 9),
            ],
        }
        result = candidate_segment_ranges_to_windows(row)
        self.assertEqual(len(result["candidate_windows"]), 1)
        window = result["candidate_windows"][0]
        self.assertEqual((window["start_seconds"], window["end_seconds"]), (0.0, 40.0))
        self.assertEqual(window["window_id"], 0)
        self.assertEqual(len(window["segment_ranges"]), 2)
        self.assertEqual(
            [item["range_error"] for item in result["invalid_candidate_ranges"]],
            ["invalid_start_segment_id", "unknown_start_segment_id"],
        )

    def test_v3_episode_proposals_remain_separate_even_when_context_overlaps(self) -> None:
        result = transcript_episode_proposals_to_windows(
            {
                "duration_seconds": 100.0,
                "candidate_padding_seconds": 20.0,
                "transcript_segments": _segments(),
                "episode_proposals": [
                    _range(0, 0, confidence=0.7),
                    _range(1, 1, confidence=0.8),
                    _range(1, 1, confidence=0.1),
                    _range(9, 9),
                ],
            }
        )
        self.assertEqual(len(result["candidate_windows"]), 2)
        self.assertEqual(
            [window["window_id"] for window in result["candidate_windows"]],
            [0, 1],
        )
        self.assertEqual(
            [len(window["segment_ranges"]) for window in result["candidate_windows"]],
            [1, 1],
        )
        self.assertLess(
            result["candidate_windows"][1]["start_seconds"],
            result["candidate_windows"][0]["end_seconds"],
        )
        self.assertEqual(
            result["invalid_episode_proposals"][0]["range_error"],
            "unknown_start_segment_id",
        )

    def test_v3_refinement_uses_only_allowed_transcript_boundaries(self) -> None:
        base = {
            "duration_seconds": 100.0,
            "transcript_segments": _segments(),
            "transcript_context_segment_ids": [0, 1],
            "candidate_windows": {
                "window_id": 4,
                "start_seconds": 0.0,
                "end_seconds": 40.0,
            },
        }
        accepted = normalize_transcript_grounded_refinement(
            {**base, "episode_refinements": [_range(0, 1)]}
        )
        event = accepted["events"][0]
        self.assertEqual((event["start_seconds"], event["end_seconds"]), (10.0, 30.0))
        self.assertTrue(event["interval_valid"])
        self.assertEqual(event["window_id"], 4)

        rejected = normalize_transcript_grounded_refinement(
            {**base, "episode_refinements": []}
        )
        self.assertEqual(rejected, {"events": []})

        outside = normalize_transcript_grounded_refinement(
            {**base, "episode_refinements": [_range(0, 2)]}
        )["events"][0]
        self.assertFalse(outside["interval_valid"])
        self.assertEqual(
            outside["interval_error"], "segment_id_outside_allowed_context"
        )

        with self.assertRaises(ValueError):
            normalize_transcript_grounded_refinement(
                {**base, "episode_refinements": [_range(0, 0), _range(1, 1)]}
            )

    def test_v4_conditions_receive_deterministic_ids(self) -> None:
        result = normalize_required_conditions(
            {
                "condition_descriptions": [
                    "  A visible device is heated. ",
                    "A sustained resonant sound is audible.",
                ]
            }
        )
        self.assertEqual(
            result["required_conditions"],
            [
                {
                    "condition_id": "condition_0",
                    "description": "A visible device is heated.",
                },
                {
                    "condition_id": "condition_1",
                    "description": "A sustained resonant sound is audible.",
                },
            ],
        )
        for descriptions in ([], ["same", " same "]):
            with self.subTest(descriptions=descriptions), self.assertRaises(ValueError):
                normalize_required_conditions(
                    {"condition_descriptions": descriptions}
                )

    def test_v4_requires_every_condition_and_derives_minimal_span(self) -> None:
        base = {
            "duration_seconds": 100.0,
            "transcript_segments": _segments(),
            "transcript_context_segment_ids": [0, 1],
            "candidate_windows": {
                "window_id": 7,
                "start_seconds": 0.0,
                "end_seconds": 40.0,
            },
            "required_conditions": [
                {"condition_id": "condition_0", "description": "first"},
                {"condition_id": "condition_1", "description": "second"},
            ],
        }
        accepted = normalize_condition_evidence_to_event(
            {
                **base,
                "condition_evidence": [
                    {
                        "condition_id": "condition_0",
                        **_range(1, 1, confidence=0.8),
                    },
                    {
                        "condition_id": "condition_1",
                        **_range(0, 0, confidence=0.9),
                    },
                ],
            }
        )["events"][0]
        self.assertEqual(
            (
                accepted["start_seconds"],
                accepted["end_seconds"],
                accepted["start_segment_id"],
                accepted["end_segment_id"],
            ),
            (10.0, 30.0, 0, 1),
        )
        self.assertEqual(accepted["confidence"], 0.8)
        self.assertTrue(accepted["interval_valid"])

        missing = normalize_condition_evidence_to_event(
            {
                **base,
                "condition_evidence": [
                    {"condition_id": "condition_0", **_range(0, 0)}
                ],
            }
        )
        self.assertEqual(missing, {"events": []})

        invalid = normalize_condition_evidence_to_event(
            {
                **base,
                "condition_evidence": [
                    {"condition_id": "not_required", **_range(0, 0)}
                ],
            }
        )["events"][0]
        self.assertFalse(invalid["interval_valid"])
        self.assertEqual(invalid["interval_error"], "unknown_condition_id")

    def test_v5_gate_evidence_is_required_but_cannot_widen_timestamp(self) -> None:
        normalized = normalize_role_aware_required_conditions(
            {
                "condition_specs": [
                    {"description": "necessary context", "role": "gate"},
                    {"description": "target occurrence", "role": "anchor"},
                ]
            }
        )
        self.assertEqual(
            [condition["role"] for condition in normalized["required_conditions"]],
            ["gate", "anchor"],
        )
        base = {
            "duration_seconds": 100.0,
            "transcript_segments": _segments(),
            "transcript_context_segment_ids": [0, 1],
            "candidate_windows": {
                "window_id": 2,
                "start_seconds": 0.0,
                "end_seconds": 40.0,
            },
            **normalized,
        }
        result = normalize_role_aware_condition_evidence_to_event(
            {
                **base,
                "condition_evidence": [
                    {"condition_id": "condition_0", **_range(0, 0)},
                    {"condition_id": "condition_1", **_range(1, 1)},
                ],
            }
        )["events"][0]
        self.assertEqual(
            (
                result["start_seconds"],
                result["end_seconds"],
                result["start_segment_id"],
                result["end_segment_id"],
            ),
            (25.0, 30.0, 1, 1),
        )
        self.assertEqual(
            result["timestamp_source"],
            "role_aware_predicate_grounded_whisper_segment_boundaries",
        )

        missing_gate = normalize_role_aware_condition_evidence_to_event(
            {
                **base,
                "condition_evidence": [
                    {"condition_id": "condition_1", **_range(1, 1)}
                ],
            }
        )
        self.assertEqual(missing_gate, {"events": []})

        with self.assertRaises(ValueError):
            normalize_role_aware_required_conditions(
                {
                    "condition_specs": [
                        {"description": "only a gate", "role": "gate"}
                    ]
                }
            )

    def test_candidate_video_view_preserves_frozen_media_metadata(self) -> None:
        result = attach_candidate_video_view(
            {
                "video": {
                    "type": "Video",
                    "path": "/tmp/lecture.mp4",
                    "fps": 1.0,
                    "sha256": "a" * 64,
                    "mime_type": "video/mp4",
                },
                "candidate_windows": {"start_seconds": 12.5, "end_seconds": 30.0},
            }
        )
        self.assertEqual(
            result["candidate_video"],
            {
                "type": "VideoView",
                "source": "/tmp/lecture.mp4",
                "start": 12.5,
                "end": 30.0,
                "fps": 1.0,
                "sha256": "a" * 64,
                "mime_type": "video/mp4",
            },
        )

    def test_video_contract_preserves_outside_clip_as_invalid(self) -> None:
        result = normalize_verified_events(
            {
                "candidate_windows": {
                    "window_id": 4,
                    "start_seconds": 10.0,
                    "end_seconds": 20.0,
                },
                "events": [
                    {
                        "start_minute": 0,
                        "start_second": 1,
                        "end_minute": 0,
                        "end_second": 15,
                        "confidence": 1,
                        "evidence": "too long",
                    }
                ],
            }
        )["events"][0]
        self.assertFalse(result["interval_valid"])
        self.assertEqual(result["interval_error"], "outside_supplied_clip")
        self.assertEqual((result["start_seconds"], result["end_seconds"]), (11.0, 25.0))

    def test_transcript_contract_never_invents_or_repairs_segment_ids(self) -> None:
        events = normalize_transcript_events(
            {
                "duration_seconds": 100.0,
                "transcript_segments": _segments(),
                "transcript_event_ranges": [
                    _range(0, 1),
                    _range(99, 99),
                    _range(1, 0),
                ],
            }
        )["events"]
        self.assertTrue(events[0]["interval_valid"])
        self.assertEqual((events[0]["start_seconds"], events[0]["end_seconds"]), (10.0, 30.0))
        self.assertFalse(events[1]["interval_valid"])
        self.assertIsNone(events[1]["start_seconds"])
        self.assertEqual(events[2]["interval_error"], "start_segment_after_end_segment")

    def test_nonfinite_segment_boundaries_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            candidate_segment_ranges_to_windows(
                {
                    "duration_seconds": 100,
                    "candidate_padding_seconds": 30,
                    "transcript_segments": [
                        {
                            "segment_id": 0,
                            "start_seconds": math.nan,
                            "end_seconds": 1,
                            "text": "bad",
                        }
                    ],
                    "candidate_ranges": [],
                }
            )

    def test_chronologically_ordered_segment_overlap_is_preserved(self) -> None:
        result = candidate_segment_ranges_to_windows(
            {
                "duration_seconds": 100,
                "candidate_padding_seconds": 0,
                "transcript_segments": [
                    {
                        "segment_id": 0,
                        "start_seconds": 10,
                        "end_seconds": 20,
                        "text": "first",
                    },
                    {
                        "segment_id": 1,
                        "start_seconds": 19.5,
                        "end_seconds": 30,
                        "text": "second",
                    },
                ],
                "candidate_ranges": [_range(0, 1)],
            }
        )
        self.assertEqual(
            (
                result["candidate_windows"][0]["start_seconds"],
                result["candidate_windows"][0]["end_seconds"],
            ),
            (10.0, 30.0),
        )

    def test_backward_segment_boundary_still_fails_closed(self) -> None:
        for second in (
            {"segment_id": 1, "start_seconds": 9, "end_seconds": 30, "text": "x"},
            {"segment_id": 1, "start_seconds": 15, "end_seconds": 19, "text": "x"},
        ):
            with self.subTest(second=second), self.assertRaises(ValueError):
                candidate_segment_ranges_to_windows(
                    {
                        "duration_seconds": 100,
                        "candidate_padding_seconds": 0,
                        "transcript_segments": [
                            {
                                "segment_id": 0,
                                "start_seconds": 10,
                                "end_seconds": 20,
                                "text": "first",
                            },
                            second,
                        ],
                        "candidate_ranges": [],
                    }
                )


class LectureClipMaterializationTests(unittest.TestCase):
    def test_ffmpeg_rebases_streams_without_post_filter_timestamp_shift(self) -> None:
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stderr": ""},
        )()
        with (
            patch(
                "scripts.experiments.clip_materialization.subprocess.run",
                return_value=completed,
            ) as run,
            patch(
                "scripts.experiments.clip_materialization.probe_materialized_clip",
                return_value=_clip_media(10.0),
            ),
        ):
            ffmpeg_materialize_clip(
                Path("source.mp4"),
                Path("clip.mp4"),
                20.0,
                30.0,
            )

        command = run.call_args.args[0]
        self.assertIn("setpts=PTS-STARTPTS", command)
        self.assertIn("asetpts=PTS-STARTPTS", command)
        self.assertNotIn("-avoid_negative_ts", command)

    def test_materialization_creates_zero_origin_content_addressed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            stage_directory = root / "materialized_clips"
            source = root / "lecture.mp4"
            source.write_bytes(b"source-video")
            atomic_write_jsonl(
                candidate_path,
                [_candidate_row(source, start=12.5, end=22.5)],
            )

            result = materialize_candidate_clips(
                candidate_path,
                stage_directory,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_clip_probe,
            )

            self.assertEqual(result["manifest"]["clip_count"], 1)
            self.assertEqual(
                result["manifest"]["materialization_contract"],
                MATERIALIZATION_CONTRACT,
            )
            self.assertEqual(result["stage"]["materialized_clip_count"], 1)
            self.assertEqual(result["stage"]["reused_clip_count"], 0)
            row = result["rows"][0]
            self.assertEqual(row["candidate_video"]["type"], "Video")
            self.assertNotIn("start", row["candidate_video"])
            self.assertNotIn("end", row["candidate_video"])
            self.assertEqual(
                row["materialized_video_context"],
                {
                    "lecture_id": "lecture",
                    "duration_seconds": 10.0,
                    "timeline_origin_seconds": 0.0,
                    "media_extent": "standalone_candidate_clip",
                },
            )
            self.assertEqual(
                (
                    row["candidate_windows"]["start_seconds"],
                    row["candidate_windows"]["end_seconds"],
                ),
                (12.5, 22.5),
            )
            validate_materialized_clip_stage(
                stage_directory,
                media_probe=_fake_clip_probe,
            )

    def test_materialization_rejects_duration_drift_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            stage_directory = root / "materialized_clips"
            source = root / "lecture.mp4"
            source.write_bytes(b"source-video")
            atomic_write_jsonl(
                candidate_path,
                [_candidate_row(source, start=10.0, end=20.0)],
            )

            def wrong_duration(
                source_path: Path,
                destination: Path,
                start_seconds: float,
                end_seconds: float,
            ) -> dict[str, object]:
                destination.write_text("9.0", encoding="utf-8")
                return _clip_media(9.0)

            with self.assertRaises(ExperimentDataError):
                materialize_candidate_clips(
                    candidate_path,
                    stage_directory,
                    clip_processor=wrong_duration,
                    media_probe=_fake_clip_probe,
                )
            self.assertFalse(any((stage_directory / "clips").rglob("*.mp4")))

    def test_materialization_rejects_nonzero_container_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            stage_directory = root / "materialized_clips"
            source = root / "lecture.mp4"
            source.write_bytes(b"source-video")
            atomic_write_jsonl(candidate_path, [_candidate_row(source, start=0, end=10)])

            def nonzero_origin(
                source_path: Path,
                destination: Path,
                start_seconds: float,
                end_seconds: float,
            ) -> dict[str, object]:
                destination.write_text("10.0", encoding="utf-8")
                return {**_clip_media(10.0), "start_time_seconds": 1.0}

            with self.assertRaises(ExperimentDataError):
                materialize_candidate_clips(
                    candidate_path,
                    stage_directory,
                    clip_processor=nonzero_origin,
                    media_probe=_fake_clip_probe,
                )

    def test_materialized_clip_corruption_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            stage_directory = root / "materialized_clips"
            source = root / "lecture.mp4"
            source.write_bytes(b"source-video")
            atomic_write_jsonl(candidate_path, [_candidate_row(source, start=0, end=10)])
            result = materialize_candidate_clips(
                candidate_path,
                stage_directory,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_clip_probe,
            )
            Path(result["rows"][0]["candidate_video"]["path"]).write_bytes(b"changed")
            with self.assertRaises(ExperimentDataError):
                validate_materialized_clip_stage(
                    stage_directory,
                    media_probe=_fake_clip_probe,
                )

    def test_materialization_resumes_from_content_checked_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            stage_directory = root / "materialized_clips"
            source = root / "lecture.mp4"
            source.write_bytes(b"source-video")
            row = _candidate_row(source, start=0, end=10)
            row["candidate_windows"].append(
                {
                    "window_id": 1,
                    "start_seconds": 20.0,
                    "end_seconds": 30.0,
                    "duration_seconds": 10.0,
                    "unpadded_start_seconds": 20.0,
                    "unpadded_end_seconds": 30.0,
                    "segment_ranges": [],
                }
            )
            atomic_write_jsonl(candidate_path, [row])
            calls = 0

            def fail_second_clip(
                source_path: Path,
                destination: Path,
                start_seconds: float,
                end_seconds: float,
            ) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ExperimentDataError("injected clip failure")
                return _fake_clip_processor(
                    source_path,
                    destination,
                    start_seconds,
                    end_seconds,
                )

            with self.assertRaises(ExperimentDataError):
                materialize_candidate_clips(
                    candidate_path,
                    stage_directory,
                    clip_processor=fail_second_clip,
                    media_probe=_fake_clip_probe,
                )
            resumed = materialize_candidate_clips(
                candidate_path,
                stage_directory,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_clip_probe,
            )
            self.assertEqual(resumed["stage"]["reused_clip_count"], 1)
            self.assertEqual(resumed["stage"]["materialized_clip_count"], 1)
            self.assertEqual(resumed["manifest"]["clip_count"], 2)


class LectureEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = {
            "pairs": [
                {
                    "lecture_id": "lecture",
                    "query_id": "positive",
                    "duration_seconds": 100,
                    "events": [
                        {"start_seconds": 0, "end_seconds": 10},
                        {"start_seconds": 10, "end_seconds": 20},
                    ],
                },
                {
                    "lecture_id": "lecture",
                    "query_id": "negative",
                    "duration_seconds": 100,
                    "events": [],
                },
            ]
        }

    def test_temporal_iou(self) -> None:
        self.assertEqual(
            temporal_iou(
                {"start_seconds": 0, "end_seconds": 10},
                {"start_seconds": 5, "end_seconds": 15},
            ),
            1 / 3,
        )

    def test_matching_maximizes_cardinality_before_total_iou(self) -> None:
        predictions = [
            _prediction("positive", 0, 20),
            _prediction("positive", 0, 10),
        ]
        matches = match_intervals(
            predictions, self.truth["pairs"][0]["events"], threshold=0.5
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {(item["prediction_index"], item["truth_index"]) for item in matches},
            {(0, 1), (1, 0)},
        )

    def test_all_methods_use_same_thresholds_and_invalid_is_false_positive(self) -> None:
        predictions = [
            _prediction("positive", 0, 20),
            {
                **_prediction("negative", 0, 1),
                "interval_valid": False,
                "start_seconds": None,
                "end_seconds": None,
            },
        ]
        evaluation = evaluate_predictions(predictions, self.truth, thresholds=(0.3, 0.5))
        self.assertEqual(set(evaluation["metrics_by_tiou"]), {"0.3", "0.5"})
        at_half = evaluation["metrics_by_tiou"]["0.5"]
        self.assertEqual(at_half["true_positives"], 1)
        self.assertEqual(at_half["false_positives"], 1)
        self.assertEqual(at_half["false_negatives"], 1)

    def test_deduplication_prefers_confidence_but_keeps_invalid(self) -> None:
        predictions = [
            _prediction("positive", 0, 10, confidence=0.2),
            _prediction("positive", 0.1, 10.1, confidence=0.9),
            {
                **_prediction("negative", 0, 1),
                "interval_valid": False,
                "start_seconds": None,
                "end_seconds": None,
            },
        ]
        evaluation = evaluate_predictions(predictions, self.truth, thresholds=(0.5,))
        per_pair = evaluation["per_pair_by_tiou"]["0.5"]
        positive = next(item for item in per_pair if item["query_id"] == "positive")
        negative = next(item for item in per_pair if item["query_id"] == "negative")
        self.assertEqual(positive["prediction_count_before_deduplication"], 2)
        self.assertEqual(positive["prediction_count"], 1)
        self.assertEqual(negative["false_positives"], 1)

    def test_candidate_metrics_measure_union_selectivity_and_coverage(self) -> None:
        rows = [
            {
                "lecture_id": "lecture",
                "query_id": "positive",
                "candidate_windows": [
                    {"start_seconds": 0, "end_seconds": 5},
                    {"start_seconds": 5, "end_seconds": 20},
                ],
            },
            {
                "lecture_id": "lecture",
                "query_id": "negative",
                "candidate_windows": [],
            },
        ]
        metrics = evaluate_candidate_windows(rows, self.truth)
        self.assertEqual(metrics["total_candidate_duration_seconds"], 20)
        self.assertEqual(metrics["candidate_selectivity"], 0.1)
        self.assertEqual(metrics["candidate_recall_at_full_coverage"], 1.0)

    def test_binary_evaluation_ors_candidate_decisions_and_counts_empty_pairs(self) -> None:
        decisions = [
            {
                "lecture_id": "lecture",
                "query_id": "positive",
                "window_id": 0,
                "event_present": False,
            },
            {
                "lecture_id": "lecture",
                "query_id": "positive",
                "window_id": 1,
                "event_present": True,
            },
        ]
        evaluation = evaluate_binary_decisions(decisions, self.truth)
        self.assertEqual(
            evaluation["metrics"],
            {
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
            },
        )
        negative = next(
            item
            for item in evaluation["per_pair"]
            if item["query_id"] == "negative"
        )
        self.assertEqual(negative["decision_count"], 0)
        self.assertFalse(negative["predicted_present"])

    def test_binary_evaluation_rejects_non_boolean_and_duplicate_decisions(self) -> None:
        invalid = {
            "lecture_id": "lecture",
            "query_id": "positive",
            "event_present": "yes",
        }
        with self.assertRaises(ExperimentDataError):
            evaluate_binary_decisions([invalid], self.truth)
        duplicate = {
            "lecture_id": "lecture",
            "query_id": "positive",
            "window_id": 0,
            "event_present": True,
        }
        with self.assertRaises(ExperimentDataError):
            evaluate_binary_decisions([duplicate, duplicate], self.truth)


class LectureDownloadAndTranscriptionTests(unittest.TestCase):
    def test_resumable_download_validates_then_completes_and_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = LectureSource(
                lecture_id="lecture",
                title="Test",
                filename="lecture.mp4",
                page_url="https://example.test/page",
                download_url="https://example.test/video.mp4",
            )
            staging = root / "videos/lecture.mp4.part"
            staging.parent.mkdir(parents=True)
            staging.write_bytes(b"abc")
            response = _FakeResponse(
                b"def",
                status=206,
                headers={"Content-Range": "bytes 3-5/6"},
            )
            manifest = download_lectures(
                root=root,
                sources=(source,),
                open_request=lambda request: response,
                media_probe=lambda path: _media(30),
            )
            destination = root / "videos/lecture.mp4"
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertEqual(manifest["lectures"][0]["status"], "complete")
            destination.write_bytes(b"changed")
            with self.assertRaises(ExperimentDataError):
                download_lectures(
                    root=root,
                    sources=(source,),
                    open_request=lambda request: self.fail("must not redownload"),
                    media_probe=lambda path: _media(30),
                )

    def test_download_recovers_after_crash_between_validation_and_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = LectureSource(
                "lecture",
                "Test",
                "lecture.mp4",
                "https://example.test/page",
                "https://example.test/download",
            )
            response = _FakeResponse(b"video")
            with self.assertRaises(OSError):
                download_lectures(
                    root=root,
                    sources=(source,),
                    open_request=lambda request: response,
                    media_probe=lambda path: _media(30),
                    promote_file=lambda source, destination: (_ for _ in ()).throw(
                        OSError("injected promotion failure")
                    ),
                )
            manifest = load_json_object(root / "manifest.json")
            self.assertEqual(manifest["lectures"][0]["status"], "validated_staging")
            completed = download_lectures(
                root=root,
                sources=(source,),
                open_request=lambda request: self.fail("must use validated staging"),
                media_probe=lambda path: self.fail("must not reprobe"),
            )
            self.assertEqual(completed["lectures"][0]["status"], "complete")

    def test_transcription_is_content_bound_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "videos/lecture.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            atomic_write_json(
                root / "manifest.json",
                {
                    "lectures": [
                        {
                            "lecture_id": "lecture",
                            "path": "videos/lecture.mp4",
                            "sha256": sha256_file(video),
                            "status": "complete",
                            "media": _media(10),
                        }
                    ]
                },
            )
            model = _FakeWhisper()
            first = transcribe_lectures(root=root, model=model)
            self.assertTrue(
                (root / "transcripts/lecture.whisper.raw.json").is_file()
            )
            normalized_path = root / "transcripts/lecture.whisper.json"
            stale = load_json_object(normalized_path)
            stale.pop("normalization_contract_version")
            atomic_write_json(normalized_path, stale)
            second = transcribe_lectures(root=root, model=model)
            self.assertEqual(model.calls, 1)
            self.assertEqual(first["lectures"], second["lectures"])
            self.assertEqual(
                load_json_object(normalized_path)["normalization_contract_version"],
                NORMALIZATION_CONTRACT_VERSION,
            )
            video.write_bytes(b"changed")
            with self.assertRaises(ExperimentDataError):
                transcribe_lectures(root=root, model=model)

    def test_raw_checkpoint_recovers_without_a_second_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "videos/lecture.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            atomic_write_json(
                root / "manifest.json",
                {
                    "lectures": [
                        {
                            "lecture_id": "lecture",
                            "path": "videos/lecture.mp4",
                            "sha256": sha256_file(video),
                            "status": "complete",
                            "media": _media(10),
                        }
                    ]
                },
            )
            model = _FakeWhisper()
            with patch(
                "scripts.lectures.transcribe.normalize_transcription",
                side_effect=ExperimentDataError("injected normalization failure"),
            ):
                with self.assertRaises(ExperimentDataError):
                    transcribe_lectures(root=root, model=model)
            self.assertEqual(model.calls, 1)
            self.assertTrue(
                (root / "transcripts/lecture.whisper.raw.json").is_file()
            )
            self.assertFalse(
                (root / "transcripts/lecture.whisper.json").exists()
            )

            transcribe_lectures(root=root, model=model)
            self.assertEqual(model.calls, 1)
            self.assertTrue(
                (root / "transcripts/lecture.whisper.json").is_file()
            )

    def test_whisper_normalization_rejects_empty_and_nonfinite_results(self) -> None:
        with self.assertRaises(ExperimentDataError):
            normalize_transcription(
                {"segments": [], "language": "en"},
                source_id="x",
                source_path="x.mp4",
                source_sha256="a" * 64,
                model_name="small",
            )
        with self.assertRaises(ExperimentDataError):
            normalize_transcription(
                {
                    "segments": [
                        {
                            "start": 0,
                            "end": 1,
                            "text": "x",
                            "avg_logprob": math.nan,
                        }
                    ],
                    "language": "en",
                },
                source_id="x",
                source_path="x.mp4",
                source_sha256="a" * 64,
                model_name="small",
            )

    def test_whisper_normalization_accepts_ordered_overlap_but_rejects_backtracking(self) -> None:
        normalized = normalize_transcription(
            {
                "language": "en",
                "text": "first second",
                "segments": [
                    {"start": 10, "end": 20, "text": "first"},
                    {"start": 19.5, "end": 30, "text": "second"},
                ],
            },
            source_id="x",
            source_path="x.mp4",
            source_sha256="a" * 64,
            model_name="small",
        )
        self.assertEqual(normalized["segments"][1]["start_seconds"], 19.5)

        for segments in (
            [
                {"start": 10, "end": 20, "text": "first"},
                {"start": 9, "end": 30, "text": "second"},
            ],
            [
                {"start": 10, "end": 20, "text": "first"},
                {"start": 15, "end": 19, "text": "second"},
            ],
        ):
            with self.subTest(segments=segments), self.assertRaises(
                ExperimentDataError
            ):
                normalize_transcription(
                    {"language": "en", "segments": segments},
                    source_id="x",
                    source_path="x.mp4",
                    source_sha256="a" * 64,
                    model_name="small",
                )

    def test_final_timestamp_quantization_is_clamped_and_audited(self) -> None:
        normalized = normalize_transcription(
            {
                "language": "en",
                "text": "Thank you.",
                "segments": [
                    {"start": 98.0, "end": 100.04, "text": "Thank you."}
                ],
            },
            source_id="x",
            source_path="x.mp4",
            source_sha256="a" * 64,
            model_name="small",
            source_duration_seconds=100.0,
        )
        segment = normalized["segments"][0]
        self.assertEqual(segment["end_seconds"], 100.0)
        self.assertEqual(segment["raw_end_seconds"], 100.04)
        self.assertEqual(
            segment["boundary_adjustment"],
            "final_segment_timestamp_quantization",
        )
        self.assertEqual(normalized["normalization_contract_version"], 2)
        self.assertAlmostEqual(
            normalized["boundary_adjustments"][0]["overrun_seconds"], 0.04
        )

    def test_internal_or_large_duration_overrun_still_fails_closed(self) -> None:
        cases = (
            [
                {"start": 0, "end": 10.01, "text": "internal"},
                {"start": 10.01, "end": 10.02, "text": "last"},
            ],
            [
                {
                    "start": 9,
                    "end": 10 + FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS + 0.01,
                    "text": "too far",
                }
            ],
        )
        for segments in cases:
            with self.subTest(segments=segments), self.assertRaises(
                ExperimentDataError
            ):
                normalize_transcription(
                    {"language": "en", "segments": segments},
                    source_id="x",
                    source_path="x.mp4",
                    source_sha256="a" * 64,
                    model_name="small",
                    source_duration_seconds=10.0,
                )


class LectureV2PlanTests(unittest.TestCase):
    def test_naive_and_optimized_share_binary_and_localization_contracts(self) -> None:
        naive_binary = render_query(build_naive_v2_binary_plan("naive.jsonl"))
        optimized_binary = render_query(
            build_materialized_v2_binary_plan("optimized.jsonl")
        )
        naive_localization = render_query(
            build_naive_v2_localization_plan("naive-decisions.jsonl")
        )
        optimized_localization = render_query(
            build_materialized_v2_localization_plan(
                "optimized-decisions.jsonl"
            )
        )
        for rendered in (naive_binary, optimized_binary):
            self.assertIn(BINARY_VERIFIER_PROMPT.splitlines()[0], rendered)
            self.assertIn('"event_present": {"type": "boolean"}', rendered)
            self.assertNotIn("start_minute", rendered)
        for rendered in (naive_localization, optimized_localization):
            self.assertIn(
                EPISODE_LOCALIZATION_PROMPT.splitlines()[0], rendered
            )
            self.assertIn('"minItems": 1', rendered)
            self.assertLess(
                rendered.index("step_1 = Filter("),
                rendered.index("step_2 = Map("),
            )
        self.assertEqual(
            BINARY_VERIFICATION_SCHEMA["verification"]["required"],
            ["event_present", "confidence", "evidence"],
        )
        self.assertEqual(EPISODE_SCHEMA["episodes"]["minItems"], 1)


class LectureV3PlanTests(unittest.TestCase):
    def test_transcript_owns_v3_timestamp_coordinates(self) -> None:
        proposal = render_query(build_v3_transcript_proposal_plan("input.jsonl"))
        refinement = render_query(build_v3_refinement_plan("refinement.jsonl"))
        self.assertIn(TRANSCRIPT_EPISODE_PROPOSAL_PROMPT.splitlines()[0], proposal)
        self.assertIn('"start_segment_id"', proposal)
        self.assertIn(
            TRANSCRIPT_GROUNDED_REFINEMENT_PROMPT.splitlines()[0], refinement
        )
        self.assertIn('Record["timestamped_transcript_context"]', refinement)
        self.assertIn('"maxItems": 1', refinement)
        self.assertNotIn("start_minute", refinement)
        self.assertNotIn("end_second", refinement)
        self.assertEqual(
            TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA["episode_refinements"][
                "maxItems"
            ],
            1,
        )


class LectureV4PlanTests(unittest.TestCase):
    def test_v4_compiles_conditions_and_derives_timestamps_from_evidence(self) -> None:
        conditions = render_query(
            build_v4_query_condition_plan("queries.jsonl")
        )
        grounding = render_query(
            build_v4_predicate_grounding_plan("grounding.jsonl")
        )
        self.assertIn(QUERY_CONDITION_PROMPT.splitlines()[0], conditions)
        self.assertIn('"minItems": 1', conditions)
        self.assertIn('"maxItems": 8', conditions)
        self.assertIn(PREDICATE_GROUNDING_PROMPT.splitlines()[0], grounding)
        self.assertIn('Record["required_conditions"]', grounding)
        self.assertIn('Record["timestamped_transcript_context"]', grounding)
        self.assertNotIn("start_minute", grounding)
        self.assertNotIn("end_second", grounding)
        self.assertLess(
            grounding.index("step_2 = Map("),
            grounding.index("step_3 = Filter("),
        )
        self.assertEqual(
            QUERY_CONDITION_SCHEMA["condition_descriptions"]["maxItems"], 8
        )
        self.assertEqual(
            PREDICATE_GROUNDING_SCHEMA["condition_evidence"]["maxItems"], 8
        )


class LectureV5PlanTests(unittest.TestCase):
    def test_v5_separates_acceptance_gates_from_timestamp_anchors(self) -> None:
        conditions = render_query(
            build_v5_query_condition_plan("queries.jsonl")
        )
        grounding = render_query(
            build_v5_predicate_grounding_plan("grounding.jsonl")
        )
        self.assertIn(ROLE_AWARE_QUERY_CONDITION_PROMPT.splitlines()[0], conditions)
        self.assertIn('"enum": ["gate", "anchor"]', conditions)
        self.assertIn(
            ROLE_AWARE_PREDICATE_GROUNDING_PROMPT.splitlines()[0], grounding
        )
        self.assertIn('Record["required_conditions"]', grounding)
        self.assertIn('Record["timestamped_transcript_context"]', grounding)
        self.assertNotIn("start_minute", grounding)
        self.assertNotIn("end_second", grounding)
        self.assertLess(
            grounding.index("step_2 = Map("),
            grounding.index("step_3 = Filter("),
        )
        self.assertEqual(
            ROLE_AWARE_QUERY_CONDITION_SCHEMA["condition_specs"]["items"][
                "properties"
            ]["role"]["enum"],
            ["gate", "anchor"],
        )


class LectureWorkflowTests(unittest.TestCase):
    def test_offline_nine_pair_workflow_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_frozen_fixture(root)
            initial_status = status(root)
            self.assertEqual(initial_status["ground_truth_review"]["total_pairs"], 9)
            self.assertTrue(initial_status["download_validation"]["valid"])
            self.assertTrue(initial_status["transcript_validation"]["valid"])
            freeze_ground_truth(root)
            config = prepare_experiment(root)
            self.assertEqual(config["pair_count"], 9)
            rows = load_jsonl_objects(experiment_directory(root) / "input.jsonl")
            self.assertEqual(len(rows), 9)
            _assert_no_forbidden_keys(self, rows)
            plans = experiment_directory(root) / "plans"
            self.assertEqual(len(list(plans.glob("*.py"))), 5)
            naive_plan_text = (plans / "naive.py").read_text(encoding="utf-8")
            optimized_plan_text = (plans / "optimized_verification.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(VERIFIER_PROMPT.splitlines()[0], naive_plan_text)
            self.assertIn(VERIFIER_PROMPT.splitlines()[0], optimized_plan_text)
            self.assertIn(
                VERIFIER_PROMPT.splitlines()[0],
                (plans / "optimized_materialized_verification.py").read_text(
                    encoding="utf-8"
                ),
            )

            input_path = experiment_directory(root) / "input.jsonl"
            run_naive(
                root,
                prompt_executor=_static_executor(
                    build_naive_plan(str(input_path.resolve())),
                    "full_lecture_audiovisual_verification",
                    _video_handler,
                ),
            )
            run_transcript_only(
                root,
                prompt_executor=_static_executor(
                    build_transcript_only_plan(str(input_path.resolve())),
                    "transcript_only_event_selection",
                    _transcript_handler,
                ),
            )
            run_candidates(
                root,
                prompt_executor=_static_executor(
                    build_candidate_plan(str(input_path.resolve())),
                    "transcript_candidate_range_selection",
                    _candidate_handler,
                ),
            )
            candidate_path = (
                experiment_directory(root)
                / "runs/transcript_candidates/candidates.jsonl"
            )
            materialize_clips(
                root,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_clip_probe,
            )
            materialized_directory = (
                experiment_directory(root) / "runs/materialized_clips"
            )
            (materialized_directory / "completion.json").unlink()
            recovered = materialize_clips(
                root,
                clip_processor=lambda *args: self.fail(
                    "finalized materialization must recover without re-encoding"
                ),
                media_probe=_fake_clip_probe,
            )
            self.assertTrue(recovered["completion_recovered"])
            materialized_input = (
                materialized_directory / "input.jsonl"
            )
            run_optimized_materialized(
                root,
                prompt_executor=_static_executor(
                    build_materialized_verification_plan(
                        str(materialized_input.resolve())
                    ),
                    "materialized_candidate_audiovisual_verification",
                    _video_handler,
                ),
                media_probe=_fake_clip_probe,
            )
            comparison = compare_results(root, media_probe=_fake_clip_probe)
            self.assertEqual(comparison["naive"]["primary_accuracy"]["f1"], 1.0)
            self.assertEqual(
                comparison["transcript_only"]["primary_accuracy"]["f1"], 1.0
            )
            self.assertEqual(comparison["optimized"]["primary_accuracy"]["f1"], 1.0)
            self.assertEqual(
                comparison["optimized"]["clip_materialization"]["clip_count"],
                3,
            )
            self.assertTrue(comparison["optimized"]["query_time_latency"]["valid"])
            self.assertEqual(
                comparison["optimized"]["query_time_latency"][
                    "reused_materialized_clip_count"
                ],
                0,
            )
            self.assertEqual(
                comparison["optimized"]["candidate_metrics"][
                    "candidate_recall_at_full_coverage"
                ],
                1.0,
            )
            workflow_status = status(root, media_probe=_fake_clip_probe)
            self.assertEqual(
                workflow_status["stages"]["materialized_clips"], "complete"
            )
            self.assertEqual(
                workflow_status["stages"]["optimized_materialized"], "complete"
            )

            v2_config = prepare_v2(root, media_probe=_fake_clip_probe)
            self.assertEqual(v2_config["method_contract_version"], 2)
            with self.assertRaises(ExperimentDataError):
                prepare_v2(root, media_probe=_fake_clip_probe)
            self.assertTrue(
                (experiment_directory(root) / V2_CONFIG_FILENAME).is_file()
            )
            v2_plans = experiment_directory(root) / "plans/v2"
            self.assertEqual(len(list(v2_plans.glob("*.py"))), 4)

            naive_binary_plan = build_naive_v2_binary_plan(
                str(input_path.resolve())
            )
            run_naive_v2_binary(
                root,
                prompt_executor=_static_executor(
                    naive_binary_plan,
                    "v2_full_lecture_binary_verification",
                    _binary_handler,
                ),
            )
            naive_decision_input = (
                experiment_directory(root)
                / "runs/v2/naive_binary/localization_input.jsonl"
            )
            self.assertEqual(len(load_jsonl_objects(naive_decision_input)), 9)
            naive_localization_calls: list[tuple[str, str]] = []

            def count_naive_localizations(
                resolved: object,
                payload: dict[str, object],
                context: object,
            ) -> dict[str, object]:
                naive_localization_calls.append(
                    (str(payload["lecture_id"]), str(payload["query_id"]))
                )
                return _episode_handler(resolved, payload, context)

            naive_localization_plan = build_naive_v2_localization_plan(
                str(naive_decision_input.resolve())
            )
            run_naive_v2_localization(
                root,
                prompt_executor=_static_executor(
                    naive_localization_plan,
                    "v2_full_lecture_complete_episode_localization",
                    count_naive_localizations,
                ),
            )
            self.assertEqual(len(naive_localization_calls), 3)
            self.assertEqual(set(naive_localization_calls), POSITIVE_PAIRS)

            optimized_binary_plan = build_materialized_v2_binary_plan(
                str(materialized_input.resolve())
            )
            run_optimized_v2_binary(
                root,
                prompt_executor=_static_executor(
                    optimized_binary_plan,
                    "v2_materialized_candidate_binary_verification",
                    _binary_handler,
                ),
                media_probe=_fake_clip_probe,
            )
            optimized_decision_input = (
                experiment_directory(root)
                / "runs/v2/optimized_binary/localization_input.jsonl"
            )
            optimized_localization_calls: list[tuple[str, str]] = []

            def count_optimized_localizations(
                resolved: object,
                payload: dict[str, object],
                context: object,
            ) -> dict[str, object]:
                optimized_localization_calls.append(
                    (str(payload["lecture_id"]), str(payload["query_id"]))
                )
                return _episode_handler(resolved, payload, context)

            optimized_localization_plan = build_materialized_v2_localization_plan(
                str(optimized_decision_input.resolve())
            )
            run_optimized_v2_localization(
                root,
                prompt_executor=_static_executor(
                    optimized_localization_plan,
                    "v2_materialized_candidate_complete_episode_localization",
                    count_optimized_localizations,
                ),
                media_probe=_fake_clip_probe,
            )
            self.assertEqual(len(optimized_localization_calls), 3)
            self.assertEqual(set(optimized_localization_calls), POSITIVE_PAIRS)

            v2_comparison = compare_v2_results(
                root, media_probe=_fake_clip_probe
            )
            self.assertEqual(v2_comparison["naive"]["binary_accuracy"]["f1"], 1.0)
            self.assertEqual(
                v2_comparison["optimized"]["binary_accuracy"]["f1"], 1.0
            )
            self.assertEqual(
                v2_comparison["naive"]["temporal_accuracy"]["f1"], 1.0
            )
            self.assertEqual(
                v2_comparison["optimized"]["temporal_accuracy"]["f1"], 1.0
            )
            self.assertEqual(
                set(v2_comparison["optimized"]["api_usage_by_stage"]),
                {
                    "transcript_candidates",
                    "binary_verification",
                    "episode_localization",
                },
            )
            self.assertTrue(
                v2_comparison["optimized"]["query_time_latency"]["valid"]
            )
            v2_status = status(root, media_probe=_fake_clip_probe)["v2"]
            self.assertEqual(v2_status["config"], "prepared")
            self.assertTrue(v2_status["comparison"])
            self.assertTrue(
                all(value == "complete" for value in v2_status["stages"].values())
            )

            v3_config = prepare_v3(root)
            self.assertEqual(v3_config["method_contract_version"], 3)
            self.assertIn("post_hoc", v3_config["evaluation_role"])
            with self.assertRaises(ExperimentDataError):
                prepare_v3(root)
            self.assertTrue(
                (experiment_directory(root) / V3_CONFIG_FILENAME).is_file()
            )
            v3_plans = experiment_directory(root) / "plans/v3"
            self.assertEqual(len(list(v3_plans.glob("*.py"))), 2)

            proposal_plan = build_v3_transcript_proposal_plan(
                str(input_path.resolve())
            )
            run_v3_transcript_proposals(
                root,
                prompt_executor=_static_executor(
                    proposal_plan,
                    "v3_transcript_complete_episode_proposals",
                    _v3_proposal_handler,
                ),
            )
            materialize_v3_proposal_clips(
                root,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_clip_probe,
            )
            v3_materialized = (
                experiment_directory(root)
                / "runs/v3/materialized_proposal_clips"
            )
            refinement_input = v3_materialized / "refinement_input.jsonl"
            (v3_materialized / "completion.json").unlink()
            refinement_input.unlink()
            recovered_v3 = materialize_v3_proposal_clips(
                root,
                clip_processor=lambda *args: self.fail(
                    "v3 recovery must not re-encode completed clips"
                ),
                media_probe=_fake_clip_probe,
            )
            self.assertTrue(recovered_v3["completion_recovered"])
            refinement_rows = load_jsonl_objects(refinement_input)
            self.assertEqual(len(refinement_rows), 3)
            _assert_no_forbidden_keys(self, refinement_rows)
            self.assertTrue(
                all(
                    row["proposed_episode"]["start_segment_id"]
                    in row["transcript_context_segment_ids"]
                    for row in refinement_rows
                )
            )

            refinement_plan = build_v3_refinement_plan(
                str(refinement_input.resolve())
            )
            run_v3_refinement(
                root,
                prompt_executor=_static_executor(
                    refinement_plan,
                    "v3_audiovisual_transcript_boundary_refinement",
                    _v3_refinement_handler,
                ),
                media_probe=_fake_clip_probe,
            )
            v3_comparison = compare_v3_results(
                root, media_probe=_fake_clip_probe
            )
            self.assertEqual(v3_comparison["naive"]["primary_accuracy"]["f1"], 1.0)
            self.assertEqual(
                v3_comparison["optimized"]["primary_accuracy"]["f1"], 1.0
            )
            self.assertEqual(
                set(v3_comparison["optimized"]["api_usage_by_stage"]),
                {
                    "transcript_episode_proposals",
                    "audiovisual_transcript_boundary_refinement",
                },
            )
            v3_status = status(root, media_probe=_fake_clip_probe)["v3"]
            self.assertEqual(v3_status["config"], "prepared")
            self.assertTrue(v3_status["comparison"])
            self.assertTrue(
                all(value == "complete" for value in v3_status["stages"].values())
            )

            v4_config = prepare_v4(root, media_probe=_fake_clip_probe)
            self.assertEqual(v4_config["method_contract_version"], 4)
            self.assertIn("post_hoc", v4_config["evaluation_role"])
            with self.assertRaises(ExperimentDataError):
                prepare_v4(root, media_probe=_fake_clip_probe)
            self.assertTrue(
                (experiment_directory(root) / V4_CONFIG_FILENAME).is_file()
            )
            v4_plans = experiment_directory(root) / "plans/v4"
            self.assertEqual(len(list(v4_plans.glob("*.py"))), 2)

            v4_query_input = experiment_directory(root) / "v4_query_input.jsonl"
            condition_plan = build_v4_query_condition_plan(
                str(v4_query_input.resolve())
            )
            condition_result = run_v4_query_conditions(
                root,
                prompt_executor=_static_executor(
                    condition_plan,
                    "v4_compile_mandatory_query_conditions",
                    _v4_condition_handler,
                ),
            )
            self.assertEqual(len(condition_result["conditions"]), 3)
            self.assertEqual(condition_result["grounding_row_count"], 3)
            v4_condition_directory = (
                experiment_directory(root) / "runs/v4/query_conditions"
            )
            grounding_input = v4_condition_directory / "grounding_input.jsonl"
            grounding_rows = load_jsonl_objects(grounding_input)
            _assert_no_forbidden_keys(self, grounding_rows)
            self.assertTrue(
                all(len(row["required_conditions"]) == 1 for row in grounding_rows)
            )

            grounding_plan = build_v4_predicate_grounding_plan(
                str(grounding_input.resolve())
            )
            run_v4_predicate_grounding(
                root,
                prompt_executor=_static_executor(
                    grounding_plan,
                    "v4_ground_every_required_condition",
                    _v4_grounding_handler,
                ),
                media_probe=_fake_clip_probe,
            )
            v4_comparison = compare_v4_results(
                root, media_probe=_fake_clip_probe
            )
            self.assertEqual(v4_comparison["naive"]["primary_accuracy"]["f1"], 1.0)
            self.assertEqual(
                v4_comparison["optimized"]["primary_accuracy"]["f1"], 1.0
            )
            self.assertEqual(
                set(v4_comparison["optimized"]["api_usage_by_stage"]),
                {
                    "transcript_episode_proposals",
                    "query_condition_compilation",
                    "mandatory_predicate_grounding",
                },
            )
            v4_status = status(root, media_probe=_fake_clip_probe)["v4"]
            self.assertEqual(v4_status["config"], "prepared")
            self.assertTrue(v4_status["comparison"])
            self.assertTrue(
                all(value == "complete" for value in v4_status["stages"].values())
            )

            v5_config = prepare_v5(root, media_probe=_fake_clip_probe)
            self.assertEqual(v5_config["method_contract_version"], 5)
            self.assertIn("post_hoc", v5_config["evaluation_role"])
            with self.assertRaises(ExperimentDataError):
                prepare_v5(root, media_probe=_fake_clip_probe)
            self.assertTrue(
                (experiment_directory(root) / V5_CONFIG_FILENAME).is_file()
            )
            v5_plans = experiment_directory(root) / "plans/v5"
            self.assertEqual(len(list(v5_plans.glob("*.py"))), 2)

            v5_query_input = experiment_directory(root) / "v5_query_input.jsonl"
            role_condition_plan = build_v5_query_condition_plan(
                str(v5_query_input.resolve())
            )
            role_condition_result = run_v5_query_conditions(
                root,
                prompt_executor=_static_executor(
                    role_condition_plan,
                    "v5_compile_nonredundant_gate_anchor_conditions",
                    _v5_condition_handler,
                ),
            )
            self.assertEqual(len(role_condition_result["conditions"]), 3)
            self.assertEqual(role_condition_result["grounding_row_count"], 3)
            v5_condition_directory = (
                experiment_directory(root) / "runs/v5/query_conditions"
            )
            v5_grounding_input = (
                v5_condition_directory / "grounding_input.jsonl"
            )
            v5_grounding_rows = load_jsonl_objects(v5_grounding_input)
            _assert_no_forbidden_keys(self, v5_grounding_rows)
            self.assertTrue(
                all(
                    [condition["role"] for condition in row["required_conditions"]]
                    == ["gate", "anchor"]
                    for row in v5_grounding_rows
                )
            )

            role_grounding_plan = build_v5_predicate_grounding_plan(
                str(v5_grounding_input.resolve())
            )
            run_v5_predicate_grounding(
                root,
                prompt_executor=_static_executor(
                    role_grounding_plan,
                    "v5_ground_every_gate_and_anchor_condition",
                    _v5_grounding_handler,
                ),
                media_probe=_fake_clip_probe,
            )
            v5_grounding_directory = (
                experiment_directory(root)
                / "runs/v5/role_aware_predicate_grounding"
            )
            v5_predictions = load_json_object(
                v5_grounding_directory / "predictions.json"
            )["predictions"]
            self.assertTrue(
                all(
                    (prediction["start_seconds"], prediction["end_seconds"])
                    == (10.0, 20.0)
                    for prediction in v5_predictions
                )
            )
            v5_comparison = compare_v5_results(
                root, media_probe=_fake_clip_probe
            )
            self.assertEqual(v5_comparison["naive"]["primary_accuracy"]["f1"], 1.0)
            self.assertEqual(
                v5_comparison["optimized"]["primary_accuracy"]["f1"], 1.0
            )
            self.assertEqual(
                set(v5_comparison["optimized"]["api_usage_by_stage"]),
                {
                    "transcript_episode_proposals",
                    "gate_anchor_query_compilation",
                    "role_aware_predicate_grounding",
                },
            )
            v5_status = status(root, media_probe=_fake_clip_probe)["v5"]
            self.assertEqual(v5_status["config"], "prepared")
            self.assertTrue(v5_status["comparison"])
            self.assertTrue(
                all(value == "complete" for value in v5_status["stages"].values())
            )

            with candidate_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            self.assertEqual(
                status(root, media_probe=_fake_clip_probe)["stages"][
                    "transcript_candidates"
                ],
                "invalid_completed_artifacts",
            )
            with self.assertRaises(ExperimentDataError):
                compare_results(root, media_probe=_fake_clip_probe)

    def test_review_gate_requires_all_pairs_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_download_fixture(root)
            initialize_ground_truth_review(root)
            with self.assertRaises(ExperimentDataError):
                initialize_ground_truth_review(root)
            with self.assertRaises(ExperimentDataError):
                freeze_ground_truth(root)

    def test_prepared_hash_change_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _build_frozen_fixture(root)
            freeze_ground_truth(root)
            prepare_experiment(root)
            truth_path = experiment_directory(root) / "ground_truth.json"
            truth = load_json_object(truth_path)
            truth["pairs"][0]["review_notes"] = "changed after prepare"
            atomic_write_json(truth_path, truth)
            with self.assertRaises(ExperimentDataError):
                run_naive(root, prompt_executor=object())

    def test_external_commands_are_preview_only_without_execute(self) -> None:
        from scripts.lectures.experiment import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--root", str(root), "download"]), 0)
                self.assertEqual(main(["--root", str(root), "transcribe"]), 0)
                self.assertEqual(
                    main(["--root", str(root), "materialize-clips"]), 0
                )
                self.assertEqual(
                    main(["--root", str(root), "run-optimized-materialized"]),
                    0,
                )
                for command in (
                    "run-naive-v2-binary",
                    "run-optimized-v2-binary",
                    "run-naive-v2-localization",
                    "run-optimized-v2-localization",
                    "run-v3-transcript-proposals",
                    "materialize-v3-proposal-clips",
                    "run-v3-refinement",
                    "run-v4-query-conditions",
                    "run-v4-predicate-grounding",
                    "run-v5-query-conditions",
                    "run-v5-predicate-grounding",
                ):
                    self.assertEqual(
                        main(["--root", str(root), command]), 0
                    )
            self.assertFalse((root / "manifest.json").exists())


class SharedUsageTests(unittest.TestCase):
    def test_video_content_hash_is_part_of_provider_cache_identity(self) -> None:
        prompt = PromptSpec(parts=("video",), output_schema={"answer": "string"})
        first = ResolvedPrompt(
            parts=({"type": "Video", "path": "/same/path.mp4", "sha256": "a" * 64},),
            output_schema=prompt.output_schema,
        )
        second = ResolvedPrompt(
            parts=({"type": "Video", "path": "/same/path.mp4", "sha256": "b" * 64},),
            output_schema=prompt.output_schema,
        )
        self.assertNotEqual(
            cache_key(model="model", op_type="map", prompt=prompt, resolved_prompt=first),
            cache_key(model="model", op_type="map", prompt=prompt, resolved_prompt=second),
        )

    def test_corrupt_provider_cache_fails_closed(self) -> None:
        class Part:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class Content:
            def __init__(self, *, parts: object) -> None:
                self.parts = parts

        class Types:
            pass

        Types.Part = Part
        Types.Content = Content

        class Models:
            def generate_content(self, **kwargs: object) -> object:
                return type(
                    "Response",
                    (),
                    {"text": '{"answer": "ok"}', "usage_metadata": None},
                )()

        class Client:
            def __init__(self) -> None:
                self.models = Models()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executor = CachingGeminiPromptExecutor(
                cache_directory=directory / "cache",
                recorder=ApiCallRecorder(directory / "api_calls.jsonl"),
                model="test-model",
                client=Client(),
                types_module=Types,
            )
            prompt = PromptSpec(parts=("prompt",), output_schema={"answer": "string"})
            resolved = ResolvedPrompt(
                parts=("prompt",), output_schema=prompt.output_schema
            )
            self.assertEqual(
                executor.execute("map", prompt, resolved, {}, {}), {"answer": "ok"}
            )
            cache_path = next((directory / "cache").glob("*.json"))
            cache_path.write_text("not-json", encoding="utf-8")
            with self.assertRaises(ExperimentDataError):
                executor.execute("map", prompt, resolved, {}, {})

    def test_enum_style_audio_modality_uses_audio_price(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "api_calls.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "elapsed_seconds": 1,
                        "usage_metadata": {
                            "prompt_token_count": 100,
                            "candidates_token_count": 10,
                            "prompt_tokens_details": [
                                {"modality": "Modality.AUDIO", "token_count": 40},
                                {"modality": "TEXT", "token_count": 60},
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = aggregate_api_usage(
                path,
                input_usd_per_million_tokens=1,
                audio_input_usd_per_million_tokens=2,
                output_usd_per_million_tokens=3,
                pricing_source="test",
            )
            self.assertEqual(result["input_tokens_by_modality"], {"audio": 40, "text": 60})
            self.assertAlmostEqual(result["estimated_input_cost_usd"], 140 / 1_000_000)

    def test_incomplete_modality_breakdown_falls_back_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "api_calls.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "elapsed_seconds": 1,
                        "usage_metadata": {
                            "prompt_token_count": 100,
                            "prompt_tokens_details": [
                                {"modality": "AUDIO", "token_count": 40}
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = aggregate_api_usage(
                path,
                input_usd_per_million_tokens=1,
                audio_input_usd_per_million_tokens=2,
                output_usd_per_million_tokens=3,
                pricing_source="test",
            )
            self.assertFalse(result["modality_breakdown_complete"])
            self.assertIsNotNone(result["cost_assumptions"]["warning"])
            self.assertAlmostEqual(result["estimated_input_cost_usd"], 100 / 1_000_000)


def _segments() -> list[dict[str, object]]:
    return [
        {"segment_id": 0, "start_seconds": 10.0, "end_seconds": 20.0, "text": "one"},
        {"segment_id": 1, "start_seconds": 25.0, "end_seconds": 30.0, "text": "two"},
        {"segment_id": 2, "start_seconds": 90.0, "end_seconds": 95.0, "text": "three"},
    ]


def _range(start: object, end: object, *, confidence: float = 1.0) -> dict[str, object]:
    return {
        "start_segment_id": start,
        "end_segment_id": end,
        "confidence": confidence,
        "evidence": "test",
    }


def _prediction(
    query_id: str, start: float, end: float, *, confidence: float = 1.0
) -> dict[str, object]:
    return {
        "lecture_id": "lecture",
        "query_id": query_id,
        "start_seconds": start,
        "end_seconds": end,
        "interval_valid": True,
        "confidence": confidence,
    }


def _media(duration: float) -> dict[str, object]:
    return {
        "duration_seconds": duration,
        "width": 640,
        "height": 360,
        "video_codec": "h264",
        "audio_stream_count": 1,
        "audio_codecs": ["aac"],
    }


def _candidate_row(
    source: Path,
    *,
    start: float,
    end: float,
) -> dict[str, object]:
    return {
        "lecture_id": "lecture",
        "query_id": "query",
        "query_text": "query",
        "duration_seconds": 100.0,
        "candidate_padding_seconds": 30.0,
        "candidate_ranges": [],
        "invalid_candidate_ranges": [],
        "candidate_windows": [
            {
                "window_id": 0,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "unpadded_start_seconds": start,
                "unpadded_end_seconds": end,
                "segment_ranges": [],
            }
        ],
        "video": {
            "type": "Video",
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "mime_type": "video/mp4",
            "fps": 1.0,
        },
    }


def _fake_clip_processor(
    source: Path,
    destination: Path,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, object]:
    duration = end_seconds - start_seconds
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(duration), encoding="utf-8")
    return _clip_media(duration)


def _fake_clip_probe(path: Path) -> dict[str, object]:
    return _clip_media(float(path.read_text(encoding="utf-8")))


def _clip_media(duration: float) -> dict[str, object]:
    return {**_media(duration), "start_time_seconds": 0.0}


class _FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self.offset = 0

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        if self.offset >= len(self.payload):
            return b""
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class _FakeWhisper:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, video_path: Path, *, language: str) -> dict[str, object]:
        self.calls += 1
        return {
            "language": "en",
            "text": "hello",
            "segments": [{"start": 0, "end": 1, "text": "hello"}],
        }


def _build_download_fixture(root: Path) -> None:
    entries = []
    for index, lecture in enumerate(LECTURES):
        path = root / "videos" / lecture.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video-{index}".encode())
        entries.append(
            {
                "lecture_id": lecture.lecture_id,
                "title": lecture.title,
                "source_page_url": lecture.page_url,
                "download_url": lecture.download_url,
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "media": _media(100),
                "status": "complete",
            }
        )
    atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "source_catalog": source_catalog_payload(),
            "lectures": entries,
        },
    )


def _build_frozen_fixture(root: Path) -> None:
    _build_download_fixture(root)
    manifest = load_json_object(root / "manifest.json")
    elapsed_entries = []
    for entry in manifest["lectures"]:
        segments = _segments()
        raw_path = (
            root
            / "transcripts"
            / f"{entry['lecture_id']}.whisper.raw.json"
        )
        atomic_write_json(
            raw_path,
            {
                "schema_version": 1,
                "source_id": entry["lecture_id"],
                "source_path": entry["path"],
                "source_sha256": entry["sha256"],
                "model": WHISPER_MODEL,
                "requested_language": "en",
                "elapsed_seconds": 2.0,
                "result": {
                    "language": "en",
                    "text": "one two three",
                    "segments": [
                        {
                            "start": segment["start_seconds"],
                            "end": segment["end_seconds"],
                            "text": segment["text"],
                        }
                        for segment in segments
                    ],
                },
            },
        )
        payload = {
            "schema_version": 1,
            "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
            "source_id": entry["lecture_id"],
            "source_path": entry["path"],
            "source_sha256": entry["sha256"],
            "model": WHISPER_MODEL,
            "source_duration_seconds": 100.0,
            "final_segment_duration_tolerance_seconds": (
                FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS
            ),
            "boundary_adjustments": [],
            "language": "en",
            "text": "one two three",
            "segments": segments,
            "elapsed_seconds": 2.0,
            "raw_checkpoint_path": str(raw_path.relative_to(root)),
            "raw_checkpoint_sha256": sha256_file(raw_path),
        }
        transcript_path = root / "transcripts" / f"{entry['lecture_id']}.whisper.json"
        atomic_write_json(transcript_path, payload)
        elapsed_entries.append(
            {
                "lecture_id": entry["lecture_id"],
                "path": str(transcript_path.relative_to(root)),
                "segment_count": len(segments),
                "elapsed_seconds": 2.0,
                "status": "complete",
            }
        )
    atomic_write_json(
        root / "transcripts/index.json",
        {"schema_version": 1, "model": WHISPER_MODEL, "lectures": elapsed_entries},
    )
    review = initialize_ground_truth_review(root)
    for pair in review["pairs"]:
        key = (pair["lecture_id"], pair["query_id"])
        pair["review_status"] = "complete"
        pair["reviewer"] = "offline-test"
        pair["events"] = (
            [
                {
                    "start_seconds": 10.0,
                    "end_seconds": 20.0,
                    "annotation_notes": "deterministic fixture",
                }
            ]
            if key in POSITIVE_PAIRS
            else []
        )
    atomic_write_json(experiment_directory(root) / "ground_truth_review.json", review)


def _static_executor(plan: object, operator_name: str, handler: object) -> StaticPromptExecutor:
    node = next(
        item
        for item in plan.walk_postorder()
        if item.kind == "map" and item.name == operator_name
    )
    return StaticPromptExecutor({("map", node.spec.cache_key()): handler})


def _is_positive(payload: dict[str, object]) -> bool:
    return (payload["lecture_id"], payload["query_id"]) in POSITIVE_PAIRS


def _video_handler(resolved: object, payload: dict[str, object], context: object) -> dict[str, object]:
    return {
        "events": (
            [
                {
                    "start_minute": 0,
                    "start_second": 10,
                    "end_minute": 0,
                    "end_second": 20,
                    "confidence": 1,
                    "evidence": "deterministic audiovisual evidence",
                }
            ]
            if _is_positive(payload)
            else []
        )
    }


def _binary_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {
        "verification": {
            "event_present": _is_positive(payload),
            "confidence": 1,
            "evidence": "deterministic binary audiovisual evidence",
        }
    }


def _episode_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {
        "episodes": [
            {
                "start_minute": 0,
                "start_second": 10,
                "end_minute": 0,
                "end_second": 20,
                "confidence": 1,
                "evidence": "deterministic complete episode",
            }
        ]
    }


def _transcript_handler(resolved: object, payload: dict[str, object], context: object) -> dict[str, object]:
    return {
        "transcript_event_ranges": (
            [_range(0, 0)] if _is_positive(payload) else []
        )
    }


def _candidate_handler(resolved: object, payload: dict[str, object], context: object) -> dict[str, object]:
    return {"candidate_ranges": ([_range(0, 0)] if _is_positive(payload) else [])}


def _v3_proposal_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {"episode_proposals": ([_range(0, 0)] if _is_positive(payload) else [])}


def _v3_refinement_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {"episode_refinements": [_range(0, 0)]}


def _v4_condition_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {"condition_descriptions": ["The requested event is directly visible."]}


def _v4_grounding_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {
        "condition_evidence": [
            {"condition_id": "condition_0", **_range(0, 0)}
        ]
    }


def _v5_condition_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {
        "condition_specs": [
            {
                "description": "A necessary contextual qualification is present.",
                "role": "gate",
            },
            {
                "description": "The requested target event directly occurs.",
                "role": "anchor",
            },
        ]
    }


def _v5_grounding_handler(
    resolved: object, payload: dict[str, object], context: object
) -> dict[str, object]:
    return {
        "condition_evidence": [
            {"condition_id": "condition_0", **_range(1, 1)},
            {"condition_id": "condition_1", **_range(0, 0)},
        ]
    }


def _assert_no_forbidden_keys(test: unittest.TestCase, rows: object) -> None:
    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                test.assertNotIn(str(key).casefold(), FORBIDDEN_INPUT_KEYS)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(rows)


if __name__ == "__main__":
    unittest.main()
