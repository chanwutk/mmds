from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mmds import StaticPromptExecutor, execute, render_query  # noqa: E402
from mmds.model import PromptSpec, ResolvedPrompt  # noqa: E402
from scripts.experiments.clip_materialization import (  # noqa: E402
    MATERIALIZATION_CONTRACT_SHA256,
)
from scripts.experiments.common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_json_object,
)
from scripts.soccernet.common import SoccerNetDataError  # noqa: E402
from scripts.soccernet.goal_eval import (  # noqa: E402
    aggregate_api_usage,
    deduplicate_predictions,
    evaluate_candidate_windows,
    evaluate_predictions,
    match_timestamps,
)
from scripts.soccernet.goal_experiment import (  # noqa: E402
    CANDIDATE_STAGE,
    EVALUATION_CONFIG_FILENAME,
    EXPERIMENT_MODEL,
    GROUND_TRUTH_FILENAME,
    MATERIALIZATION_STAGE,
    PREDECESSOR_EXPERIMENT_NAME,
    SELECTED_GAMES,
    TRANSCRIPT_VIDEO_STAGE,
    _combined_latency_measurement,
    _validate_frozen_experiment,
    compare_results,
    evaluate_results,
    experiment_directory,
    main,
    materialize_clips,
    prepare_experiment,
    run_candidates,
    run_naive,
    run_transcript_only,
    run_transcript_video,
    status,
)
from scripts.soccernet.goal_materialize import (  # noqa: E402
    MATERIALIZATION_CHECKPOINT_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    materialize_goal_candidate_clips,
    validate_goal_materialized_clip_stage,
)
from scripts.soccernet.goal_plans import (  # noqa: E402
    GOAL_LIST_SCHEMA,
    VIDEO_GOAL_LOCALIZATION_PROMPT,
    build_llm_candidate_plan,
    build_naive_plan,
    build_transcript_only_plan,
    build_transcript_video_plan,
)
from scripts.soccernet.goal_runtime import (  # noqa: E402
    ApiCallRecorder,
    CachingGeminiPromptExecutor,
)
from udfs.soccernet_goal_ops import (  # noqa: E402
    MIN_EXCLUDE_SIGNAL_COUNT,
    MIN_INCLUDE_SIGNAL_COUNT,
    merge_candidate_windows,
    normalize_candidate_clip_goals,
    normalize_naive_goals,
    normalize_signal_lexicon,
    normalize_transcript_goals,
    signal_goal_candidates,
)


def _clip_media(duration: float = 60.0) -> dict[str, object]:
    return {
        "duration_seconds": duration,
        "width": 224,
        "height": 224,
        "video_codec": "h264",
        "audio_stream_count": 1,
        "audio_codecs": ["aac"],
        "start_time_seconds": 0.0,
    }


def _static_executor(plan: object, node_name: str, response: dict) -> StaticPromptExecutor:
    node = next(node for node in plan.walk_postorder() if node.name == node_name)
    if not isinstance(node.spec, PromptSpec):
        raise AssertionError(f"{node_name} is not prompt-backed")
    executor = StaticPromptExecutor({("map", node.spec.cache_key()): response})
    executor.cache_hits = 0
    executor.cache_misses = 1
    return executor


class GoalUdfTests(unittest.TestCase):
    def test_signal_candidates_preserve_independent_includes(self) -> None:
        result = signal_goal_candidates(
            {
                "goal_signal_lexicon": {
                    "include_phrases": ["goal", "scores"],
                    "exclude_phrases": ["goal kick", "possible offside"],
                },
                "transcript_segments": [
                    {"start": 10, "end": 12, "text": "That will be a goal kick."},
                    {"start": 20, "end": 23, "text": "He scores; possible offside."},
                    {"start": 40, "end": 43, "text": "What a goal!"},
                ],
            }
        )
        self.assertEqual(
            [item["time_seconds"] for item in result["goal_candidates"]],
            [20.0, 40.0],
        )
        self.assertEqual(
            [item["time_seconds"] for item in result["excluded_signal_matches"]],
            [10.0],
        )

    def test_historical_signal_lexicon_normalization_is_mechanical(self) -> None:
        include = [
            f"positive signal number {index}"
            for index in range(MIN_INCLUDE_SIGNAL_COUNT)
        ]
        exclude = [
            f"negative context number {index}"
            for index in range(MIN_EXCLUDE_SIGNAL_COUNT)
        ]
        result = normalize_signal_lexicon(
            {
                "include_phrases": include + ["POSITIVE SIGNAL NUMBER 0", "in", None],
                "exclude_phrases": exclude,
            }
        )
        lexicon = result["goal_signal_lexicon"]
        self.assertEqual(len(lexicon["include_phrases"]), MIN_INCLUDE_SIGNAL_COUNT)
        self.assertEqual(
            [item["reason"] for item in lexicon["rejected_include_phrases"]],
            ["duplicate", "short_single_token", "not_a_string"],
        )

    def test_merge_candidate_windows_clamps_deduplicates_and_merges(self) -> None:
        result = merge_candidate_windows(
            {
                "duration_seconds": 120,
                "window_radius_seconds": 30,
                "goal_candidates": [
                    {"time_seconds": 10, "evidence": "a", "signal_type": "goal"},
                    {"time_seconds": 10, "evidence": "duplicate", "signal_type": "goal"},
                    {"time_seconds": 50, "evidence": "b", "signal_type": "lead"},
                    {"time_seconds": 115, "evidence": "c", "signal_type": "goal"},
                    {"time_seconds": 999, "evidence": "bad", "signal_type": "goal"},
                ],
            }
        )
        windows = result["candidate_windows"]
        self.assertEqual(len(windows), 2)
        self.assertEqual(
            (windows[0]["start_seconds"], windows[0]["end_seconds"]),
            (0.0, 80.0),
        )
        self.assertEqual(
            (windows[1]["start_seconds"], windows[1]["end_seconds"]),
            (85.0, 120.0),
        )

    def test_naive_timestamp_components_are_strictly_validated(self) -> None:
        result = normalize_naive_goals(
            {
                "duration_seconds": 120,
                "goals": [
                    {
                        "clip_minute": 1,
                        "clip_second": 2.5,
                        "confidence": 1.2,
                        "evidence": "goal",
                    },
                    {
                        "clip_minute": 2,
                        "clip_second": 1,
                        "confidence": 0.5,
                        "evidence": "outside",
                    },
                ],
            }
        )
        self.assertEqual(
            [goal["time_seconds"] for goal in result["goals"]],
            [62.5, 121.0],
        )
        self.assertTrue(result["goals"][0]["timestamp_valid"])
        self.assertFalse(result["goals"][1]["timestamp_valid"])
        for invalid in (
            {"clip_minute": 1.5, "clip_second": 2},
            {"clip_minute": 1, "clip_second": 60},
            {"clip_minute": True, "clip_second": 2},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_naive_goals({"duration_seconds": 120, "goals": [invalid]})

    def test_materialized_clip_offset_is_added_exactly_once(self) -> None:
        result = normalize_candidate_clip_goals(
            {
                "game_id": "game",
                "half": 2,
                "duration_seconds": 2700,
                "candidate_strategy": "llm_high_recall",
                "candidate_window": {
                    "window_id": 3,
                    "start_seconds": 100,
                    "end_seconds": 160,
                },
                "candidate_video": {"type": "Video", "path": "/tmp/clip.mp4"},
                "materialized_video_context": {
                    "game_id": "game",
                    "half": 2,
                    "duration_seconds": 60,
                    "timeline_origin_seconds": 0,
                    "media_extent": "standalone_candidate_clip",
                },
                "goals": [
                    {
                        "clip_minute": 0,
                        "clip_second": 12.5,
                        "confidence": 0.9,
                        "evidence": "goal",
                    },
                    {
                        "clip_minute": 1,
                        "clip_second": 1,
                        "confidence": 0.5,
                        "evidence": "outside",
                    },
                ],
            }
        )
        goals = result["transcript_video_goals"]
        self.assertEqual([item["time_seconds"] for item in goals], [112.5, 161.0])
        self.assertEqual(goals[0]["window_id"], 3)
        self.assertFalse(goals[1]["timestamp_valid"])

    def test_materialized_normalizer_rejects_views_and_nonzero_origin(self) -> None:
        base = {
            "game_id": "game",
            "half": 1,
            "duration_seconds": 100,
            "candidate_strategy": "llm_high_recall",
            "candidate_window": {
                "window_id": 0,
                "start_seconds": 10,
                "end_seconds": 20,
            },
            "candidate_video": {"type": "Video", "path": "/tmp/clip.mp4"},
            "materialized_video_context": {
                "game_id": "game",
                "half": 1,
                "duration_seconds": 10,
                "timeline_origin_seconds": 0,
                "media_extent": "standalone_candidate_clip",
            },
            "goals": [],
        }
        for mutation in (
            {"candidate_video": {"type": "VideoView", "source": "/tmp/full.mkv"}},
            {
                "materialized_video_context": {
                    **base["materialized_video_context"],
                    "timeline_origin_seconds": 10,
                }
            },
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                normalize_candidate_clip_goals({**base, **mutation})

    def test_transcript_only_requires_an_existing_segment_start(self) -> None:
        result = normalize_transcript_goals(
            {
                "duration_seconds": 100,
                "transcript_segments": [
                    {"start": 10, "end": 12, "text": "Goal"},
                    {"start": 20, "end": 23, "text": "Replay"},
                ],
                "transcript_goals": [
                    {"time_seconds": 10, "confidence": 0.9, "evidence": "Goal"},
                    {"time_seconds": 11, "confidence": 0.9, "evidence": "invented"},
                ],
            }
        )
        self.assertTrue(result["transcript_goals"][0]["timestamp_valid"])
        self.assertFalse(result["transcript_goals"][1]["timestamp_valid"])


class GoalPlanTests(unittest.TestCase):
    def test_v4_plans_render_without_labels_or_video_views(self) -> None:
        plans = (
            build_naive_plan("input.jsonl"),
            build_transcript_only_plan("input.jsonl"),
            build_llm_candidate_plan("input.jsonl"),
            build_transcript_video_plan("materialized.jsonl"),
        )
        for plan in plans:
            rendered = render_query(plan)
            self.assertNotIn("ground_truth", rendered)
            self.assertNotIn("Labels-v2", rendered)
        optimized = render_query(plans[-1])
        self.assertNotIn("VideoView", optimized)
        self.assertNotIn("attach_candidate_video_view", optimized)
        self.assertIn('Record["candidate_video"]', optimized)

    def test_video_methods_share_exact_prompt_schema_and_timestamp_contract(self) -> None:
        naive = next(
            node
            for node in build_naive_plan("input.jsonl").walk_postorder()
            if node.name == "full_video_goal_detection"
        )
        optimized = next(
            node
            for node in build_transcript_video_plan("clips.jsonl").walk_postorder()
            if node.name == "localize_goal_in_materialized_candidate_clip"
        )
        self.assertIsInstance(naive.spec, PromptSpec)
        self.assertIsInstance(optimized.spec, PromptSpec)
        self.assertEqual(naive.spec.output_schema, GOAL_LIST_SCHEMA)
        self.assertEqual(optimized.spec.output_schema, GOAL_LIST_SCHEMA)
        self.assertEqual(naive.spec.parts[2], VIDEO_GOAL_LOCALIZATION_PROMPT)
        self.assertEqual(optimized.spec.parts[2], VIDEO_GOAL_LOCALIZATION_PROMPT)

    def test_materialized_plan_executes_clip_local_to_source_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clips.jsonl"
            atomic_write_jsonl(
                path,
                [
                    {
                        "game_id": "game",
                        "half": 1,
                        "duration_seconds": 2700,
                        "candidate_strategy": "llm_high_recall",
                        "candidate_window": {
                            "window_id": 0,
                            "start_seconds": 100,
                            "end_seconds": 160,
                        },
                        "candidate_video": {
                            "type": "Video",
                            "path": "/tmp/clip.mp4",
                        },
                        "materialized_video_context": {
                            "game_id": "game",
                            "half": 1,
                            "duration_seconds": 60,
                            "timeline_origin_seconds": 0,
                            "media_extent": "standalone_candidate_clip",
                        },
                    }
                ],
            )
            plan = build_transcript_video_plan(str(path))
            executor = _static_executor(
                plan,
                "localize_goal_in_materialized_candidate_clip",
                {
                    "goals": [
                        {
                            "clip_minute": 0,
                            "clip_second": 12,
                            "confidence": 0.9,
                            "evidence": "goal",
                        }
                    ]
                },
            )
            rows = execute(plan, prompt_executor=executor)
        self.assertEqual(rows[0]["transcript_video_goals"]["time_seconds"], 112.0)

    def test_execute_flag_is_required_for_model_and_encoding_commands(self) -> None:
        for command in (
            "run-naive",
            "run-transcript-only",
            "run-candidates",
            "materialize-clips",
            "run-transcript-video",
        ):
            output = io.StringIO()
            with self.subTest(command=command), redirect_stdout(output):
                self.assertEqual(main([command]), 0)
            self.assertIn("Preview only", output.getvalue())


class GoalMaterializationTests(unittest.TestCase):
    def _candidate_rows(self, root: Path, *, count: int = 2) -> list[dict]:
        rows = []
        for index in range(count):
            game_id = f"league/season/game_{index}"
            source = root / game_id / "1_224p.mkv"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"source-{index}".encode())
            from scripts.experiments.common import sha256_file

            rows.append(
                {
                    "game_id": game_id,
                    "half": 1,
                    "duration_seconds": 100,
                    "video": {
                        "type": "Video",
                        "path": str(source.resolve()),
                        "sha256": sha256_file(source),
                    },
                    "candidate_strategy": "llm_high_recall",
                    "goal_candidates": [],
                    "candidate_windows": [
                        {
                            "window_id": 0,
                            "start_seconds": 10,
                            "end_seconds": 70,
                            "duration_seconds": 60,
                        }
                    ],
                }
            )
        return rows

    @staticmethod
    def _processor(source: Path, destination: Path, start: float, end: float) -> dict:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes() + f"-{start}-{end}".encode())
        return _clip_media(end - start)

    def test_materialization_produces_standalone_zero_origin_video_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates = root / "candidates.jsonl"
            atomic_write_jsonl(candidates, self._candidate_rows(root, count=1))
            stage = root / "materialized"
            result = materialize_goal_candidate_clips(
                candidates,
                stage,
                clip_processor=self._processor,
                media_probe=lambda path: _clip_media(),
            )
            validated = validate_goal_materialized_clip_stage(
                stage, media_probe=lambda path: _clip_media()
            )
        self.assertEqual(result["stage"]["materialized_clip_count"], 1)
        row = validated["rows"][0]
        self.assertEqual(row["candidate_video"]["type"], "Video")
        self.assertNotIn("source", row["candidate_video"])
        self.assertEqual(
            row["materialized_video_context"]["timeline_origin_seconds"], 0.0
        )
        self.assertEqual(
            result["stage"]["materialization_contract_sha256"],
            MATERIALIZATION_CONTRACT_SHA256,
        )

    def test_corrupt_clip_fails_content_and_media_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates = root / "candidates.jsonl"
            atomic_write_jsonl(candidates, self._candidate_rows(root, count=1))
            stage = root / "materialized"
            result = materialize_goal_candidate_clips(
                candidates,
                stage,
                clip_processor=self._processor,
                media_probe=lambda path: _clip_media(),
            )
            clip = stage / result["manifest"]["clips"][0]["path"]
            clip.write_bytes(b"corrupt")
            with self.assertRaisesRegex(SoccerNetDataError, "bytes changed"):
                validate_goal_materialized_clip_stage(
                    stage, media_probe=lambda path: _clip_media()
                )

    def test_interrupted_materialization_resumes_only_completed_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates = root / "candidates.jsonl"
            atomic_write_jsonl(candidates, self._candidate_rows(root, count=2))
            stage = root / "materialized"
            calls = 0

            def fail_second(
                source: Path, destination: Path, start: float, end: float
            ) -> dict:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise SoccerNetDataError("interrupted")
                return self._processor(source, destination, start, end)

            with self.assertRaisesRegex(SoccerNetDataError, "interrupted"):
                materialize_goal_candidate_clips(
                    candidates,
                    stage,
                    clip_processor=fail_second,
                    media_probe=lambda path: _clip_media(),
                )
            checkpoint = load_json_object(
                stage / MATERIALIZATION_CHECKPOINT_FILENAME
            )
            self.assertEqual(len(checkpoint["clips"]), 1)
            resumed = materialize_goal_candidate_clips(
                candidates,
                stage,
                clip_processor=self._processor,
                media_probe=lambda path: _clip_media(),
            )
        self.assertEqual(resumed["stage"]["reused_clip_count"], 1)
        self.assertEqual(resumed["stage"]["materialized_clip_count"], 1)
        self.assertEqual(resumed["stage"]["failed_clip_attempt_count"], 1)


class GoalEvaluationTests(unittest.TestCase):
    @staticmethod
    def _truth() -> dict:
        return {
            "games": [
                {
                    "game_id": "game",
                    "halves": {
                        "1": [{"time_seconds": 10.0}, {"time_seconds": 40.0}],
                        "2": [{"time_seconds": 20.0}],
                    },
                }
            ]
        }

    def test_matcher_maximizes_cardinality_before_minimizing_error(self) -> None:
        self.assertEqual(len(match_timestamps([4, 9], [0, 5], 5)), 2)

    def test_deduplication_prefers_valid_timestamp_over_invalid_confidence(self) -> None:
        result = deduplicate_predictions(
            [
                {
                    "game_id": "g",
                    "half": 1,
                    "time_seconds": 10,
                    "confidence": 0.2,
                    "timestamp_valid": True,
                },
                {
                    "game_id": "g",
                    "half": 1,
                    "time_seconds": 11,
                    "confidence": 1.0,
                    "timestamp_valid": False,
                },
            ]
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["timestamp_valid"])

    def test_evaluation_reports_precision_recall_and_invalid_predictions(self) -> None:
        result = evaluate_predictions(
            [
                {"game_id": "game", "half": 1, "time_seconds": 11, "confidence": 1},
                {"game_id": "game", "half": 1, "time_seconds": 70, "confidence": 1},
                {
                    "game_id": "game",
                    "half": 2,
                    "time_seconds": 20,
                    "confidence": 1,
                    "timestamp_valid": False,
                },
            ],
            self._truth(),
            tolerances=(5,),
        )
        metric = result["metrics_by_tolerance_seconds"]["5"]
        self.assertEqual(metric["true_positives"], 1)
        self.assertEqual(metric["false_positives"], 2)
        self.assertEqual(metric["false_negatives"], 2)
        self.assertEqual(metric["invalid_prediction_count"], 1)

    def test_candidate_metrics_measure_recall_and_union_duration(self) -> None:
        rows = [
            {
                "game_id": "game",
                "half": 1,
                "duration_seconds": 100,
                "candidate_windows": [
                    {"start_seconds": 0, "end_seconds": 20},
                    {"start_seconds": 70, "end_seconds": 80},
                ],
            },
            {
                "game_id": "game",
                "half": 2,
                "duration_seconds": 100,
                "candidate_windows": [],
            },
        ]
        result = evaluate_candidate_windows(rows, self._truth())
        self.assertEqual(result["covered_goal_count"], 1)
        self.assertAlmostEqual(result["candidate_recall"], 1 / 3)
        self.assertAlmostEqual(result["selectivity"], 30 / 200)

    def test_api_usage_aggregation_preserves_cost_assumptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "api_calls.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "elapsed_seconds": 2,
                        "usage_metadata": {
                            "prompt_token_count": 1_000_000,
                            "candidates_token_count": 100_000,
                            "thoughts_token_count": 100_000,
                            "total_token_count": 1_200_000,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = aggregate_api_usage(path)
        self.assertEqual(result["api_call_count"], 1)
        self.assertEqual(result["estimated_cost_usd"], 0.55)

    def test_combined_latency_includes_materialization_and_rejects_reuse(self) -> None:
        provider = {
            "end_to_end_seconds": 2,
            "cache_hits": 0,
            "api_usage": {
                "api_elapsed_seconds": 1.5,
                "failed_api_call_count": 0,
            },
            "media_uploads": {"failed_upload_count": 0},
        }
        clean = _combined_latency_measurement(
            ("candidate", provider),
            (
                "materialized",
                {
                    "stage_type": "local_candidate_clip_materialization",
                    "end_to_end_seconds": 3,
                    "reused_clip_count": 0,
                },
            ),
            ("video", provider),
        )
        self.assertTrue(clean["valid"])
        self.assertEqual(clean["end_to_end_seconds"], 7)
        reused = _combined_latency_measurement(
            (
                "materialized",
                {
                    "stage_type": "local_candidate_clip_materialization",
                    "end_to_end_seconds": 3,
                    "reused_clip_count": 1,
                },
            )
        )
        self.assertFalse(reused["valid"])
        self.assertIsNone(reused["end_to_end_seconds"])


class _FakePart:
    def __init__(self, text=None, inline_data=None, file_data=None, video_metadata=None):
        self.text = text
        self.inline_data = inline_data
        self.file_data = file_data
        self.video_metadata = video_metadata


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeTypes:
    Content = _FakeContent
    Part = _FakePart


class _FakeUsage:
    def model_dump(self, **kwargs):
        return {"prompt_token_count": 10, "candidates_token_count": 2}


class _FakeModels:
    def __init__(self):
        self.call_count = 0

    def generate_content(self, **kwargs):
        self.call_count += 1
        return type(
            "Response",
            (),
            {
                "text": '{"answer": "goal"}',
                "response_id": "response-1",
                "model_version": "test-model",
                "usage_metadata": _FakeUsage(),
            },
        )()


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


class GoalRuntimeTests(unittest.TestCase):
    def test_executor_caches_response_and_records_one_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            client = _FakeClient()
            executor = CachingGeminiPromptExecutor(
                cache_directory=directory / "cache",
                recorder=ApiCallRecorder(directory / "api_calls.jsonl"),
                model="test-model",
                client=client,
                types_module=_FakeTypes,
            )
            prompt = PromptSpec(parts=("Find goals",), output_schema={"answer": "string"})
            resolved = ResolvedPrompt(parts=("Find goals",), output_schema=prompt.output_schema)
            first = executor.execute("map", prompt, resolved, {}, {})
            second = executor.execute("map", prompt, resolved, {}, {})
        self.assertEqual(first, {"answer": "goal"})
        self.assertEqual(second, first)
        self.assertEqual(client.models.call_count, 1)
        self.assertEqual(executor.cache_hits, 1)


class SoccerV4WorkflowTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        predecessor = (
            root
            / "experiments"
            / PREDECESSOR_EXPERIMENT_NAME
            / "runs"
            / "naive"
            / "predictions.json"
        )
        predecessor.parent.mkdir(parents=True)
        predecessor.write_text('{"v3": "preserve"}\n', encoding="utf-8")
        manifest_games = []
        for game_id in SELECTED_GAMES:
            videos = {"224p": {}}
            for half in (1, 2):
                video = root / game_id / f"{half}_224p.mkv"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(f"video-{game_id}-{half}".encode())
                videos["224p"][str(half)] = {
                    "status": "valid",
                    "path": str(video.relative_to(root)),
                    "duration_seconds": 60,
                }
                transcript = root / "transcripts" / game_id / f"{half}.whisper.json"
                transcript.parent.mkdir(parents=True, exist_ok=True)
                transcript.write_text(
                    json.dumps(
                        {
                            "language": "en",
                            "segments": [
                                {"start": 12, "end": 14, "text": "It is a goal."},
                                {"start": 30, "end": 32, "text": "Play resumes."},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            (root / game_id / "Labels-v2.json").write_text(
                json.dumps(
                    {
                        "annotations": [
                            {
                                "label": "Goal",
                                "gameTime": "1 - 00:12",
                                "position": "12000",
                                "visibility": "visible",
                            },
                            {
                                "label": "Goal",
                                "gameTime": "2 - 00:12",
                                "position": "12000",
                                "visibility": "visible",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_games.append(
                {"game_id": game_id, "status": "complete", "videos": videos}
            )
        (root / "manifest.json").write_text(
            json.dumps({"games": manifest_games}), encoding="utf-8"
        )
        return predecessor

    def test_full_label_isolated_materialized_workflow_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predecessor = self._fixture(root)
            prepared = prepare_experiment(root)
            directory = experiment_directory(root)
            self.assertEqual(prepared["half_count"], 6)
            self.assertEqual(prepared["goal_count"], 6)
            self.assertEqual(prepared["materialization_contract_sha256"], MATERIALIZATION_CONTRACT_SHA256)
            prediction_config = load_json_object(directory / "prediction_config.json")
            self.assertNotIn("ground_truth_path", prediction_config)
            self.assertEqual(prediction_config["model"], EXPERIMENT_MODEL)
            self.assertEqual(
                prediction_config["prompt_sha256"]["naive"],
                prediction_config["prompt_sha256"]["transcript_video"],
            )
            self.assertNotIn(
                "VideoView",
                (directory / "plans" / "transcript_video.py").read_text(
                    encoding="utf-8"
                ),
            )
            _validate_frozen_experiment(directory)

            ground_truth_backup = directory / "ground_truth.hidden"
            evaluation_config_backup = directory / "evaluation_config.hidden"
            (directory / GROUND_TRUTH_FILENAME).rename(ground_truth_backup)
            (directory / EVALUATION_CONFIG_FILENAME).rename(evaluation_config_backup)
            for game_id in SELECTED_GAMES:
                (root / game_id / "Labels-v2.json").unlink()

            naive_plan = build_naive_plan(str((directory / "input.jsonl").resolve()))
            naive_executor = _static_executor(
                naive_plan,
                "full_video_goal_detection",
                {
                    "goals": [
                        {
                            "clip_minute": 0,
                            "clip_second": 12,
                            "confidence": 1,
                            "evidence": "goal",
                        }
                    ]
                },
            )
            naive = run_naive(root, Path(".env"), prompt_executor=naive_executor)
            self.assertEqual(len(naive["predictions"]), 6)

            transcript_plan = build_transcript_only_plan(
                str((directory / "input.jsonl").resolve())
            )
            transcript_executor = _static_executor(
                transcript_plan,
                "transcript_only_final_goal_detection",
                {
                    "transcript_goals": [
                        {
                            "time_seconds": 12,
                            "confidence": 1,
                            "evidence": "It is a goal.",
                        }
                    ]
                },
            )
            transcript = run_transcript_only(
                root, Path(".env"), prompt_executor=transcript_executor
            )
            self.assertEqual(len(transcript["predictions"]), 6)

            candidate_plan = build_llm_candidate_plan(
                str((directory / "input.jsonl").resolve())
            )
            candidate_executor = _static_executor(
                candidate_plan,
                "transcript_goal_candidate_generation",
                {
                    "goal_candidates": [
                        {
                            "time_seconds": 30,
                            "evidence": "possible goal",
                            "signal_type": "goal_call",
                            "confidence": 0.9,
                        }
                    ]
                },
            )
            candidates = run_candidates(
                root, Path(".env"), prompt_executor=candidate_executor
            )
            self.assertEqual(candidates["audit"]["candidate_window_count"], 6)

            def processor(
                source: Path, destination: Path, start: float, end: float
            ) -> dict:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes() + b"-clip")
                return _clip_media(end - start)

            materialized = materialize_clips(
                root,
                clip_processor=processor,
                media_probe=lambda path: _clip_media(),
            )
            self.assertEqual(materialized["stage"]["clip_count"], 6)
            completion = directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
            completion.unlink()
            recovered = materialize_clips(root, media_probe=lambda path: _clip_media())
            self.assertTrue(recovered["completion_recovered"])

            materialized_input = (
                directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
            )
            video_plan = build_transcript_video_plan(str(materialized_input.resolve()))
            video_executor = _static_executor(
                video_plan,
                "localize_goal_in_materialized_candidate_clip",
                {
                    "goals": [
                        {
                            "clip_minute": 0,
                            "clip_second": 12,
                            "confidence": 1,
                            "evidence": "goal",
                        }
                    ]
                },
            )
            optimized = run_transcript_video(
                root,
                Path(".env"),
                prompt_executor=video_executor,
                media_probe=lambda path: _clip_media(),
            )
            self.assertEqual(len(optimized["predictions"]), 6)
            self.assertTrue(
                all(item["time_seconds"] == 12 for item in optimized["predictions"])
            )

            ground_truth_backup.rename(directory / GROUND_TRUTH_FILENAME)
            evaluation_config_backup.rename(directory / EVALUATION_CONFIG_FILENAME)
            evaluations = evaluate_results(root, media_probe=lambda path: _clip_media())
            comparison = compare_results(root)
            self.assertEqual(
                evaluations[TRANSCRIPT_VIDEO_STAGE]["metrics_by_tolerance_seconds"]["30"]["f1"],
                1.0,
            )
            self.assertEqual(
                comparison["methods"][TRANSCRIPT_VIDEO_STAGE]["physical_stages"],
                [CANDIDATE_STAGE, MATERIALIZATION_STAGE, TRANSCRIPT_VIDEO_STAGE],
            )
            self.assertEqual(
                status(root, media_probe=lambda path: _clip_media())["comparison"],
                "complete",
            )
            self.assertEqual(predecessor.read_text(encoding="utf-8"), '{"v3": "preserve"}\n')

            prediction_path = directory / "runs" / "naive" / "predictions.json"
            original = prediction_path.read_text(encoding="utf-8")
            prediction_path.write_text(original + " ", encoding="utf-8")
            self.assertEqual(
                status(root, media_probe=lambda path: _clip_media())["stages"]["naive"],
                "invalid",
            )


if __name__ == "__main__":
    unittest.main()
