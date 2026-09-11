from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mmds import StaticPromptExecutor
from mmds.model import PromptSpec

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    load_json_object,
    sha256_file,
)
from scripts.experiments.gemini_runtime import aggregate_media_uploads
from scripts.experiments.whisper import NORMALIZATION_CONTRACT_VERSION
from scripts.lecture_paper_eval import artifacts, workflow
from scripts.lecture_paper_eval.catalog import (
    DEFAULT_PROFILE,
    EvaluationProfile,
    VERIFIED_3PAIR_PROFILE,
    WHISPER_MODEL,
    source_catalog_payload,
)
from scripts.lecture_paper_eval.evaluation import build_superseded_ground_truth
from scripts.lecture_paper_eval.experiment import main
from scripts.lecture_paper_eval.plans import (
    build_candidate_plan,
    build_naive_plan,
    build_transcript_only_plan,
    build_transcript_video_plan,
)


class LecturePaperEvalWorkflowTests(unittest.TestCase):
    def test_prediction_stage_never_opens_labels_or_evaluation_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_frozen_sources(root)
            workflow.prepare(root)
            original_workflow_loader = workflow.load_json_object
            original_artifact_loader = artifacts.load_json_object

            def reject_labels(path: Path, *args: object, **kwargs: object) -> dict[str, object]:
                if Path(path).name in {
                    workflow.GROUND_TRUTH_FILENAME,
                    workflow.EVALUATION_CONFIG_FILENAME,
                }:
                    raise AssertionError(f"prediction attempted label access: {path}")
                loader = (
                    original_artifact_loader
                    if Path(path).name
                    in {
                        "manifest.json",
                        "index.json",
                        workflow.PREDICTION_CONFIG_FILENAME,
                        workflow.PREDICTION_PREFIX_COMPLETION_FILENAME,
                    }
                    or Path(path).name.endswith(".whisper.json")
                    else original_workflow_loader
                )
                return loader(path, *args, **kwargs)

            with (
                patch.object(workflow, "load_json_object", side_effect=reject_labels),
                patch.object(artifacts, "load_json_object", side_effect=reject_labels),
            ):
                result = workflow.run_naive(
                    root,
                    prompt_executor=_executor(
                        build_naive_plan("unused.jsonl"), lambda payload: {"events": []}
                    ),
                )
            self.assertEqual(result["predictions"], [])
            completion = load_json_object(
                workflow.experiment_directory(root)
                / "runs"
                / workflow.NAIVE_STAGE
                / "completion.json"
            )
            dependency_names = set(completion["dependencies_sha256"])
            self.assertNotIn("ground_truth", dependency_names)
            self.assertNotIn("evaluation_config", dependency_names)

    def test_ground_truth_change_does_not_block_later_prediction_but_blocks_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_frozen_sources(root)
            workflow.prepare(root)
            directory = workflow.experiment_directory(root)
            truth_path = directory / workflow.GROUND_TRUTH_FILENAME
            truth = load_json_object(truth_path)
            truth["annotator"] = "tampered after freeze"
            atomic_write_json(truth_path, truth)
            result = workflow.run_transcript_only(
                root,
                prompt_executor=_executor(
                    build_transcript_only_plan("unused.jsonl"),
                    lambda payload: {"transcript_event_ranges": []},
                ),
            )
            self.assertEqual(result["predictions"], [])
            with self.assertRaises(ExperimentDataError):
                workflow._validate_prepare(root)

    def test_annotation_amendment_preserves_v1_and_completed_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_frozen_sources(root)
            workflow.prepare(root)
            workflow.run_naive(
                root,
                prompt_executor=_executor(
                    build_naive_plan("unused.jsonl"), lambda payload: {"events": []}
                ),
            )
            workflow.run_transcript_only(
                root,
                prompt_executor=_executor(
                    build_transcript_only_plan("unused.jsonl"),
                    lambda payload: {"transcript_event_ranges": []},
                ),
            )
            _replace_preparation_with_annotation_v1(root)
            directory = workflow.experiment_directory(root)
            before = workflow.status(root, media_probe=_fake_media_probe)
            self.assertIn(
                "requires the completed annotation-v2 amendment",
                before["prepared"],
            )
            preserved_paths = (
                directory / workflow.GROUND_TRUTH_FILENAME,
                directory / workflow.PREDICTION_PREFIX_COMPLETION_FILENAME,
                directory / "runs" / workflow.NAIVE_STAGE / "predictions.json",
                directory
                / "runs"
                / workflow.TRANSCRIPT_ONLY_STAGE
                / "predictions.json",
            )
            hashes_before = {path: sha256_file(path) for path in preserved_paths}

            result = workflow.amend_ground_truth(root)

            self.assertEqual(result["from_annotation_version"], 1)
            self.assertEqual(result["to_annotation_version"], 2)
            self.assertEqual(result["event_count"], 7)
            self.assertFalse(result["prediction_artifacts_changed"])
            self.assertEqual(
                {path: sha256_file(path) for path in preserved_paths}, hashes_before
            )
            amended = load_json_object(
                directory / artifacts.AMENDED_GROUND_TRUTH_FILENAME
            )
            self.assertEqual(amended["annotation_version"], 2)
            self.assertEqual(amended["event_count"], 7)
            status = workflow.status(root, media_probe=_fake_media_probe)
            self.assertEqual(status["prepared"], "complete")
            self.assertEqual(
                status["annotation"],
                {"status": "amended", "version": 2, "event_count": 7},
            )
            self.assertEqual(status["stages"][workflow.NAIVE_STAGE], "complete")
            self.assertEqual(
                status["stages"][workflow.TRANSCRIPT_ONLY_STAGE], "complete"
            )
            with self.assertRaises(ExperimentDataError):
                workflow.amend_ground_truth(root)

    def test_full_eight_stage_workflow_and_stale_artifact_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_frozen_sources(root)
            prepared = workflow.prepare(root)
            self.assertEqual(prepared["pair_count"], 16)
            self.assertEqual(prepared["negative_pair_count"], 12)

            call_counts: dict[str, int] = {}

            def count(name: str, response: dict[str, object]):
                def handler(payload: dict[str, object]) -> dict[str, object]:
                    call_counts[name] = call_counts.get(name, 0) + 1
                    return response

                return handler

            workflow.run_naive(
                root,
                prompt_executor=_executor(
                    build_naive_plan("unused.jsonl"), count("naive", {"events": []})
                ),
            )
            workflow.run_transcript_only(
                root,
                prompt_executor=_executor(
                    build_transcript_only_plan("unused.jsonl"),
                    count("transcript_only", {"transcript_event_ranges": []}),
                ),
            )
            workflow.run_candidates(
                root,
                prompt_executor=_executor(
                    build_candidate_plan("unused.jsonl"),
                    count("candidates", {"candidate_ranges": [_range(0, 1)]}),
                ),
            )
            workflow.materialize_clips(
                root,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_media_probe,
            )
            workflow.run_transcript_video(
                root,
                prompt_executor=_executor(
                    build_transcript_video_plan("unused.jsonl"),
                    count("transcript_video", {"events": []}),
                ),
                media_probe=_fake_media_probe,
            )
            evaluations = workflow.evaluate(root)
            comparison = workflow.compare(root)
            self.assertEqual(set(evaluations), {"naive", "transcript_only", "transcript_video", "candidates"})
            self.assertEqual(set(comparison["methods"]), {"naive", "transcript_only", "transcript_video"})
            self.assertEqual(
                call_counts,
                {
                    "naive": 16,
                    "transcript_only": 16,
                    "candidates": 16,
                    "transcript_video": 16,
                },
            )
            current = workflow.status(root, media_probe=_fake_media_probe)
            self.assertEqual(current["prepared"], "complete")
            self.assertTrue(
                all(value == "complete" for value in current["stages"].values()),
                current,
            )
            self.assertEqual(current["comparison"], "complete")
            with self.assertRaises(ExperimentDataError):
                workflow.amend_ground_truth(root)

            prediction_path = (
                workflow.experiment_directory(root)
                / "runs"
                / workflow.NAIVE_STAGE
                / "predictions.json"
            )
            prediction_path.write_text("{}\n", encoding="utf-8")
            current = workflow.status(root, media_probe=_fake_media_probe)
            self.assertTrue(current["stages"][workflow.NAIVE_STAGE].startswith("invalid:"))

    def test_external_commands_are_preview_only_without_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--root", temporary, "download"])
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["preview_only"])
            self.assertFalse((Path(temporary) / "manifest.json").exists())

    def test_verified_three_pair_profile_runs_the_full_offline_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = VERIFIED_3PAIR_PROFILE
            _write_frozen_sources(root, profile=profile)
            prepared = workflow.prepare(root, profile=profile)
            self.assertEqual(prepared["pair_count"], 3)
            self.assertEqual(prepared["positive_pair_count"], 3)
            self.assertEqual(prepared["negative_pair_count"], 0)
            self.assertEqual(prepared["event_count"], 4)

            call_counts: dict[str, int] = {}

            def count(name: str, response: dict[str, object]):
                def handler(payload: dict[str, object]) -> dict[str, object]:
                    call_counts[name] = call_counts.get(name, 0) + 1
                    return response

                return handler

            workflow.run_naive(
                root,
                profile=profile,
                prompt_executor=_executor(
                    build_naive_plan("unused.jsonl"),
                    count("naive", {"events": []}),
                ),
            )
            workflow.run_transcript_only(
                root,
                profile=profile,
                prompt_executor=_executor(
                    build_transcript_only_plan("unused.jsonl"),
                    count("transcript_only", {"transcript_event_ranges": []}),
                ),
            )
            workflow.run_candidates(
                root,
                profile=profile,
                prompt_executor=_executor(
                    build_candidate_plan("unused.jsonl"),
                    count("candidates", {"candidate_ranges": [_range(0, 1)]}),
                ),
            )
            workflow.materialize_clips(
                root,
                profile=profile,
                clip_processor=_fake_clip_processor,
                media_probe=_fake_media_probe,
            )
            workflow.run_transcript_video(
                root,
                profile=profile,
                prompt_executor=_executor(
                    build_transcript_video_plan("unused.jsonl"),
                    count("transcript_video", {"events": []}),
                ),
                media_probe=_fake_media_probe,
            )
            evaluations = workflow.evaluate(root, profile=profile)
            comparison = workflow.compare(root, profile=profile)
            self.assertEqual(
                call_counts,
                {
                    "naive": 3,
                    "transcript_only": 3,
                    "candidates": 3,
                    "transcript_video": 3,
                },
            )
            self.assertEqual(evaluations["naive"]["pair_count"], 3)
            self.assertEqual(comparison["experiment"], profile.experiment_name)
            current = workflow.status(
                root, profile=profile, media_probe=_fake_media_probe
            )
            self.assertEqual(current["profile"], profile.profile_id)
            self.assertEqual(current["prepared"], "complete")
            self.assertEqual(current["comparison"], "complete")

    def test_media_upload_summary_separates_upload_reuse_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "media_uploads.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in (
                        {"path": "a.mp4", "size_bytes": 10, "elapsed_seconds": 2, "status": "ok"},
                        {"path": "a.mp4", "size_bytes": 10, "elapsed_seconds": 0, "status": "reused"},
                        {"path": "b.mp4", "size_bytes": 20, "elapsed_seconds": 3, "status": "error"},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            result = aggregate_media_uploads(path)
            self.assertEqual(result["successful_upload_count"], 1)
            self.assertEqual(result["reused_upload_count"], 1)
            self.assertEqual(result["failed_upload_count"], 1)
            self.assertEqual(result["uploaded_bytes"], 10)
            self.assertEqual(result["media_upload_elapsed_seconds"], 5)


def _write_frozen_sources(
    root: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    manifest_entries = []
    transcript_entries = []
    for lecture in profile.lectures:
        video_path = root / "videos" / lecture.filename
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"video:{lecture.lecture_id}".encode())
        relative_video = str(video_path.relative_to(root))
        video_sha = sha256_file(video_path)
        manifest_entries.append(
            {
                "lecture_id": lecture.lecture_id,
                "title": lecture.title,
                "source_page_url": lecture.page_url,
                "download_url": lecture.download_url,
                "path": relative_video,
                "size_bytes": video_path.stat().st_size,
                "sha256": video_sha,
                "media": {
                    "duration_seconds": 5000.0,
                    "width": 640,
                    "height": 360,
                    "video_codec": "h264",
                    "audio_stream_count": 1,
                    "audio_codecs": ["aac"],
                },
                "status": "complete",
            }
        )
        transcript_directory = root / "transcripts"
        transcript_directory.mkdir(parents=True, exist_ok=True)
        raw_path = transcript_directory / f"{lecture.lecture_id}.whisper.raw.json"
        atomic_write_json(raw_path, {"raw": lecture.lecture_id})
        segments = [
            {"segment_id": 0, "start_seconds": 10.0, "end_seconds": 20.0, "text": "setup"},
            {"segment_id": 1, "start_seconds": 20.0, "end_seconds": 30.0, "text": "demonstration"},
        ]
        transcript_path = transcript_directory / f"{lecture.lecture_id}.whisper.json"
        atomic_write_json(
            transcript_path,
            {
                "schema_version": 1,
                "source_id": lecture.lecture_id,
                "source_path": relative_video,
                "source_sha256": video_sha,
                "source_duration_seconds": 5000.0,
                "model": WHISPER_MODEL,
                "language": "en",
                "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
                "raw_checkpoint_path": str(raw_path.relative_to(root)),
                "raw_checkpoint_sha256": sha256_file(raw_path),
                "elapsed_seconds": 1.0,
                "text": "setup demonstration",
                "segments": segments,
            },
        )
        (transcript_directory / f"{lecture.lecture_id}.txt").write_text(
            "setup demonstration\n", encoding="utf-8"
        )
        transcript_entries.append(
            {
                "lecture_id": lecture.lecture_id,
                "path": str(transcript_path.relative_to(root)),
                "segment_count": 2,
                "elapsed_seconds": 1.0,
                "status": "complete",
            }
        )
    atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "source_catalog": source_catalog_payload(profile),
            "lectures": manifest_entries,
        },
    )
    atomic_write_json(
        root / "transcripts" / "index.json",
        {
            "schema_version": 1,
            "model": WHISPER_MODEL,
            "lectures": transcript_entries,
        },
    )


def _replace_preparation_with_annotation_v1(root: Path) -> None:
    directory = workflow.experiment_directory(root)
    videos = artifacts.validated_video_entries(root)
    transcripts = artifacts.validated_transcripts(root, videos)
    truth_path = directory / workflow.GROUND_TRUTH_FILENAME
    truth = build_superseded_ground_truth(videos)
    atomic_write_json(truth_path, truth)
    atomic_write_json(
        directory / workflow.EVALUATION_CONFIG_FILENAME,
        {
            "schema_version": 1,
            "experiment": truth["experiment"],
            "ground_truth_path": str(truth_path.resolve()),
            "ground_truth_sha256": sha256_file(truth_path),
            "pair_count": 16,
            "positive_pair_count": 4,
            "negative_pair_count": 12,
            "event_count": 6,
        },
    )
    plans = artifacts.rendered_plans(directory)
    artifacts.write_completion(
        directory / workflow.PREPARE_COMPLETION_FILENAME,
        stage_name="prepare",
        directory=directory,
        artifact_names=(
            workflow.INPUT_FILENAME,
            workflow.GROUND_TRUTH_FILENAME,
            workflow.PREDICTION_CONFIG_FILENAME,
            workflow.EVALUATION_CONFIG_FILENAME,
            workflow.PREDICTION_PREFIX_COMPLETION_FILENAME,
            *(f"plans/{name}.py" for name in sorted(plans)),
        ),
        dependencies={
            "source_manifest": root / "manifest.json",
            **{
                f"transcript:{lecture_id}": transcript["path"]
                for lecture_id, transcript in transcripts.items()
            },
        },
    )
def _executor(plan: object, handler: object) -> StaticPromptExecutor:
    prompt_nodes = [
        node
        for node in plan.walk_postorder()  # type: ignore[attr-defined]
        if isinstance(node.spec, PromptSpec)
    ]
    if len(prompt_nodes) != 1:
        raise AssertionError(prompt_nodes)
    return StaticPromptExecutor(
        {
            ("map", prompt_nodes[0].spec.cache_key()): (
                lambda resolved, payload, context: handler(payload)  # type: ignore[operator]
            )
        }
    )


def _range(start: int, end: int) -> dict[str, object]:
    return {
        "start_segment_id": start,
        "end_segment_id": end,
        "confidence": 0.9,
        "evidence": "possible current demonstration",
    }


def _fake_clip_processor(
    source: Path, destination: Path, start: float, end: float
) -> dict[str, object]:
    del source
    duration = end - start
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(duration), encoding="utf-8")
    return _fake_media(duration)


def _fake_media_probe(path: Path) -> dict[str, object]:
    return _fake_media(float(path.read_text(encoding="utf-8")))


def _fake_media(duration: float) -> dict[str, object]:
    return {
        "duration_seconds": duration,
        "width": 640,
        "height": 360,
        "video_codec": "h264",
        "audio_stream_count": 1,
        "audio_codecs": ["aac"],
        "start_time_seconds": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
