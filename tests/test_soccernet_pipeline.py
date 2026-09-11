from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.soccernet import audit_audio, download, transcribe  # noqa: E402
from scripts.soccernet.common import (  # noqa: E402
    MediaProbe,
    SoccerNetDataError,
    _parse_ffprobe,
    game_directory,
    load_secret,
    select_first_games,
    validate_labels,
)


def _write_labels(path: Path, *, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations = [
        {
            "gameTime": "1 - 12:34",
            "label": "Goal",
            "position": "754000",
            "visibility": "shown",
        },
        {
            "gameTime": "2 - 45:00",
            "label": "Goal",
            "position": "2700000",
            "visibility": "not shown",
        },
        {"gameTime": "1 - 01:02", "label": "Foul", "position": "62000"},
    ]
    if malformed:
        annotations[0]["position"] = "bad"
    path.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")


def _probe(height: int = 224, *, duration: float = 2700.0) -> MediaProbe:
    return MediaProbe(
        duration_seconds=duration,
        width=398 if height == 224 else 1280,
        height=height,
        video_codec="h264",
        audio_stream_count=1,
        audio_codecs=("aac",),
    )


class CommonTests(unittest.TestCase):
    def test_select_first_games_preserves_official_order(self) -> None:
        self.assertEqual(select_first_games(["g2", "g1", "g3"], 2), ["g2", "g1"])

    def test_select_first_games_rejects_invalid_limits_and_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            select_first_games(["g1"], 0)
        with self.assertRaises(ValueError):
            select_first_games(["g1"], 2)
        with self.assertRaises(SoccerNetDataError):
            select_first_games(["g1", "g1"], 2)

    def test_game_directory_rejects_traversal(self) -> None:
        with self.assertRaises(SoccerNetDataError):
            game_directory(Path("/dataset"), "../secret")

    def test_load_secret_supports_export_spacing_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                'OTHER=x\n export SOCCERNET_PASSWORD = "correct horse"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    load_secret("SOCCERNET_PASSWORD", env_file), "correct horse"
                )

    def test_load_secret_prefers_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SOCCERNET_PASSWORD=file-value\n", encoding="utf-8")
            with patch.dict(
                os.environ, {"SOCCERNET_PASSWORD": "process-value"}, clear=True
            ):
                self.assertEqual(
                    load_secret("SOCCERNET_PASSWORD", env_file), "process-value"
                )

    def test_load_secret_never_evaluates_shell_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SOCCERNET_PASSWORD=$(unsafe)\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SoccerNetDataError):
                    load_secret("SOCCERNET_PASSWORD", env_file)

    def test_validate_labels_counts_shown_and_not_shown_goals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "Labels-v2.json"
            _write_labels(path)
            summary = validate_labels(path)
        self.assertEqual(summary.annotation_count, 3)
        self.assertEqual(summary.goal_count, 2)
        self.assertEqual(summary.shown_goal_count, 1)
        self.assertEqual(summary.not_shown_goal_count, 1)

    def test_validate_labels_rejects_invalid_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "Labels-v2.json"
            _write_labels(path, malformed=True)
            with self.assertRaises(SoccerNetDataError):
                validate_labels(path)

    def test_parse_ffprobe_extracts_video_and_audio_metadata(self) -> None:
        payload = {
            "streams": [
                {"codec_type": "video", "width": 398, "height": 224, "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "2701.5"},
        }
        probe = _parse_ffprobe(payload, Path("game.mkv"))
        self.assertEqual(probe.height, 224)
        self.assertEqual(probe.audio_stream_count, 1)
        self.assertAlmostEqual(probe.duration_seconds, 2701.5)

    def test_parse_ffprobe_rejects_media_without_video(self) -> None:
        with self.assertRaises(SoccerNetDataError):
            _parse_ffprobe(
                {"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}},
                Path("audio.mka"),
            )


class SelectionTests(unittest.TestCase):
    def test_make_selection_records_first_strategy_and_both_resolutions(self) -> None:
        with patch.object(download, "_package_version", return_value="test"):
            selection = download.make_selection(
                games=["g1", "g2", "g3"],
                split="train",
                limit=2,
                resolutions=["224p", "720p"],
            )
        self.assertEqual(selection["games"], ["g1", "g2"])
        self.assertEqual(selection["strategy"], "first_in_official_registry")
        self.assertEqual(selection["resolutions"], ["224p", "720p"])

    def test_expected_files_are_labels_then_both_halves_per_resolution(self) -> None:
        selection = {
            "games": ["g1"],
            "resolutions": ["224p", "720p"],
        }
        self.assertEqual(
            download.expected_files(selection),
            [
                ("g1", "Labels-v2.json", None),
                ("g1", "1_224p.mkv", "224p"),
                ("g1", "2_224p.mkv", "224p"),
                ("g1", "1_720p.mkv", "720p"),
                ("g1", "2_720p.mkv", "720p"),
            ],
        )

    def test_freeze_selection_refuses_to_change_existing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            initial = {
                "schema_version": 1,
                "split": "train",
                "strategy": "first_in_official_registry",
                "limit": 1,
                "resolutions": ["224p"],
                "games": ["g1"],
            }
            download.freeze_selection(root, initial)
            changed = {**initial, "games": ["g2"]}
            with self.assertRaises(SoccerNetDataError):
                download.freeze_selection(root, changed)


class FakeDownloadBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def download_file(
        self,
        *,
        game_id: str,
        filename: str,
        split: str,
        destination_root: Path,
        password: str,
        verbose: bool,
    ) -> None:
        self.calls.append((game_id, filename, password))
        path = game_directory(destination_root, game_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "Labels-v2.json":
            _write_labels(path)
        else:
            path.write_bytes(b"fake-video")


class DownloadTests(unittest.TestCase):
    def _selection(self) -> dict:
        return {
            "schema_version": 1,
            "dataset": "SoccerNet Action Spotting",
            "task": "spotting",
            "split": "train",
            "strategy": "first_in_official_registry",
            "limit": 1,
            "resolutions": ["224p", "720p"],
            "label_filename": "Labels-v2.json",
            "soccernet_version": "test",
            "games": ["league/season/game"],
        }

    @staticmethod
    def _probe_for_path(path: Path) -> MediaProbe:
        return _probe(720 if "720p" in path.name else 224)

    def test_download_is_staged_validated_manifested_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            backend = FakeDownloadBackend()
            with patch.object(download, "probe_media", side_effect=self._probe_for_path):
                manifest = download.download_selection(
                    root=root,
                    selection=self._selection(),
                    password="top-secret",
                    minimum_free_gib=0,
                    backend=backend,
                    verbose=False,
                )
                first_call_count = len(backend.calls)
                second_manifest = download.download_selection(
                    root=root,
                    selection=self._selection(),
                    password="top-secret",
                    minimum_free_gib=0,
                    backend=backend,
                    verbose=False,
                )

            self.assertEqual(first_call_count, 5)
            self.assertEqual(len(backend.calls), first_call_count)
            self.assertEqual(manifest["summary"]["complete_game_count"], 1)
            self.assertEqual(
                manifest["summary"]["resolution_comparison"]["paired_half_count"], 2
            )
            self.assertEqual(second_manifest["summary"]["complete_game_count"], 1)
            self.assertNotIn("top-secret", json.dumps(manifest))
            self.assertTrue((root / download.MANIFEST_FILENAME).is_file())

    def test_invalid_existing_destination_is_never_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            destination = root / "league/season/game/Labels-v2.json"
            _write_labels(destination, malformed=True)
            backend = FakeDownloadBackend()
            with self.assertRaises(SoccerNetDataError):
                download.download_selection(
                    root=root,
                    selection=self._selection(),
                    password="secret",
                    minimum_free_gib=0,
                    backend=backend,
                    verbose=False,
                )
            self.assertEqual(backend.calls, [])

    def test_valid_interrupted_staging_file_is_promoted_without_download(self) -> None:
        selection = {**self._selection(), "resolutions": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staged = root / ".partial/league/season/game/Labels-v2.json"
            _write_labels(staged)
            backend = FakeDownloadBackend()
            manifest = download.download_selection(
                root=root,
                selection=selection,
                password="secret",
                minimum_free_gib=0,
                backend=backend,
                verbose=False,
            )
            self.assertEqual(backend.calls, [])
            self.assertTrue((root / "league/season/game/Labels-v2.json").is_file())
            self.assertEqual(manifest["summary"]["complete_game_count"], 1)


class AudioAuditTests(unittest.TestCase):
    def test_sample_offsets_cover_middle_of_half(self) -> None:
        offsets = audit_audio.sample_offsets(2700, sample_count=3, sample_seconds=30)
        self.assertEqual(len(offsets), 3)
        self.assertGreater(offsets[0], 0)
        self.assertAlmostEqual(offsets[1], (2700 - 30) / 2)
        self.assertLess(offsets[-1], 2700 - 30)

    def test_classify_samples_requires_speech_and_reports_language_mix(self) -> None:
        quiet = audit_audio.SpeechSample(0, "en", "", 0, 0.9, False)
        self.assertEqual(
            audit_audio.classify_samples([quiet])["automatic_status"], "no_clear_speech"
        )
        speech = [
            audit_audio.SpeechSample(0, "en", "goal scored", 3, 0.1, True),
            audit_audio.SpeechSample(1, "en", "great pass there", 3, 0.1, True),
            audit_audio.SpeechSample(2, "es", "hola mundo ahora", 3, 0.1, True),
        ]
        classified = audit_audio.classify_samples(speech)
        self.assertEqual(classified["automatic_status"], "likely_english_commentary")
        self.assertAlmostEqual(classified["english_speech_fraction"], 2 / 3)
        self.assertTrue(classified["needs_manual_review"])

    def test_review_template_does_not_overwrite_manual_work(self) -> None:
        audit = {"games": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "review.csv"
            path.write_text("manual work\n", encoding="utf-8")
            with self.assertRaises(SoccerNetDataError):
                audit_audio.write_review_template(path, audit)
            self.assertEqual(path.read_text(encoding="utf-8"), "manual work\n")

    def test_audit_dataset_samples_both_halves_and_keeps_manual_gate(self) -> None:
        manifest = {
            "selection": {"resolutions": ["224p"]},
            "games": [
                {
                    "game_id": "league/season/game",
                    "status": "complete",
                    "videos": {
                        "224p": {
                            "1": {"path": "game/1.mkv", "duration_seconds": 120},
                            "2": {"path": "game/2.mkv", "duration_seconds": 120},
                        }
                    },
                }
            ],
        }

        class Analyzer:
            def analyze(self, audio_path: Path, *, start_seconds: float):
                return audit_audio.SpeechSample(
                    start_seconds, "en", "English match commentary", 3, 0.1, True
                )

        def fake_extract(video_path: Path, output_path: Path, **kwargs) -> None:
            output_path.write_bytes(b"wav")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(audit_audio, "extract_audio_sample", side_effect=fake_extract):
                audit = audit_audio.audit_dataset(
                    root=Path(tmpdir),
                    manifest=manifest,
                    analyzer=Analyzer(),
                    sample_count=2,
                    sample_seconds=10,
                )
        game = audit["games"][0]
        self.assertEqual(game["automatic_status"], "likely_english_commentary")
        self.assertEqual(len(game["halves"]), 2)
        self.assertEqual(sum(len(half["samples"]) for half in game["halves"]), 4)
        self.assertTrue(game["needs_manual_review"])


class FakeTranscriptionModel:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(self, video_path: Path, *, language: str) -> dict:
        self.calls.append(video_path)
        return {
            "language": language,
            "text": "A goal was scored.",
            "segments": [
                {
                    "id": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "A goal was scored.",
                    "no_speech_prob": 0.01,
                    "avg_logprob": -0.2,
                }
            ],
        }


class TranscriptionTests(unittest.TestCase):
    def test_review_requires_explicit_english_commentary_confirmation(self) -> None:
        fieldnames = [
            "game_id",
            "manual_has_commentary",
            "manual_language",
            "approved_for_transcription",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "review.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "game_id": "g1",
                        "manual_has_commentary": "no",
                        "manual_language": "en",
                        "approved_for_transcription": "yes",
                    }
                )
            with self.assertRaises(SoccerNetDataError):
                transcribe.load_approved_games(path)

    def test_transcription_writes_segment_json_text_and_resumes(self) -> None:
        manifest = {
            "games": [
                {
                    "game_id": "league/season/game",
                    "status": "complete",
                    "videos": {
                        "224p": {
                            "1": {"status": "valid", "path": "league/season/game/1_224p.mkv"},
                            "2": {"status": "valid", "path": "league/season/game/2_224p.mkv"},
                        }
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for half in (1, 2):
                path = root / f"league/season/game/{half}_224p.mkv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            model = FakeTranscriptionModel()
            index = transcribe.transcribe_games(
                root=root,
                manifest=manifest,
                approved_games=["league/season/game"],
                model=model,
                model_name="test",
            )
            second_index = transcribe.transcribe_games(
                root=root,
                manifest=manifest,
                approved_games=["league/season/game"],
                model=model,
                model_name="test",
            )

            self.assertEqual(len(model.calls), 2)
            self.assertEqual(index["halves"][0]["status"], "transcribed")
            self.assertEqual(second_index["halves"][0]["status"], "reused")
            output = root / "transcripts/league/season/game/1.whisper.json"
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["segments"][0]["start"], 1.0)
            self.assertEqual(
                (output.parent / "1.txt").read_text(encoding="utf-8").strip(),
                "A goal was scored.",
            )

    def test_missing_text_file_is_recovered_from_completed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            json_path = directory / "1.whisper.json"
            text_path = directory / "1.txt"
            json_path.write_text(
                json.dumps(
                    {
                        "game_id": "game",
                        "half": 1,
                        "source_path": "game/1_224p.mkv",
                        "text": "Recovered transcript",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                transcribe._validate_completed_transcript(
                    json_path,
                    text_path,
                    game_id="game",
                    half=1,
                    source_path="game/1_224p.mkv",
                )
            )
            self.assertEqual(text_path.read_text(encoding="utf-8").strip(), "Recovered transcript")


if __name__ == "__main__":
    unittest.main()
