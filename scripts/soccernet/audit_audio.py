"""Sample SoccerNet audio to identify likely commentary and spoken language."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .common import (
    DEFAULT_ROOT,
    SCHEMA_VERSION,
    SoccerNetDataError,
    atomic_write_json,
    load_json_object,
    write_csv,
)
from .download import MANIFEST_FILENAME


AUDIT_FILENAME = "audio_audit.json"
REVIEW_FILENAME = "audio_review.csv"


@dataclass(frozen=True)
class SpeechSample:
    start_seconds: float
    language: str
    text: str
    word_count: int
    mean_no_speech_probability: float
    has_speech: bool


class SampleAnalyzer(Protocol):
    def analyze(self, audio_path: Path, *, start_seconds: float) -> SpeechSample: ...


class WhisperSampleAnalyzer:
    """One lazily loaded Whisper model shared by all sampled clips."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SoccerNetDataError(
                "openai-whisper is not installed. Install scripts/soccernet/requirements.txt."
            ) from exc
        kwargs = {"device": device} if device else {}
        self._model = whisper.load_model(model_name, **kwargs)

    def analyze(self, audio_path: Path, *, start_seconds: float) -> SpeechSample:
        result = self._model.transcribe(
            str(audio_path),
            task="transcribe",
            verbose=False,
            condition_on_previous_text=False,
            fp16=False,
        )
        text = str(result.get("text", "")).strip()
        segments = result.get("segments") or []
        probabilities = [
            float(segment["no_speech_prob"])
            for segment in segments
            if segment.get("no_speech_prob") is not None
        ]
        mean_no_speech = (
            sum(probabilities) / len(probabilities) if probabilities else 1.0
        )
        word_count = len(text.split())
        return SpeechSample(
            start_seconds=start_seconds,
            language=str(result.get("language", "unknown")),
            text=text,
            word_count=word_count,
            mean_no_speech_probability=mean_no_speech,
            has_speech=word_count >= 3 and mean_no_speech < 0.60,
        )


def sample_offsets(
    duration_seconds: float,
    *,
    sample_count: int = 3,
    sample_seconds: float = 30.0,
) -> list[float]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_count <= 0 or sample_seconds <= 0:
        raise ValueError("sample_count and sample_seconds must be positive")
    latest_start = max(0.0, duration_seconds - sample_seconds)
    if sample_count == 1 or latest_start == 0:
        return [latest_start / 2]
    # Avoid opening/closing sequences while spreading samples across the half.
    fractions = (
        0.20 + 0.60 * index / (sample_count - 1) for index in range(sample_count)
    )
    return [latest_start * fraction for fraction in fractions]


def extract_audio_sample(
    video_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    sample_seconds: float,
    ffmpeg: str = "ffmpeg",
) -> None:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{sample_seconds:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SoccerNetDataError(f"ffmpeg failed for {video_path}: {exc}") from exc
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        error = result.stderr.strip() or f"exit code {result.returncode}"
        raise SoccerNetDataError(f"Cannot extract audio from {video_path}: {error}")


def classify_samples(samples: Sequence[SpeechSample]) -> dict[str, Any]:
    speech_samples = [sample for sample in samples if sample.has_speech]
    if not speech_samples:
        return {
            "automatic_status": "no_clear_speech",
            "speech_sample_count": 0,
            "english_speech_fraction": 0.0,
            "needs_manual_review": True,
        }
    english_count = sum(sample.language.casefold() == "en" for sample in speech_samples)
    english_fraction = english_count / len(speech_samples)
    if english_fraction >= 2 / 3:
        status = "likely_english_commentary"
    elif english_fraction <= 1 / 3:
        status = "likely_non_english_commentary"
    else:
        status = "mixed_or_uncertain_language"
    return {
        "automatic_status": status,
        "speech_sample_count": len(speech_samples),
        "english_speech_fraction": english_fraction,
        # Whisper detects speech/language, not the semantic role of the speaker.
        "needs_manual_review": True,
    }


def _validate_manifest(manifest: dict[str, Any], resolution: str) -> None:
    games = manifest.get("games")
    if not isinstance(games, list) or not games:
        raise SoccerNetDataError("Dataset manifest contains no games")
    incomplete = [game.get("game_id") for game in games if game.get("status") != "complete"]
    if incomplete:
        raise SoccerNetDataError(
            f"Dataset has {len(incomplete)} incomplete games; finish validation before audio audit"
        )
    if resolution not in manifest.get("selection", {}).get("resolutions", []):
        raise SoccerNetDataError(f"Resolution {resolution!r} is absent from the manifest")


def audit_dataset(
    *,
    root: Path,
    manifest: dict[str, Any],
    analyzer: SampleAnalyzer,
    resolution: str = "224p",
    sample_count: int = 3,
    sample_seconds: float = 30.0,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    _validate_manifest(manifest, resolution)
    audited_games: list[dict[str, Any]] = []

    for game in manifest["games"]:
        samples: list[SpeechSample] = []
        half_payloads: list[dict[str, Any]] = []
        for half in ("1", "2"):
            media = game["videos"][resolution][half]
            video_path = root / media["path"]
            duration = float(media["duration_seconds"])
            half_samples: list[SpeechSample] = []
            for offset in sample_offsets(
                duration,
                sample_count=sample_count,
                sample_seconds=sample_seconds,
            ):
                with tempfile.TemporaryDirectory(prefix="soccernet-audio-") as tmpdir:
                    audio_path = Path(tmpdir) / "sample.wav"
                    extract_audio_sample(
                        video_path,
                        audio_path,
                        start_seconds=offset,
                        sample_seconds=sample_seconds,
                        ffmpeg=ffmpeg,
                    )
                    sample = analyzer.analyze(audio_path, start_seconds=offset)
                half_samples.append(sample)
                samples.append(sample)
            half_payloads.append(
                {
                    "half": int(half),
                    "video_path": media["path"],
                    "samples": [asdict(sample) for sample in half_samples],
                }
            )

        audited_games.append(
            {
                "game_id": game["game_id"],
                **classify_samples(samples),
                "halves": half_payloads,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "analyzer": "openai-whisper",
            "resolution": resolution,
            "sample_count_per_half": sample_count,
            "sample_seconds": sample_seconds,
            "interpretation": (
                "Automatic status detects likely speech and language only; manual review "
                "must confirm that speech is match commentary."
            ),
        },
        "games": audited_games,
    }


def review_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for game in audit["games"]:
        texts = [
            sample["text"]
            for half in game["halves"]
            for sample in half["samples"]
            if sample["text"]
        ]
        rows.append(
            {
                "game_id": game["game_id"],
                "automatic_status": game["automatic_status"],
                "english_speech_fraction": f"{game['english_speech_fraction']:.3f}",
                "sample_text": " | ".join(texts)[:2000],
                "manual_has_commentary": "",
                "manual_language": "",
                "approved_for_transcription": "",
            }
        )
    return rows


def write_review_template(path: Path, audit: dict[str, Any], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise SoccerNetDataError(
            f"Refusing to overwrite existing manual review file {path}; use --force-review-template"
        )
    fieldnames = (
        "game_id",
        "automatic_status",
        "english_speech_fraction",
        "sample_text",
        "manual_has_commentary",
        "manual_language",
        "approved_for_transcription",
    )
    write_csv(path, fieldnames, review_rows(audit))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resolution", default="224p", choices=("224p", "720p"))
    parser.add_argument("--model", default="small")
    parser.add_argument("--device")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--force-review-template", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print(
            "Preview only. The audit will sample "
            f"{args.sample_count} x {args.sample_seconds:g}s from each half at {args.resolution}."
        )
        return 0
    try:
        review_path = args.root / REVIEW_FILENAME
        if review_path.exists() and not args.force_review_template:
            raise SoccerNetDataError(
                f"Refusing to overwrite existing manual review file {review_path}; "
                "use --force-review-template"
            )
        manifest = load_json_object(args.root / MANIFEST_FILENAME)
        analyzer = WhisperSampleAnalyzer(args.model, args.device)
        audit = audit_dataset(
            root=args.root,
            manifest=manifest,
            analyzer=analyzer,
            resolution=args.resolution,
            sample_count=args.sample_count,
            sample_seconds=args.sample_seconds,
            ffmpeg=args.ffmpeg,
        )
        atomic_write_json(args.root / AUDIT_FILENAME, audit)
        write_review_template(
            review_path,
            audit,
            force=args.force_review_template,
        )
        print(f"Wrote {args.root / AUDIT_FILENAME} and {review_path}")
        return 0
    except (SoccerNetDataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
