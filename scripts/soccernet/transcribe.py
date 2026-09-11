"""Transcribe manually approved English-commentary SoccerNet games with Whisper."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .audit_audio import REVIEW_FILENAME
from .common import (
    DEFAULT_ROOT,
    SCHEMA_VERSION,
    SoccerNetDataError,
    atomic_write_json,
    game_directory,
    load_json_object,
)
from .download import MANIFEST_FILENAME


TRANSCRIPT_DIRECTORY = "transcripts"
_TRUE_VALUES = {"1", "true", "yes", "y"}
_ENGLISH_VALUES = {"en", "eng", "english"}


class TranscriptionModel(Protocol):
    def transcribe(self, video_path: Path, *, language: str) -> Mapping[str, Any]: ...


class WhisperTranscriptionModel:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SoccerNetDataError(
                "openai-whisper is not installed. Install scripts/soccernet/requirements.txt."
            ) from exc
        kwargs = {"device": device} if device else {}
        self._model = whisper.load_model(model_name, **kwargs)

    def transcribe(self, video_path: Path, *, language: str) -> Mapping[str, Any]:
        return self._model.transcribe(
            str(video_path),
            language=language,
            task="transcribe",
            verbose=False,
            condition_on_previous_text=True,
            fp16=False,
        )


def load_approved_games(path: Path) -> list[str]:
    if not path.is_file():
        raise SoccerNetDataError(f"Manual review file does not exist: {path}")
    approved: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "game_id",
            "manual_has_commentary",
            "manual_language",
            "approved_for_transcription",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SoccerNetDataError(
                f"Manual review file is missing columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            is_approved = row["approved_for_transcription"].strip().casefold() in _TRUE_VALUES
            if not is_approved:
                continue
            has_commentary = row["manual_has_commentary"].strip().casefold() in _TRUE_VALUES
            is_english = row["manual_language"].strip().casefold() in _ENGLISH_VALUES
            if not has_commentary or not is_english:
                raise SoccerNetDataError(
                    f"Review line {line_number} approves transcription without confirming "
                    "English commentary"
                )
            game_id = row["game_id"].strip()
            if not game_id:
                raise SoccerNetDataError(f"Review line {line_number} has no game_id")
            approved.append(game_id)
    if len(set(approved)) != len(approved):
        raise SoccerNetDataError("Manual review file approves a game more than once")
    if not approved:
        raise SoccerNetDataError("No games are approved for transcription")
    return approved


def _normalize_transcription(
    result: Mapping[str, Any], *, game_id: str, half: int, source_path: str, model_name: str
) -> dict[str, Any]:
    segments = []
    for segment in result.get("segments") or []:
        segments.append(
            {
                "id": int(segment.get("id", len(segments))),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment.get("text", "")).strip(),
                "no_speech_probability": float(segment.get("no_speech_prob", 0.0)),
                "average_log_probability": float(segment.get("avg_logprob", 0.0)),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "half": half,
        "source_path": source_path,
        "model": model_name,
        "language": str(result.get("language", "unknown")),
        "text": str(result.get("text", "")).strip(),
        "segments": segments,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _validate_completed_transcript(
    json_path: Path, text_path: Path, *, game_id: str, half: int, source_path: str
) -> bool:
    if not json_path.exists() and not text_path.exists():
        return False
    if not json_path.is_file():
        raise SoccerNetDataError(
            f"Transcript text exists without metadata for {game_id} half {half}"
        )
    payload = load_json_object(json_path)
    expected = (game_id, half, source_path)
    actual = (payload.get("game_id"), payload.get("half"), payload.get("source_path"))
    if actual != expected:
        raise SoccerNetDataError(
            f"Existing transcript metadata does not match {game_id} half {half}"
        )
    expected_text = str(payload.get("text", "")).strip()
    if not text_path.exists():
        # Recover the harmless second half of an interrupted two-file commit.
        _atomic_write_text(text_path, expected_text)
    elif text_path.read_text(encoding="utf-8").strip() != expected_text:
        raise SoccerNetDataError(
            f"Transcript text does not match metadata for {game_id} half {half}"
        )
    return True


def transcribe_games(
    *,
    root: Path,
    manifest: dict[str, Any],
    approved_games: Sequence[str],
    model: TranscriptionModel,
    model_name: str,
    resolution: str = "224p",
    language: str = "en",
) -> dict[str, Any]:
    manifest_games = {game["game_id"]: game for game in manifest.get("games", [])}
    unknown = [game_id for game_id in approved_games if game_id not in manifest_games]
    if unknown:
        raise SoccerNetDataError(
            f"Manual review contains {len(unknown)} games absent from the dataset manifest"
        )

    completed: list[dict[str, Any]] = []
    for game_id in approved_games:
        game = manifest_games[game_id]
        if game.get("status") != "complete":
            raise SoccerNetDataError(f"Cannot transcribe incomplete game {game_id}")
        for half in (1, 2):
            media = game.get("videos", {}).get(resolution, {}).get(str(half))
            if not media or media.get("status") != "valid":
                raise SoccerNetDataError(
                    f"No validated {resolution} video for {game_id} half {half}"
                )
            source_path = str(media["path"])
            video_path = root / source_path
            output_directory = game_directory(root / TRANSCRIPT_DIRECTORY, game_id)
            json_path = output_directory / f"{half}.whisper.json"
            text_path = output_directory / f"{half}.txt"
            if _validate_completed_transcript(
                json_path,
                text_path,
                game_id=game_id,
                half=half,
                source_path=source_path,
            ):
                completed.append(
                    {"game_id": game_id, "half": half, "status": "reused"}
                )
                continue

            result = model.transcribe(video_path, language=language)
            normalized = _normalize_transcription(
                result,
                game_id=game_id,
                half=half,
                source_path=source_path,
                model_name=model_name,
            )
            atomic_write_json(json_path, normalized)
            _atomic_write_text(text_path, normalized["text"])
            completed.append(
                {"game_id": game_id, "half": half, "status": "transcribed"}
            )

    index = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "language": language,
        "resolution": resolution,
        "approved_game_count": len(approved_games),
        "halves": completed,
    }
    atomic_write_json(root / TRANSCRIPT_DIRECTORY / "index.json", index)
    return index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--resolution", default="224p", choices=("224p", "720p"))
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--device")
    parser.add_argument("--language", default="en")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    review_file = args.review_file or args.root / REVIEW_FILENAME
    if not args.execute:
        print(
            "Preview only. Full transcription requires manual confirmation of commentary "
            f"and English in {review_file}."
        )
        return 0
    try:
        approved_games = load_approved_games(review_file)
        manifest = load_json_object(args.root / MANIFEST_FILENAME)
        model = WhisperTranscriptionModel(args.model, args.device)
        index = transcribe_games(
            root=args.root,
            manifest=manifest,
            approved_games=approved_games,
            model=model,
            model_name=args.model,
            resolution=args.resolution,
            language=args.language,
        )
        print(
            f"Prepared {len(index['halves'])} transcript halves under "
            f"{args.root / TRANSCRIPT_DIRECTORY}"
        )
        return 0
    except (SoccerNetDataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
