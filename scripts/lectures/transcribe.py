"""One-time, resumable Whisper materialization for lecture videos."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    atomic_write_text,
    load_json_object,
    sha256_file,
)
from scripts.experiments.whisper import (
    NORMALIZATION_CONTRACT_VERSION,
    TranscriptionModel,
    WhisperTranscriptionModel,
    normalize_transcription,
)

from .catalog import DEFAULT_ROOT, LANGUAGE, WHISPER_MODEL
from .download import MANIFEST_FILENAME


TRANSCRIPT_DIRECTORY = "transcripts"
RAW_CHECKPOINT_SUFFIX = ".whisper.raw.json"


def _transcript_paths(root: Path, lecture_id: str) -> tuple[Path, Path]:
    directory = root / TRANSCRIPT_DIRECTORY
    return directory / f"{lecture_id}.whisper.json", directory / f"{lecture_id}.txt"


def _raw_checkpoint_path(root: Path, lecture_id: str) -> Path:
    return root / TRANSCRIPT_DIRECTORY / f"{lecture_id}{RAW_CHECKPOINT_SUFFIX}"


def _validate_completed(
    json_path: Path,
    text_path: Path,
    raw_path: Path,
    *,
    lecture_id: str,
    source_path: str,
    source_sha256: str,
    model_name: str,
    source_duration_seconds: float,
) -> bool:
    if not json_path.exists() and not text_path.exists():
        return False
    if not json_path.is_file():
        raise ExperimentDataError(
            f"Transcript text exists without metadata for {lecture_id}"
        )
    payload = load_json_object(json_path)
    expected = (lecture_id, source_path, source_sha256, model_name)
    actual = (
        payload.get("source_id"),
        payload.get("source_path"),
        payload.get("source_sha256"),
        payload.get("model"),
    )
    if actual != expected:
        raise ExperimentDataError(f"Existing transcript metadata changed for {lecture_id}")
    raw_hash = payload.get("raw_checkpoint_sha256")
    if (
        not isinstance(raw_hash, str)
        or not raw_path.is_file()
        or sha256_file(raw_path) != raw_hash
    ):
        raise ExperimentDataError(
            f"Raw Whisper checkpoint is missing or changed for {lecture_id}"
        )
    contract_version = payload.get("normalization_contract_version")
    if contract_version != NORMALIZATION_CONTRACT_VERSION:
        if contract_version is None or (
            isinstance(contract_version, int)
            and not isinstance(contract_version, bool)
            and contract_version < NORMALIZATION_CONTRACT_VERSION
        ):
            # A matching raw checkpoint can deterministically rebuild this
            # stale normalized artifact without another model invocation.
            return False
        raise ExperimentDataError(
            f"Unsupported transcript normalization contract for {lecture_id}: "
            f"{contract_version!r}"
        )
    normalized_duration = payload.get("source_duration_seconds")
    if (
        not isinstance(normalized_duration, (int, float))
        or isinstance(normalized_duration, bool)
        or not math.isclose(
            float(normalized_duration), source_duration_seconds, abs_tol=1e-9
        )
    ):
        raise ExperimentDataError(
            f"Transcript source duration changed for {lecture_id}"
        )
    expected_text = str(payload.get("text", "")).strip()
    if not text_path.exists():
        atomic_write_text(text_path, expected_text)
    elif text_path.read_text(encoding="utf-8").strip() != expected_text:
        raise ExperimentDataError(f"Transcript text does not match JSON for {lecture_id}")
    return True


def _load_raw_checkpoint(
    raw_path: Path,
    *,
    lecture_id: str,
    source_path: str,
    source_sha256: str,
    model_name: str,
    language: str,
) -> tuple[dict[str, Any], float] | None:
    if not raw_path.exists():
        return None
    payload = load_json_object(raw_path)
    expected = (
        1,
        lecture_id,
        source_path,
        source_sha256,
        model_name,
        language,
    )
    actual = (
        payload.get("schema_version"),
        payload.get("source_id"),
        payload.get("source_path"),
        payload.get("source_sha256"),
        payload.get("model"),
        payload.get("requested_language"),
    )
    if actual != expected:
        raise ExperimentDataError(
            f"Raw Whisper checkpoint provenance changed for {lecture_id}"
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ExperimentDataError(
            f"Raw Whisper checkpoint has no result object for {lecture_id}"
        )
    elapsed = payload.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise ExperimentDataError(
            f"Raw Whisper checkpoint has invalid elapsed time for {lecture_id}"
        )
    return result, float(elapsed)


def _write_raw_checkpoint(
    raw_path: Path,
    result: Mapping[str, Any],
    *,
    lecture_id: str,
    source_path: str,
    source_sha256: str,
    model_name: str,
    language: str,
    elapsed_seconds: float,
) -> None:
    atomic_write_json(
        raw_path,
        {
            "schema_version": 1,
            "source_id": lecture_id,
            "source_path": source_path,
            "source_sha256": source_sha256,
            "model": model_name,
            "requested_language": language,
            "elapsed_seconds": elapsed_seconds,
            "result": dict(result),
        },
    )


def transcribe_lectures(
    *,
    root: Path = DEFAULT_ROOT,
    model: TranscriptionModel,
    model_name: str = WHISPER_MODEL,
    language: str = LANGUAGE,
) -> dict[str, Any]:
    manifest = load_json_object(root / MANIFEST_FILENAME)
    lectures = manifest.get("lectures")
    if not isinstance(lectures, list) or not lectures:
        raise ExperimentDataError("Lecture download manifest contains no lectures")
    completed: list[dict[str, Any]] = []
    for entry in lectures:
        if not isinstance(entry, Mapping) or entry.get("status") != "complete":
            raise ExperimentDataError("All lecture downloads must be complete before transcription")
        lecture_id = str(entry.get("lecture_id", ""))
        relative_path = str(entry.get("path", ""))
        source_hash = str(entry.get("sha256", ""))
        media = entry.get("media")
        if not isinstance(media, Mapping):
            raise ExperimentDataError(
                f"Lecture manifest has no media metadata for {lecture_id}"
            )
        duration_value = media.get("duration_seconds")
        if (
            not isinstance(duration_value, (int, float))
            or isinstance(duration_value, bool)
            or not math.isfinite(float(duration_value))
            or float(duration_value) <= 0
        ):
            raise ExperimentDataError(
                f"Lecture manifest has invalid duration for {lecture_id}"
            )
        source_duration = float(duration_value)
        video_path = root / relative_path
        if not video_path.is_file() or sha256_file(video_path) != source_hash:
            raise ExperimentDataError(
                f"Lecture video is missing or changed before transcription: {video_path}"
            )
        json_path, text_path = _transcript_paths(root, lecture_id)
        raw_path = _raw_checkpoint_path(root, lecture_id)
        if _validate_completed(
            json_path,
            text_path,
            raw_path,
            lecture_id=lecture_id,
            source_path=relative_path,
            source_sha256=source_hash,
            model_name=model_name,
            source_duration_seconds=source_duration,
        ):
            payload = load_json_object(json_path)
            completed.append(
                {
                    "lecture_id": lecture_id,
                    "path": str(json_path.relative_to(root)),
                    "segment_count": len(payload.get("segments", [])),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "status": "complete",
                }
            )
            continue
        checkpoint = _load_raw_checkpoint(
            raw_path,
            lecture_id=lecture_id,
            source_path=relative_path,
            source_sha256=source_hash,
            model_name=model_name,
            language=language,
        )
        if checkpoint is None:
            started = time.perf_counter()
            raw = model.transcribe(video_path, language=language)
            elapsed = time.perf_counter() - started
            if not isinstance(raw, Mapping):
                raise ExperimentDataError(
                    f"Whisper returned a non-object for {lecture_id}"
                )
            _write_raw_checkpoint(
                raw_path,
                raw,
                lecture_id=lecture_id,
                source_path=relative_path,
                source_sha256=source_hash,
                model_name=model_name,
                language=language,
                elapsed_seconds=elapsed,
            )
        else:
            raw, elapsed = checkpoint
        normalized = normalize_transcription(
            raw,
            source_id=lecture_id,
            source_path=relative_path,
            source_sha256=source_hash,
            model_name=model_name,
            source_duration_seconds=source_duration,
        )
        if normalized["language"].casefold() not in {"en", "eng", "english"}:
            raise ExperimentDataError(
                f"Whisper did not identify English for {lecture_id}: "
                f"{normalized['language']!r}"
            )
        normalized["elapsed_seconds"] = elapsed
        normalized["raw_checkpoint_path"] = str(raw_path.relative_to(root))
        normalized["raw_checkpoint_sha256"] = sha256_file(raw_path)
        atomic_write_json(json_path, normalized)
        atomic_write_text(text_path, normalized["text"])
        completed.append(
            {
                "lecture_id": lecture_id,
                "path": str(json_path.relative_to(root)),
                "segment_count": len(normalized["segments"]),
                "elapsed_seconds": elapsed,
                "status": "complete",
            }
        )
        atomic_write_json(
            root / TRANSCRIPT_DIRECTORY / "index.json",
            {"schema_version": 1, "model": model_name, "lectures": completed},
        )
    index = {"schema_version": 1, "model": model_name, "lectures": completed}
    atomic_write_json(root / TRANSCRIPT_DIRECTORY / "index.json", index)
    return index


def create_whisper_model(
    model_name: str = WHISPER_MODEL, device: str | None = None
) -> WhisperTranscriptionModel:
    return WhisperTranscriptionModel(model_name, device=device)
