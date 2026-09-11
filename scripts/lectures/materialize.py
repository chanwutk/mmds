"""Deterministic standalone clip materialization for lecture candidate windows."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json_object,
    load_jsonl_objects,
    sha256_file,
)
from scripts.experiments.clip_materialization import (
    CLIP_DURATION_TOLERANCE_SECONDS,
    CLIP_START_TIME_TOLERANCE_SECONDS,
    MATERIALIZATION_CONTRACT,
    MATERIALIZATION_CONTRACT_SHA256,
    MATERIALIZATION_CONTRACT_VERSION,
    ClipProcessor,
    MediaProbe,
    ffmpeg_materialize_clip,
    probe_materialized_clip,
    validate_clip_media as _validate_clip_media,
    validate_frozen_clip,
)


MATERIALIZATION_MANIFEST_FILENAME = "manifest.json"
MATERIALIZED_INPUT_FILENAME = "input.jsonl"
MATERIALIZATION_STAGE_FILENAME = "stage.json"
MATERIALIZATION_CHECKPOINT_FILENAME = "checkpoint.json"


def materialize_candidate_clips(
    candidate_path: Path,
    stage_directory: Path,
    *,
    clip_processor: ClipProcessor = ffmpeg_materialize_clip,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Materialize every valid candidate window and freeze clip provenance."""
    candidate_sha256 = sha256_file(candidate_path)
    rows = load_jsonl_objects(candidate_path)
    expected = _expected_windows(rows)
    checkpoint_path = stage_directory / MATERIALIZATION_CHECKPOINT_FILENAME
    checkpoint = _load_or_initialize_checkpoint(
        checkpoint_path,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
    )
    entries_by_key = {
        _entry_key(entry): entry
        for entry in checkpoint.get("clips", [])
        if isinstance(entry, Mapping)
    }
    if len(entries_by_key) != len(checkpoint.get("clips", [])):
        raise ExperimentDataError("Materialization checkpoint has duplicate clip entries")

    started = time.perf_counter()
    reused_clip_count = 0
    materialized_clip_count = 0
    materialized_rows: list[dict[str, Any]] = []
    frozen_entries: list[dict[str, Any]] = []
    for row, window in expected:
        key = _window_key(row, window)
        destination = _clip_path(stage_directory, *key)
        existing = entries_by_key.get(key)
        if existing is not None:
            _validate_checkpoint_entry(
                existing,
                destination,
                row=row,
                window=window,
                media_probe=media_probe,
            )
            entry = dict(existing)
            reused_clip_count += 1
        else:
            entry = _materialize_one(
                destination,
                stage_directory=stage_directory,
                row=row,
                window=window,
                clip_processor=clip_processor,
            )
            entries_by_key[key] = entry
            checkpoint["clips"] = [
                entries_by_key[item_key] for item_key in sorted(entries_by_key)
            ]
            checkpoint["elapsed_seconds"] = float(
                checkpoint.get("elapsed_seconds", 0.0)
            ) + (time.perf_counter() - started)
            atomic_write_json(checkpoint_path, checkpoint)
            started = time.perf_counter()
            materialized_clip_count += 1
        frozen_entries.append(entry)
        materialized_rows.append(_materialized_input_row(row, window, entry, destination))

    elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0)) + (
        time.perf_counter() - started
    )
    manifest = {
        "schema_version": 1,
        "candidate_input_path": str(candidate_path.resolve()),
        "candidate_input_sha256": candidate_sha256,
        "materialization_contract": MATERIALIZATION_CONTRACT,
        "materialization_contract_sha256": MATERIALIZATION_CONTRACT_SHA256,
        "clip_count": len(frozen_entries),
        "clips": frozen_entries,
    }
    stage = {
        "schema_version": 1,
        "stage_type": "local_candidate_clip_materialization",
        "end_to_end_seconds": elapsed_seconds,
        "clip_count": len(frozen_entries),
        "materialized_clip_count": materialized_clip_count,
        "reused_clip_count": reused_clip_count,
        "total_requested_duration_seconds": sum(
            float(entry["requested_duration_seconds"]) for entry in frozen_entries
        ),
        "total_output_duration_seconds": sum(
            float(entry["media"]["duration_seconds"]) for entry in frozen_entries
        ),
        "total_output_bytes": sum(int(entry["size_bytes"]) for entry in frozen_entries),
        "materialization_contract_sha256": MATERIALIZATION_CONTRACT_SHA256,
    }
    atomic_write_json(stage_directory / MATERIALIZATION_MANIFEST_FILENAME, manifest)
    atomic_write_jsonl(stage_directory / MATERIALIZED_INPUT_FILENAME, materialized_rows)
    atomic_write_json(stage_directory / MATERIALIZATION_STAGE_FILENAME, stage)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return {"manifest": manifest, "stage": stage, "rows": materialized_rows}


def validate_materialized_clip_stage(
    stage_directory: Path,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Validate every frozen clip byte and its label-isolated input row."""
    manifest = load_json_object(stage_directory / MATERIALIZATION_MANIFEST_FILENAME)
    if manifest.get("materialization_contract") != MATERIALIZATION_CONTRACT:
        raise ExperimentDataError("Materialized clips use a different encoding contract")
    if manifest.get("materialization_contract_sha256") != MATERIALIZATION_CONTRACT_SHA256:
        raise ExperimentDataError("Materialized clip contract hash is invalid")
    candidate_path_value = manifest.get("candidate_input_path")
    candidate_sha256 = manifest.get("candidate_input_sha256")
    if not isinstance(candidate_path_value, str) or not isinstance(candidate_sha256, str):
        raise ExperimentDataError("Materialized clip manifest has no candidate provenance")
    candidate_path = Path(candidate_path_value)
    if not candidate_path.is_file() or sha256_file(candidate_path) != candidate_sha256:
        raise ExperimentDataError("Candidate input changed after clip materialization")
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise ExperimentDataError("Materialized clip manifest has no clip list")
    entries_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for entry in clips:
        if not isinstance(entry, Mapping):
            raise ExperimentDataError("Materialized clip entry must be an object")
        key = _entry_key(entry)
        if key in entries_by_key:
            raise ExperimentDataError(f"Duplicate materialized clip entry: {key}")
        entries_by_key[key] = entry
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            raise ExperimentDataError(f"Materialized clip has no path: {key}")
        path = stage_directory / relative_path
        _validate_frozen_clip(path, entry, media_probe=media_probe)
    rows = load_jsonl_objects(stage_directory / MATERIALIZED_INPUT_FILENAME)
    if len(rows) != len(clips):
        raise ExperimentDataError("Materialized input row count does not match clip manifest")
    row_keys: set[tuple[str, str, int]] = set()
    for row in rows:
        window = row.get("candidate_windows")
        if not isinstance(window, Mapping):
            raise ExperimentDataError("Materialized input has no candidate window")
        key = _window_key(row, window)
        if key in row_keys or key not in entries_by_key:
            raise ExperimentDataError(f"Unexpected materialized input row: {key}")
        row_keys.add(key)
        _validate_materialized_input_row(
            row,
            entries_by_key[key],
            stage_directory=stage_directory,
        )
    if row_keys != set(entries_by_key):
        raise ExperimentDataError("Materialized input rows do not match clip manifest")
    return {"manifest": manifest, "rows": rows}


def _expected_windows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    expected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen: set[tuple[str, str, int]] = set()
    verified_source_hashes: dict[Path, str] = {}
    for row in rows:
        video = row.get("video")
        windows = row.get("candidate_windows")
        if not isinstance(video, Mapping) or not isinstance(windows, list):
            raise ExperimentDataError("Candidate row has invalid video or window metadata")
        source_value = video.get("path")
        source_sha256 = video.get("sha256")
        if not isinstance(source_value, str) or not isinstance(source_sha256, str):
            raise ExperimentDataError("Candidate video must contain path and sha256")
        source = Path(source_value)
        if not source.is_file():
            raise ExperimentDataError(f"Candidate source video changed: {source}")
        resolved_source = source.resolve()
        actual_source_sha256 = verified_source_hashes.get(resolved_source)
        if actual_source_sha256 is None:
            actual_source_sha256 = sha256_file(source)
            verified_source_hashes[resolved_source] = actual_source_sha256
        if actual_source_sha256 != source_sha256:
            raise ExperimentDataError(f"Candidate source video changed: {source}")
        for window in windows:
            if not isinstance(window, Mapping):
                raise ExperimentDataError("Candidate window must be an object")
            key = _window_key(row, window)
            if key in seen:
                raise ExperimentDataError(f"Duplicate candidate window: {key}")
            seen.add(key)
            _window_bounds(window)
            expected.append((row, window))
    return expected


def _load_or_initialize_checkpoint(
    checkpoint_path: Path,
    *,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, Any]:
    if checkpoint_path.is_file():
        checkpoint = load_json_object(checkpoint_path)
        expected = (
            str(candidate_path.resolve()),
            candidate_sha256,
            MATERIALIZATION_CONTRACT_SHA256,
        )
        actual = (
            checkpoint.get("candidate_input_path"),
            checkpoint.get("candidate_input_sha256"),
            checkpoint.get("materialization_contract_sha256"),
        )
        if actual != expected:
            raise ExperimentDataError("Materialization checkpoint provenance changed")
        if not isinstance(checkpoint.get("clips"), list):
            raise ExperimentDataError("Materialization checkpoint clip list is invalid")
        return checkpoint
    return {
        "schema_version": 1,
        "candidate_input_path": str(candidate_path.resolve()),
        "candidate_input_sha256": candidate_sha256,
        "materialization_contract_sha256": MATERIALIZATION_CONTRACT_SHA256,
        "elapsed_seconds": 0.0,
        "clips": [],
    }


def _materialize_one(
    destination: Path,
    *,
    stage_directory: Path,
    row: Mapping[str, Any],
    window: Mapping[str, Any],
    clip_processor: ClipProcessor,
) -> dict[str, Any]:
    video = row["video"]
    source = Path(str(video["path"]))
    start_seconds, end_seconds = _window_bounds(window)
    staging = destination.with_name(f".{destination.stem}.part{destination.suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ExperimentDataError(f"Refusing unmanifested materialized clip: {destination}")
    if staging.exists():
        staging.unlink()
    started = time.perf_counter()
    try:
        media = dict(clip_processor(source, staging, start_seconds, end_seconds))
        _validate_clip_media(media, end_seconds - start_seconds, staging)
        if not staging.is_file() or staging.stat().st_size <= 0:
            raise ExperimentDataError(f"Clip processor produced no bytes: {staging}")
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        content_hash = sha256_file(staging)
        size_bytes = staging.stat().st_size
        os.replace(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
    return {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "window_id": int(window["window_id"]),
        "path": str(destination.relative_to(stage_directory)),
        "source_path": str(source.resolve()),
        "source_sha256": str(video["sha256"]),
        "fps": float(video.get("fps", 1.0)),
        "source_start_seconds": start_seconds,
        "source_end_seconds": end_seconds,
        "requested_duration_seconds": end_seconds - start_seconds,
        "size_bytes": size_bytes,
        "sha256": content_hash,
        "media": media,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "complete",
    }


def _materialized_input_row(
    row: Mapping[str, Any],
    window: Mapping[str, Any],
    entry: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    video = row["video"]
    return {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "query_text": str(row["query_text"]),
        "duration_seconds": float(row["duration_seconds"]),
        "candidate_windows": dict(window),
        "candidate_video": {
            "type": "Video",
            "path": str(destination.resolve()),
            "sha256": str(entry["sha256"]),
            "mime_type": "video/mp4",
            "fps": float(video.get("fps", 1.0)),
        },
        "materialized_video_context": {
            "lecture_id": str(row["lecture_id"]),
            "duration_seconds": float(entry["media"]["duration_seconds"]),
            "timeline_origin_seconds": 0.0,
            "media_extent": "standalone_candidate_clip",
        },
    }


def _validate_checkpoint_entry(
    entry: Mapping[str, Any],
    destination: Path,
    *,
    row: Mapping[str, Any],
    window: Mapping[str, Any],
    media_probe: MediaProbe,
) -> None:
    start_seconds, end_seconds = _window_bounds(window)
    expected = (
        str(row["lecture_id"]),
        str(row["query_id"]),
        int(window["window_id"]),
        str(Path(str(row["video"]["path"])).resolve()),
        str(row["video"]["sha256"]),
        float(row["video"].get("fps", 1.0)),
        start_seconds,
        end_seconds,
    )
    actual = (
        entry.get("lecture_id"),
        entry.get("query_id"),
        entry.get("window_id"),
        entry.get("source_path"),
        entry.get("source_sha256"),
        entry.get("fps"),
        entry.get("source_start_seconds"),
        entry.get("source_end_seconds"),
    )
    if actual != expected:
        raise ExperimentDataError(f"Materialized clip checkpoint changed: {expected[:3]}")
    _validate_frozen_clip(destination, entry, media_probe=media_probe)


def _validate_materialized_input_row(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    stage_directory: Path,
) -> None:
    video = row.get("candidate_video")
    window = row.get("candidate_windows")
    context = row.get("materialized_video_context")
    if not isinstance(video, Mapping) or not isinstance(window, Mapping):
        raise ExperimentDataError("Materialized input media fields must be objects")
    if not isinstance(context, Mapping):
        raise ExperimentDataError("Materialized input context must be an object")
    clip_path = stage_directory / str(entry["path"])
    expected_video = {
        "type": "Video",
        "path": str(clip_path.resolve()),
        "sha256": str(entry["sha256"]),
        "mime_type": "video/mp4",
        "fps": float(entry["fps"]),
    }
    if dict(video) != expected_video:
        raise ExperimentDataError("Materialized input video does not match clip manifest")
    try:
        expected_window = (
            int(entry["window_id"]),
            float(entry["source_start_seconds"]),
            float(entry["source_end_seconds"]),
        )
        actual_window = (
            int(window.get("window_id", -1)),
            float(window.get("start_seconds", -1)),
            float(window.get("end_seconds", -1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataError(
            "Materialized input window metadata is invalid"
        ) from exc
    if actual_window != expected_window:
        raise ExperimentDataError("Materialized input window does not match clip manifest")
    expected_context = {
        "lecture_id": str(entry["lecture_id"]),
        "duration_seconds": float(entry["media"]["duration_seconds"]),
        "timeline_origin_seconds": 0.0,
        "media_extent": "standalone_candidate_clip",
    }
    if dict(context) != expected_context:
        raise ExperimentDataError("Materialized input exposes an invalid video context")


def _validate_frozen_clip(
    path: Path,
    entry: Mapping[str, Any],
    *,
    media_probe: MediaProbe,
) -> None:
    validate_frozen_clip(path, entry, media_probe=media_probe)


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        raw_window_id = entry["window_id"]
        if (
            not isinstance(raw_window_id, int)
            or isinstance(raw_window_id, bool)
            or raw_window_id < 0
        ):
            raise ValueError
        return (
            str(entry["lecture_id"]),
            str(entry["query_id"]),
            raw_window_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataError("Materialized clip entry key is invalid") from exc


def _window_key(
    row: Mapping[str, Any], window: Mapping[str, Any]
) -> tuple[str, str, int]:
    try:
        raw_window_id = window["window_id"]
        if (
            not isinstance(raw_window_id, int)
            or isinstance(raw_window_id, bool)
            or raw_window_id < 0
        ):
            raise ValueError
        return (
            str(row["lecture_id"]),
            str(row["query_id"]),
            raw_window_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataError("Candidate window key is invalid") from exc


def _window_bounds(window: Mapping[str, Any]) -> tuple[float, float]:
    try:
        start = float(window["start_seconds"])
        end = float(window["end_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataError("Candidate window boundaries are invalid") from exc
    if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end):
        raise ExperimentDataError("Candidate window must satisfy 0 <= start < end")
    return start, end


def _clip_path(
    stage_directory: Path, lecture_id: str, query_id: str, window_id: int
) -> Path:
    for label, value in (("lecture_id", lecture_id), ("query_id", query_id)):
        safe_characters = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )
        if not value or any(character not in safe_characters for character in value):
            raise ExperimentDataError(f"Unsafe {label} in candidate window: {value!r}")
    if window_id < 0:
        raise ExperimentDataError("Candidate window_id must be non-negative")
    return stage_directory / "clips" / lecture_id / query_id / f"window_{window_id:03d}.mp4"
