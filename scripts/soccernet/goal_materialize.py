"""Durable standalone-clip materialization for SoccerNet goal candidates."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.experiments.clip_materialization import (
    MATERIALIZATION_CONTRACT,
    MATERIALIZATION_CONTRACT_SHA256,
    ClipProcessor,
    MediaProbe,
    ffmpeg_materialize_clip,
    probe_materialized_clip,
    validate_clip_media,
    validate_frozen_clip,
)
from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    atomic_write_jsonl,
    load_json_object,
    load_jsonl_objects,
    sha256_file,
)

from .common import SoccerNetDataError, game_directory


MATERIALIZATION_MANIFEST_FILENAME = "manifest.json"
MATERIALIZED_INPUT_FILENAME = "input.jsonl"
MATERIALIZATION_STAGE_FILENAME = "stage.json"
MATERIALIZATION_CHECKPOINT_FILENAME = "checkpoint.json"


def materialize_goal_candidate_clips(
    candidate_path: Path,
    stage_directory: Path,
    *,
    clip_processor: ClipProcessor = ffmpeg_materialize_clip,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Materialize all candidate windows serially with resumable checkpoints."""
    candidate_sha256 = sha256_file(candidate_path)
    rows = load_jsonl_objects(candidate_path, error_type=SoccerNetDataError)
    expected = _expected_windows(rows)
    checkpoint_path = stage_directory / MATERIALIZATION_CHECKPOINT_FILENAME
    checkpoint = _load_or_initialize_checkpoint(
        checkpoint_path,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
    )
    raw_checkpoint_entries = checkpoint.get("clips")
    if not isinstance(raw_checkpoint_entries, list):
        raise SoccerNetDataError("Materialization checkpoint clip list is invalid")
    entries_by_key = {
        _entry_key(entry): dict(entry)
        for entry in raw_checkpoint_entries
        if isinstance(entry, Mapping)
    }
    if len(entries_by_key) != len(raw_checkpoint_entries):
        raise SoccerNetDataError(
            "Materialization checkpoint has invalid or duplicate clip entries"
        )

    interval_started = time.perf_counter()
    elapsed_before_interval = float(checkpoint.get("elapsed_seconds", 0.0))
    reused_clip_count = 0
    materialized_clip_count = 0
    frozen_entries: list[dict[str, Any]] = []
    materialized_rows: list[dict[str, Any]] = []
    for row, window in expected:
        key = _window_key(row, window)
        destination = _clip_path(stage_directory, *key)
        existing = entries_by_key.get(key)
        if existing is not None:
            _validate_checkpoint_entry(
                existing,
                destination,
                stage_directory=stage_directory,
                row=row,
                window=window,
                media_probe=media_probe,
            )
            entry = existing
            reused_clip_count += 1
        else:
            try:
                entry = _materialize_one(
                    destination,
                    stage_directory=stage_directory,
                    row=row,
                    window=window,
                    clip_processor=clip_processor,
                )
            except Exception:
                elapsed_before_interval += time.perf_counter() - interval_started
                checkpoint["elapsed_seconds"] = elapsed_before_interval
                checkpoint["failed_clip_attempt_count"] = int(
                    checkpoint.get("failed_clip_attempt_count", 0)
                ) + 1
                atomic_write_json(checkpoint_path, checkpoint)
                raise
            entries_by_key[key] = entry
            elapsed_before_interval += time.perf_counter() - interval_started
            checkpoint["elapsed_seconds"] = elapsed_before_interval
            checkpoint["clips"] = [
                entries_by_key[item_key] for item_key in sorted(entries_by_key)
            ]
            atomic_write_json(checkpoint_path, checkpoint)
            interval_started = time.perf_counter()
            materialized_clip_count += 1
        frozen_entries.append(entry)
        materialized_rows.append(
            _materialized_input_row(row, window, entry, destination)
        )

    elapsed_seconds = elapsed_before_interval + time.perf_counter() - interval_started
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
        "max_in_flight_encodes": 1,
        "end_to_end_seconds": elapsed_seconds,
        "clip_count": len(frozen_entries),
        "materialized_clip_count": materialized_clip_count,
        "reused_clip_count": reused_clip_count,
        "failed_clip_attempt_count": int(
            checkpoint.get("failed_clip_attempt_count", 0)
        ),
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


def validate_goal_materialized_clip_stage(
    stage_directory: Path,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Validate candidate provenance, every clip, and every model input row."""
    manifest = load_json_object(
        stage_directory / MATERIALIZATION_MANIFEST_FILENAME,
        error_type=SoccerNetDataError,
    )
    if manifest.get("materialization_contract") != MATERIALIZATION_CONTRACT:
        raise SoccerNetDataError("Materialized clips use a different encoding contract")
    if manifest.get("materialization_contract_sha256") != MATERIALIZATION_CONTRACT_SHA256:
        raise SoccerNetDataError("Materialized clip contract hash is invalid")
    candidate_value = manifest.get("candidate_input_path")
    candidate_sha256 = manifest.get("candidate_input_sha256")
    if not isinstance(candidate_value, str) or not isinstance(candidate_sha256, str):
        raise SoccerNetDataError("Materialized manifest has no candidate provenance")
    candidate_path = Path(candidate_value)
    if not candidate_path.is_file() or sha256_file(candidate_path) != candidate_sha256:
        raise SoccerNetDataError("Candidate input changed after clip materialization")

    expected = _expected_windows(
        load_jsonl_objects(candidate_path, error_type=SoccerNetDataError)
    )
    raw_entries = manifest.get("clips")
    if not isinstance(raw_entries, list):
        raise SoccerNetDataError("Materialized clip manifest has no clip list")
    entries_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise SoccerNetDataError("Materialized clip entry must be an object")
        key = _entry_key(raw_entry)
        if key in entries_by_key:
            raise SoccerNetDataError(f"Duplicate materialized clip entry: {key}")
        entries_by_key[key] = raw_entry
        relative = raw_entry.get("path")
        if not isinstance(relative, str):
            raise SoccerNetDataError(f"Materialized clip has no path: {key}")
        _safe_relative_clip_path(stage_directory, relative)

    expected_keys = {_window_key(row, window) for row, window in expected}
    if set(entries_by_key) != expected_keys or manifest.get("clip_count") != len(expected):
        raise SoccerNetDataError("Materialized clip manifest differs from candidates")
    for row, window in expected:
        _validate_checkpoint_entry(
            entries_by_key[_window_key(row, window)],
            _clip_path(stage_directory, *_window_key(row, window)),
            stage_directory=stage_directory,
            row=row,
            window=window,
            media_probe=media_probe,
        )

    rows = load_jsonl_objects(
        stage_directory / MATERIALIZED_INPUT_FILENAME,
        error_type=SoccerNetDataError,
    )
    if len(rows) != len(expected):
        raise SoccerNetDataError("Materialized input row count differs from candidates")
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        window = row.get("candidate_window")
        if not isinstance(window, Mapping):
            raise SoccerNetDataError("Materialized input has no candidate_window")
        key = _window_key(row, window)
        if key in seen or key not in entries_by_key:
            raise SoccerNetDataError(f"Unexpected materialized input row: {key}")
        seen.add(key)
        _validate_materialized_input_row(
            row,
            entries_by_key[key],
            stage_directory=stage_directory,
        )
    if seen != expected_keys:
        raise SoccerNetDataError("Materialized input rows differ from clip manifest")
    return {"manifest": manifest, "rows": rows}


def _expected_windows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    expected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen: set[tuple[str, int, int]] = set()
    verified_source_hashes: dict[Path, str] = {}
    seen_halves: set[tuple[str, int]] = set()
    for row in rows:
        game_id, half = _row_key(row)
        if (game_id, half) in seen_halves:
            raise SoccerNetDataError(f"Duplicate candidate half: {(game_id, half)}")
        seen_halves.add((game_id, half))
        if row.get("candidate_strategy") != "llm_high_recall":
            raise SoccerNetDataError("Candidate row uses a non-paper candidate strategy")
        video = row.get("video")
        windows = row.get("candidate_windows")
        if not isinstance(video, Mapping) or not isinstance(windows, list):
            raise SoccerNetDataError("Candidate row has invalid video or window metadata")
        source_value = video.get("path")
        source_sha256 = video.get("sha256")
        if not isinstance(source_value, str) or not isinstance(source_sha256, str):
            raise SoccerNetDataError("Candidate video must contain path and sha256")
        source = Path(source_value)
        if not source.is_file():
            raise SoccerNetDataError(f"Candidate source video changed: {source}")
        resolved_source = source.resolve()
        actual_hash = verified_source_hashes.get(resolved_source)
        if actual_hash is None:
            actual_hash = sha256_file(source)
            verified_source_hashes[resolved_source] = actual_hash
        if actual_hash != source_sha256:
            raise SoccerNetDataError(f"Candidate source video changed: {source}")
        duration = _positive_finite(row.get("duration_seconds"), "half duration")
        for window in windows:
            if not isinstance(window, Mapping):
                raise SoccerNetDataError("Candidate window must be an object")
            key = _window_key(row, window)
            if key in seen:
                raise SoccerNetDataError(f"Duplicate candidate window: {key}")
            seen.add(key)
            _, end = _window_bounds(window)
            if end > duration:
                raise SoccerNetDataError(f"Candidate window exceeds half duration: {key}")
            expected.append((row, window))
    return expected


def _load_or_initialize_checkpoint(
    checkpoint_path: Path,
    *,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, Any]:
    expected = (
        str(candidate_path.resolve()),
        candidate_sha256,
        MATERIALIZATION_CONTRACT_SHA256,
    )
    if checkpoint_path.is_file():
        checkpoint = load_json_object(checkpoint_path, error_type=SoccerNetDataError)
        actual = (
            checkpoint.get("candidate_input_path"),
            checkpoint.get("candidate_input_sha256"),
            checkpoint.get("materialization_contract_sha256"),
        )
        if actual != expected:
            raise SoccerNetDataError("Materialization checkpoint provenance changed")
        return checkpoint
    return {
        "schema_version": 1,
        "candidate_input_path": expected[0],
        "candidate_input_sha256": candidate_sha256,
        "materialization_contract_sha256": MATERIALIZATION_CONTRACT_SHA256,
        "elapsed_seconds": 0.0,
        "failed_clip_attempt_count": 0,
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
        raise SoccerNetDataError(f"Refusing unmanifested materialized clip: {destination}")
    if staging.exists():
        staging.unlink()
    started = time.perf_counter()
    try:
        try:
            media = dict(clip_processor(source, staging, start_seconds, end_seconds))
        except ExperimentDataError as exc:
            raise SoccerNetDataError(str(exc)) from exc
        validate_clip_media(
            media,
            end_seconds - start_seconds,
            staging,
            error_type=SoccerNetDataError,
        )
        if not staging.is_file() or staging.stat().st_size <= 0:
            raise SoccerNetDataError(f"Clip processor produced no bytes: {staging}")
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        content_hash = sha256_file(staging)
        size_bytes = staging.stat().st_size
        os.replace(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
    game_id, half = _row_key(row)
    return {
        "game_id": game_id,
        "half": half,
        "window_id": _window_id(window),
        "path": str(destination.relative_to(stage_directory)),
        "source_path": str(source.resolve()),
        "source_sha256": str(video["sha256"]),
        "source_duration_seconds": float(row["duration_seconds"]),
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
    game_id, half = _row_key(row)
    return {
        "game_id": game_id,
        "half": half,
        "duration_seconds": float(row["duration_seconds"]),
        "candidate_strategy": "llm_high_recall",
        "candidate_window": dict(window),
        "candidate_video": {
            "type": "Video",
            "path": str(destination.resolve()),
            "sha256": str(entry["sha256"]),
            "mime_type": "video/mp4",
        },
        "materialized_video_context": {
            "game_id": game_id,
            "half": half,
            "duration_seconds": float(entry["media"]["duration_seconds"]),
            "timeline_origin_seconds": 0.0,
            "media_extent": "standalone_candidate_clip",
        },
    }


def _validate_checkpoint_entry(
    entry: Mapping[str, Any],
    destination: Path,
    *,
    stage_directory: Path,
    row: Mapping[str, Any],
    window: Mapping[str, Any],
    media_probe: MediaProbe,
) -> None:
    game_id, half = _row_key(row)
    start, end = _window_bounds(window)
    expected = (
        game_id,
        half,
        _window_id(window),
        str(Path(str(row["video"]["path"])).resolve()),
        str(row["video"]["sha256"]),
        float(row["duration_seconds"]),
        start,
        end,
        str(destination.relative_to(stage_directory)),
    )
    actual = (
        entry.get("game_id"),
        entry.get("half"),
        entry.get("window_id"),
        entry.get("source_path"),
        entry.get("source_sha256"),
        entry.get("source_duration_seconds"),
        entry.get("source_start_seconds"),
        entry.get("source_end_seconds"),
        entry.get("path"),
    )
    if actual != expected:
        raise SoccerNetDataError(f"Materialized clip checkpoint changed: {expected[:3]}")
    validate_frozen_clip(
        destination,
        entry,
        media_probe=media_probe,
        error_type=SoccerNetDataError,
    )


def _validate_materialized_input_row(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    stage_directory: Path,
) -> None:
    game_id, half = _row_key(row)
    video = row.get("candidate_video")
    window = row.get("candidate_window")
    context = row.get("materialized_video_context")
    if not isinstance(video, Mapping) or not isinstance(window, Mapping):
        raise SoccerNetDataError("Materialized input media fields must be objects")
    if not isinstance(context, Mapping):
        raise SoccerNetDataError("Materialized input context must be an object")
    clip_path = _safe_relative_clip_path(stage_directory, str(entry["path"]))
    expected_video = {
        "type": "Video",
        "path": str(clip_path.resolve()),
        "sha256": str(entry["sha256"]),
        "mime_type": "video/mp4",
    }
    if dict(video) != expected_video:
        raise SoccerNetDataError("Materialized input video differs from clip manifest")
    expected_window = (
        int(entry["window_id"]),
        float(entry["source_start_seconds"]),
        float(entry["source_end_seconds"]),
    )
    actual_window = (
        _window_id(window),
        float(window.get("start_seconds", -1)),
        float(window.get("end_seconds", -1)),
    )
    if actual_window != expected_window:
        raise SoccerNetDataError("Materialized input window differs from clip manifest")
    expected_context = {
        "game_id": game_id,
        "half": half,
        "duration_seconds": float(entry["media"]["duration_seconds"]),
        "timeline_origin_seconds": 0.0,
        "media_extent": "standalone_candidate_clip",
    }
    if dict(context) != expected_context:
        raise SoccerNetDataError("Materialized input exposes an invalid video context")
    if row.get("candidate_strategy") != "llm_high_recall":
        raise SoccerNetDataError("Materialized input uses a non-paper candidate strategy")
    if float(row.get("duration_seconds", -1)) != float(
        entry["source_duration_seconds"]
    ):
        raise SoccerNetDataError("Materialized input has the wrong source duration")


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        return _validated_key(entry["game_id"], entry["half"], entry["window_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SoccerNetDataError("Materialized clip entry key is invalid") from exc


def _window_key(
    row: Mapping[str, Any], window: Mapping[str, Any]
) -> tuple[str, int, int]:
    game_id, half = _row_key(row)
    return game_id, half, _window_id(window)


def _row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    try:
        game_id, half, _ = _validated_key(row["game_id"], row["half"], 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise SoccerNetDataError("Candidate row key is invalid") from exc
    return game_id, half


def _validated_key(game_id: Any, half: Any, window_id: Any) -> tuple[str, int, int]:
    if not isinstance(game_id, str) or not game_id:
        raise ValueError
    # game_directory performs the canonical absolute/traversal check.
    game_directory(Path("."), game_id)
    if not isinstance(half, int) or isinstance(half, bool) or half not in (1, 2):
        raise ValueError
    if (
        not isinstance(window_id, int)
        or isinstance(window_id, bool)
        or window_id < 0
    ):
        raise ValueError
    return game_id, half, window_id


def _window_id(window: Mapping[str, Any]) -> int:
    raw = window.get("window_id")
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise SoccerNetDataError("Candidate window_id must be a non-negative integer")
    return raw


def _window_bounds(window: Mapping[str, Any]) -> tuple[float, float]:
    try:
        start = float(window["start_seconds"])
        end = float(window["end_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SoccerNetDataError("Candidate window boundaries are invalid") from exc
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise SoccerNetDataError("Candidate window must satisfy 0 <= start < end")
    return start, end


def _clip_path(
    stage_directory: Path, game_id: str, half: int, window_id: int
) -> Path:
    _validated_key(game_id, half, window_id)
    game_root = game_directory(stage_directory / "clips", game_id)
    return game_root / f"half_{half}" / f"window_{window_id:03d}.mp4"


def _safe_relative_clip_path(stage_directory: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise SoccerNetDataError(f"Unsafe materialized clip path: {relative!r}")
    path = stage_directory / value
    try:
        path.resolve().relative_to(stage_directory.resolve())
    except ValueError as exc:
        raise SoccerNetDataError(
            f"Materialized clip path escapes its stage: {relative!r}"
        ) from exc
    return path


def _positive_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise SoccerNetDataError(f"{label} must be finite and positive")
    return float(value)


__all__ = [
    "MATERIALIZATION_CHECKPOINT_FILENAME",
    "MATERIALIZATION_MANIFEST_FILENAME",
    "MATERIALIZATION_STAGE_FILENAME",
    "MATERIALIZED_INPUT_FILENAME",
    "materialize_goal_candidate_clips",
    "validate_goal_materialized_clip_stage",
]
