"""Staged, validated, resumable downloads for the approved MIT lectures."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    load_json_object,
    sha256_file,
)

from .catalog import DEFAULT_ROOT, LECTURES, LectureSource, source_catalog_payload


MANIFEST_FILENAME = "manifest.json"


def _parse_ffprobe(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ExperimentDataError(f"ffprobe returned no stream list for {path}")
    video_streams = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
    ]
    if not video_streams:
        raise ExperimentDataError(f"Downloaded lecture has no video stream: {path}")
    if not audio_streams:
        raise ExperimentDataError(f"Downloaded lecture has no audio stream: {path}")
    video = video_streams[0]
    duration_value = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_value)
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataError(f"Invalid ffprobe metadata for {path}") from exc
    if duration <= 0 or width <= 0 or height <= 0:
        raise ExperimentDataError(f"Non-positive media metadata for {path}")
    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "video_codec": str(video.get("codec_name", "unknown")),
        "audio_stream_count": len(audio_streams),
        "audio_codecs": [str(stream.get("codec_name", "unknown")) for stream in audio_streams],
    }


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExperimentDataError(
            f"ffprobe failed for {path}: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentDataError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(payload, Mapping):
        raise ExperimentDataError(f"ffprobe returned a non-object for {path}")
    return _parse_ffprobe(payload, path)


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def _download_to_staging(
    source: LectureSource,
    staging_path: Path,
    *,
    open_request: Callable[[Request], Any],
) -> None:
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    existing_size = staging_path.stat().st_size if staging_path.exists() else 0
    headers = {"User-Agent": "mmds-lecture-experiment/1"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = Request(source.download_url, headers=headers)
    with open_request(request) as response:
        status = _response_status(response)
        append = existing_size > 0 and status == 206
        if existing_size > 0 and status not in {200, 206}:
            raise ExperimentDataError(
                f"Cannot resume {source.lecture_id}: HTTP status {status}"
            )
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing_size}-"):
                raise ExperimentDataError(
                    f"Invalid Content-Range while resuming {source.lecture_id}: "
                    f"{content_range!r}"
                )
        mode = "ab" if append else "wb"
        with staging_path.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if not staging_path.is_file() or staging_path.stat().st_size == 0:
        raise ExperimentDataError(f"Download produced an empty file for {source.lecture_id}")


def _entry_for_source(
    source: LectureSource,
    *,
    relative_path: str,
    size_bytes: int,
    sha256: str,
    media: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "lecture_id": source.lecture_id,
        "title": source.title,
        "source_page_url": source.page_url,
        "download_url": source.download_url,
        "path": relative_path,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "media": dict(media),
        "status": status,
    }


def _validate_complete_entry(root: Path, source: LectureSource, entry: Mapping[str, Any]) -> None:
    if entry.get("download_url") != source.download_url:
        raise ExperimentDataError(f"Frozen source URL changed for {source.lecture_id}")
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise ExperimentDataError(f"Manifest path missing for {source.lecture_id}")
    path = root / relative
    if not path.is_file():
        raise ExperimentDataError(f"Completed lecture file is missing: {path}")
    if sha256_file(path) != entry.get("sha256"):
        raise ExperimentDataError(f"Completed lecture hash mismatch: {path}")


def download_lectures(
    *,
    root: Path = DEFAULT_ROOT,
    sources: Sequence[LectureSource] = LECTURES,
    source_catalog: Mapping[str, Any] | None = None,
    open_request: Callable[[Request], Any] = urlopen,
    media_probe: Callable[[Path], Mapping[str, Any]] = probe_media,
    promote_file: Callable[[Path, Path], None] = os.replace,
) -> dict[str, Any]:
    frozen_source_catalog = dict(
        source_catalog_payload() if source_catalog is None else source_catalog
    )
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = load_json_object(manifest_path)
        if manifest.get("source_catalog") != frozen_source_catalog:
            raise ExperimentDataError("Existing lecture manifest uses a different catalog")
    else:
        manifest = {
            "schema_version": 1,
            "source_catalog": frozen_source_catalog,
            "lectures": [],
        }
    entries = {
        str(entry.get("lecture_id")): entry
        for entry in manifest.get("lectures", [])
        if isinstance(entry, Mapping)
    }

    for source in sources:
        destination = root / "videos" / source.filename
        staging = destination.with_suffix(destination.suffix + ".part")
        existing = entries.get(source.lecture_id)
        if isinstance(existing, Mapping) and existing.get("status") == "complete":
            _validate_complete_entry(root, source, existing)
            continue
        if destination.exists() and not isinstance(existing, Mapping):
            raise ExperimentDataError(
                f"Refusing unmanifested destination for {source.lecture_id}: {destination}"
            )

        if isinstance(existing, Mapping) and existing.get("status") == "validated_staging":
            expected_hash = existing.get("sha256")
            candidate = destination if destination.is_file() else staging
            if not candidate.is_file() or sha256_file(candidate) != expected_hash:
                raise ExperimentDataError(
                    f"Validated staged file is missing or changed for {source.lecture_id}"
                )
            if candidate == staging:
                destination.parent.mkdir(parents=True, exist_ok=True)
                promote_file(staging, destination)
            completed = dict(existing)
            completed["status"] = "complete"
            entries[source.lecture_id] = completed
            manifest["lectures"] = [entries[item.lecture_id] for item in sources if item.lecture_id in entries]
            atomic_write_json(manifest_path, manifest)
            continue

        _download_to_staging(source, staging, open_request=open_request)
        media = dict(media_probe(staging))
        content_hash = sha256_file(staging)
        relative = str(destination.relative_to(root))
        staged_entry = _entry_for_source(
            source,
            relative_path=relative,
            size_bytes=staging.stat().st_size,
            sha256=content_hash,
            media=media,
            status="validated_staging",
        )
        entries[source.lecture_id] = staged_entry
        manifest["lectures"] = [entries[item.lecture_id] for item in sources if item.lecture_id in entries]
        atomic_write_json(manifest_path, manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        promote_file(staging, destination)
        staged_entry["status"] = "complete"
        atomic_write_json(manifest_path, manifest)

    return manifest
