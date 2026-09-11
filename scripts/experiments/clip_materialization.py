"""Shared standalone zero-origin video clip encoding and validation."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from scripts.experiments.common import ExperimentDataError, sha256_file, sha256_json


MATERIALIZATION_CONTRACT_VERSION = 1
CLIP_DURATION_TOLERANCE_SECONDS = 0.25
CLIP_START_TIME_TOLERANCE_SECONDS = 0.05

MATERIALIZATION_CONTRACT: dict[str, Any] = {
    "version": MATERIALIZATION_CONTRACT_VERSION,
    "container": "mp4",
    "video_codec": "libx264",
    "video_crf": 18,
    "video_preset": "veryfast",
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
    "audio_bitrate": "128k",
    "timestamp_policy": "standalone_zero_origin",
    "duration_tolerance_seconds": CLIP_DURATION_TOLERANCE_SECONDS,
    "start_time_tolerance_seconds": CLIP_START_TIME_TOLERANCE_SECONDS,
}
MATERIALIZATION_CONTRACT_SHA256 = sha256_json(MATERIALIZATION_CONTRACT)


ClipProcessor = Callable[[Path, Path, float, float], Mapping[str, Any]]
MediaProbe = Callable[[Path], Mapping[str, Any]]
ErrorT = TypeVar("ErrorT", bound=Exception)


def ffmpeg_materialize_clip(
    source: Path,
    destination: Path,
    start_seconds: float,
    end_seconds: float,
    *,
    ffmpeg: str = "ffmpeg",
) -> Mapping[str, Any]:
    """Accurately re-encode one source range as a zero-origin MP4 clip."""
    duration_seconds = end_seconds - start_seconds
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            _format_seconds(start_seconds),
            "-i",
            str(source),
            "-t",
            _format_seconds(duration_seconds),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            "setpts=PTS-STARTPTS",
            "-af",
            "asetpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExperimentDataError(
            f"ffmpeg failed while materializing {source}: {result.stderr.strip()}"
        )
    return probe_materialized_clip(destination)


def probe_materialized_clip(
    path: Path,
    *,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Probe required streams, duration, and standalone container origin."""
    media = _probe_media(path, ffprobe=ffprobe)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=start_time",
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
            f"ffprobe failed while reading clip start time {path}: "
            f"{result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
        start_time_seconds = float(payload["format"]["start_time"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ExperimentDataError(
            f"ffprobe returned invalid clip start time for {path}"
        ) from exc
    if not math.isfinite(start_time_seconds):
        raise ExperimentDataError(f"Materialized clip start time is non-finite: {path}")
    media["start_time_seconds"] = start_time_seconds
    return media


def validate_clip_media(
    media: Mapping[str, Any],
    requested_duration_seconds: float,
    path: Path,
    *,
    error_type: type[ErrorT] = ExperimentDataError,
) -> None:
    """Validate the shared zero-origin audiovisual clip contract."""
    try:
        duration = float(media["duration_seconds"])
        width = int(media["width"])
        height = int(media["height"])
        audio_stream_count = int(media["audio_stream_count"])
        start_time_seconds = float(media["start_time_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise error_type(f"Materialized clip metadata is invalid: {path}") from exc
    if (
        not math.isfinite(requested_duration_seconds)
        or requested_duration_seconds <= 0
    ):
        raise error_type(f"Requested clip duration is invalid: {path}")
    if width <= 0 or height <= 0 or audio_stream_count <= 0:
        raise error_type(f"Materialized clip lacks video or audio: {path}")
    if (
        not math.isfinite(start_time_seconds)
        or abs(start_time_seconds) > CLIP_START_TIME_TOLERANCE_SECONDS
    ):
        raise error_type(
            f"Materialized clip timeline does not start at zero: {path} "
            f"({start_time_seconds})"
        )
    if (
        not math.isfinite(duration)
        or abs(duration - requested_duration_seconds)
        > CLIP_DURATION_TOLERANCE_SECONDS
    ):
        raise error_type(
            f"Materialized clip duration differs from requested range: {path} "
            f"({duration} vs {requested_duration_seconds})"
        )


def validate_frozen_clip(
    path: Path,
    entry: Mapping[str, Any],
    *,
    media_probe: MediaProbe = probe_materialized_clip,
    error_type: type[ErrorT] = ExperimentDataError,
) -> None:
    """Validate frozen clip bytes, media metadata, and encoding contract."""
    if entry.get("status") != "complete" or not path.is_file():
        raise error_type(f"Materialized clip is incomplete: {path}")
    if (
        sha256_file(path) != entry.get("sha256")
        or path.stat().st_size != entry.get("size_bytes")
    ):
        raise error_type(f"Materialized clip bytes changed: {path}")
    try:
        media = dict(media_probe(path))
    except error_type:
        raise
    except Exception as exc:
        raise error_type(f"Cannot validate materialized clip media: {path}: {exc}") from exc
    if media != entry.get("media"):
        raise error_type(f"Materialized clip media metadata changed: {path}")
    try:
        requested = float(entry["requested_duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise error_type(f"Materialized clip requested duration is invalid: {path}") from exc
    validate_clip_media(media, requested, path, error_type=error_type)


def _probe_media(path: Path, *, ffprobe: str) -> dict[str, Any]:
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
        raise ExperimentDataError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentDataError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(payload, Mapping):
        raise ExperimentDataError(f"ffprobe returned a non-object for {path}")
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
    if not video_streams or not audio_streams:
        raise ExperimentDataError(f"Materialized clip lacks video or audio: {path}")
    video = video_streams[0]
    try:
        duration = float((payload.get("format") or {})["duration"])
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
        "audio_codecs": [
            str(stream.get("codec_name", "unknown")) for stream in audio_streams
        ],
    }


def _format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


__all__ = [
    "CLIP_DURATION_TOLERANCE_SECONDS",
    "CLIP_START_TIME_TOLERANCE_SECONDS",
    "ClipProcessor",
    "MATERIALIZATION_CONTRACT",
    "MATERIALIZATION_CONTRACT_SHA256",
    "MATERIALIZATION_CONTRACT_VERSION",
    "MediaProbe",
    "ffmpeg_materialize_clip",
    "probe_materialized_clip",
    "validate_clip_media",
    "validate_frozen_clip",
]
