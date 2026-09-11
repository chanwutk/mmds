"""Atomic standalone video materialization for the View operator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..model import MMDSExecutionError
from .context import MaterializedClip


CLIP_DURATION_TOLERANCE_SECONDS = 0.25
CLIP_START_TIME_TOLERANCE_SECONDS = 0.05
MATERIALIZATION_CONTRACT = {
    "container": "mp4",
    "video_codec": "libx264",
    "video_crf": 18,
    "video_preset": "veryfast",
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
    "audio_bitrate": "128k",
    "timestamp_policy": "standalone_zero_origin",
}


class FFmpegMaterializer:
    """Create validated zero-origin MP4 clips and promote them atomically."""

    def __init__(self, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def materialize(
        self, source: Path, destination: Path, start_seconds: float, end_seconds: float
    ) -> MaterializedClip:
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(end_seconds)
            or start_seconds < 0
            or end_seconds <= start_seconds
        ):
            raise MMDSExecutionError(
                "View materialization requires finite boundaries satisfying 0 <= start < end."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}"
        )
        duration = end_seconds - start_seconds
        command = [
            self.ffmpeg,
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
            _format_seconds(duration),
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
            str(temporary),
        ]
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise MMDSExecutionError(
                    "FFmpeg failed to materialize View "
                    f"{source}[{start_seconds}, {end_seconds}): {result.stderr.strip()}"
                )
            media = probe_materialized_clip(temporary, ffprobe=self.ffprobe)
            validate_materialized_clip(media, duration, temporary)
            os.replace(temporary, destination)
        except OSError as exc:
            raise MMDSExecutionError(
                f"Cannot materialize View to {destination}: {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return MaterializedClip(
            path=destination,
            size_bytes=destination.stat().st_size,
            sha256=_sha256_file(destination),
            duration_seconds=float(media["duration_seconds"]),
        )


def probe_materialized_clip(
    path: Path, *, ffprobe: str = "ffprobe"
) -> dict[str, Any]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MMDSExecutionError(
            f"ffprobe failed for materialized View {path}: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MMDSExecutionError(
            f"ffprobe returned invalid JSON for materialized View {path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise MMDSExecutionError(f"ffprobe returned invalid metadata for {path}.")
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MMDSExecutionError(f"ffprobe returned no stream list for {path}.")
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
    try:
        metadata = payload["format"]
        duration = float(metadata["duration"])
        start_time = float(metadata.get("start_time", 0.0))
        width = int(video_streams[0]["width"])
        height = int(video_streams[0]["height"])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise MMDSExecutionError(
            f"ffprobe returned incomplete media metadata for {path}."
        ) from exc
    return {
        "duration_seconds": duration,
        "start_time_seconds": start_time,
        "width": width,
        "height": height,
        "audio_stream_count": len(audio_streams),
    }


def validate_materialized_clip(
    media: Mapping[str, Any], requested_duration_seconds: float, path: Path
) -> None:
    try:
        duration = float(media["duration_seconds"])
        start_time = float(media["start_time_seconds"])
        width = int(media["width"])
        height = int(media["height"])
        audio_stream_count = int(media["audio_stream_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MMDSExecutionError(
            f"Materialized View metadata is invalid: {path}"
        ) from exc
    if (
        not math.isfinite(duration)
        or abs(duration - requested_duration_seconds)
        > CLIP_DURATION_TOLERANCE_SECONDS
    ):
        raise MMDSExecutionError(
            "Materialized View duration differs from its requested interval: "
            f"{path} ({duration} vs {requested_duration_seconds})."
        )
    if not math.isfinite(start_time) or abs(start_time) > CLIP_START_TIME_TOLERANCE_SECONDS:
        raise MMDSExecutionError(
            f"Materialized View does not start at zero: {path} ({start_time})."
        )
    if width <= 0 or height <= 0 or audio_stream_count <= 0:
        raise MMDSExecutionError(
            f"Materialized View must contain video and audio streams: {path}."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


__all__ = [
    "CLIP_DURATION_TOLERANCE_SECONDS",
    "CLIP_START_TIME_TOLERANCE_SECONDS",
    "FFmpegMaterializer",
    "MATERIALIZATION_CONTRACT",
    "probe_materialized_clip",
    "validate_materialized_clip",
]
