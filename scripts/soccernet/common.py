"""Shared, dependency-light helpers for the SoccerNet experiment pipeline."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json as _shared_atomic_write_json,
    load_secret as _shared_load_secret,
)


SCHEMA_VERSION = 1
DEFAULT_ROOT = Path("data/soccernet")
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 25
DEFAULT_RESOLUTIONS = ("224p", "720p")
LABEL_FILENAME = "Labels-v2.json"
VIDEO_FILENAMES = {
    "224p": ("1_224p.mkv", "2_224p.mkv"),
    "720p": ("1_720p.mkv", "2_720p.mkv"),
}
_GAME_TIME_RE = re.compile(r"^(?P<half>[12])\s*-\s*\d{1,3}:\d{2}$")
class SoccerNetDataError(ExperimentDataError):
    """Raised when dataset state is unsafe, invalid, or incomplete."""


@dataclass(frozen=True)
class LabelSummary:
    annotation_count: int
    goal_count: int
    shown_goal_count: int
    not_shown_goal_count: int


@dataclass(frozen=True)
class MediaProbe:
    duration_seconds: float
    width: int
    height: int
    video_codec: str
    audio_stream_count: int
    audio_codecs: tuple[str, ...]


def game_directory(root: Path, game_id: str) -> Path:
    """Resolve a SoccerNet game id below *root* without allowing traversal."""
    relative = Path(game_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise SoccerNetDataError(f"Unsafe SoccerNet game id: {game_id!r}")
    return root.joinpath(relative)


def select_first_games(games: Sequence[str], limit: int = DEFAULT_LIMIT) -> list[str]:
    """Return the first *limit* unique games in official registry order."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(games):
        raise ValueError(f"limit {limit} exceeds the {len(games)} available games")
    selected = list(games[:limit])
    if len(set(selected)) != len(selected):
        raise SoccerNetDataError("Official game registry contains duplicate game ids")
    return selected


def load_official_games(split: str = DEFAULT_SPLIT) -> list[str]:
    """Load game ids lazily from the installed official SoccerNet package."""
    try:
        from SoccerNet.utils import getListGames
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SoccerNetDataError(
            "SoccerNet is not installed. Install scripts/soccernet/requirements.txt."
        ) from exc
    games = list(getListGames(split, task="spotting"))
    if not games:
        raise SoccerNetDataError(f"SoccerNet returned no games for split {split!r}")
    return games


def load_secret(key: str, env_file: Path | None = Path(".env")) -> str:
    return _shared_load_secret(key, env_file, error_type=SoccerNetDataError)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _shared_atomic_write_json(path, payload)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SoccerNetDataError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SoccerNetDataError(f"Expected a JSON object in {path}")
    return payload


def validate_labels(path: Path) -> LabelSummary:
    payload = load_json_object(path)
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise SoccerNetDataError(f"{path} has no annotations list")

    goal_count = shown_goal_count = not_shown_goal_count = 0
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise SoccerNetDataError(f"{path}: annotation {index} is not an object")
        label = annotation.get("label")
        game_time = annotation.get("gameTime")
        position = annotation.get("position")
        if not isinstance(label, str) or not label.strip():
            raise SoccerNetDataError(f"{path}: annotation {index} has no label")
        if not isinstance(game_time, str) or not _GAME_TIME_RE.match(game_time):
            raise SoccerNetDataError(
                f"{path}: annotation {index} has invalid gameTime {game_time!r}"
            )
        try:
            position_ms = int(position)
        except (TypeError, ValueError) as exc:
            raise SoccerNetDataError(
                f"{path}: annotation {index} has invalid position {position!r}"
            ) from exc
        if position_ms < 0:
            raise SoccerNetDataError(f"{path}: annotation {index} has negative position")

        if label.casefold() == "goal":
            goal_count += 1
            if annotation.get("visibility") == "not shown":
                not_shown_goal_count += 1
            else:
                shown_goal_count += 1

    return LabelSummary(
        annotation_count=len(annotations),
        goal_count=goal_count,
        shown_goal_count=shown_goal_count,
        not_shown_goal_count=not_shown_goal_count,
    )


def _parse_ffprobe(payload: Mapping[str, Any], path: Path) -> MediaProbe:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise SoccerNetDataError(f"ffprobe returned no streams for {path}")
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise SoccerNetDataError(f"No video stream found in {path}")
    video = video_streams[0]
    try:
        width = int(video["width"])
        height = int(video["height"])
        duration = float(payload.get("format", {}).get("duration", video.get("duration")))
    except (KeyError, TypeError, ValueError) as exc:
        raise SoccerNetDataError(f"Incomplete ffprobe metadata for {path}") from exc
    if width <= 0 or height <= 0 or duration <= 0:
        raise SoccerNetDataError(f"Invalid dimensions or duration for {path}")
    return MediaProbe(
        duration_seconds=duration,
        width=width,
        height=height,
        video_codec=str(video.get("codec_name", "unknown")),
        audio_stream_count=len(audio_streams),
        audio_codecs=tuple(str(stream.get("codec_name", "unknown")) for stream in audio_streams),
    )


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> MediaProbe:
    if not path.is_file() or path.stat().st_size == 0:
        raise SoccerNetDataError(f"Missing or empty media file: {path}")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SoccerNetDataError(f"ffprobe failed for {path}: {exc}") from exc
    if result.returncode != 0:
        error = result.stderr.strip() or f"exit code {result.returncode}"
        raise SoccerNetDataError(f"ffprobe rejected {path}: {error}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SoccerNetDataError(f"ffprobe returned invalid JSON for {path}") from exc
    return _parse_ffprobe(payload, path)


def validate_resolution(probe: MediaProbe, resolution: str, path: Path) -> None:
    expected_height = int(resolution.removesuffix("p"))
    if probe.height != expected_height:
        raise SoccerNetDataError(
            f"{path} is {probe.width}x{probe.height}; expected {resolution}"
        )


def ensure_free_space(root: Path, minimum_free_gib: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    required_bytes = int(minimum_free_gib * 1024**3)
    if free_bytes < required_bytes:
        raise SoccerNetDataError(
            f"Only {free_bytes / 1024**3:.1f} GiB free under {root}; "
            f"the configured reserve is {minimum_free_gib:.1f} GiB"
        )


def serialize_probe(probe: MediaProbe, path: Path, root: Path) -> dict[str, Any]:
    payload = asdict(probe)
    payload["audio_codecs"] = list(probe.audio_codecs)
    payload["path"] = path.relative_to(root).as_posix()
    payload["size_bytes"] = path.stat().st_size
    return payload


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
