"""WildLife Documentary (WLD) films: labels, subtitles, and prepared videos.

WLD (Chen et al., "Discover and Learn New Objects from Documentaries",
CVPR 2017; github.com/hellock/WLD) has 15 narrated documentaries from
YouTube. Every animal in every frame has a box, and each film comes with
YouTube's subtitles, corrected by hand. Both camera-ready experiments read
their ground truth from here:

- presence: an animal is present in a film when it is visible for at least
  ``MIN_PRESENT_SECONDS`` in total, absent when it never appears, and
  "brief" (left out of scoring) in between;
- localization: the labeled intervals are the frames where the animal is
  visible, merged across gaps up to ``MERGE_GAP_SECONDS``; merged stretches
  shorter than ``MIN_PRESENT_SECONDS`` are ignored time.

Annotation files list ``categories`` by name and give each box as
``[x1, y1, x2, y2, class_id]``, where ``class_id`` counts from 1.

``download`` fetches each film still on YouTube at its annotation's
resolution and records what it got in ``downloads.json``. ``prepare``
re-encodes every downloaded film to 360p, keeping every frame, and accepts a
film only if its frame count is within ``MAX_FRAME_DIFF`` of the
annotation's, so frame ``i`` of the video is frame ``i`` of the labels. Run
from the repo root, after cloning github.com/hellock/WLD's annotations,
subtitles and filelist.txt into ``data/wld/``::

    PYTHONPATH=src:. python -m scripts.query_types.wld download
    PYTHONPATH=src:. python -m scripts.query_types.wld prepare

Both experiments write under ``OUT_ROOT``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.query_types.common import (
    Interval,
    encode_video,
    merge_intervals,
    probe_video,
    read_json,
    write_json_atomic,
)

WLD_ROOT = Path("data/wld")
# Results of both experiments: yoloe/{variant}, presence/run{r}/{plan},
# localization/run{r}/{stage}, and scratch space.
OUT_ROOT = Path("data/query_types/wld")
VIDEO_HEIGHT = 360
# yt-dlp has to keep up with YouTube, so the latest release is used and its
# version is recorded with every download.
YT_DLP = ("uvx", "-q", "yt-dlp@latest")
MIN_PRESENT_SECONDS = 1.0
MERGE_GAP_SECONDS = 1.0

# Each query word and the WLD categories it covers. These are the classes that
# appear in at least 3 of the 15 films; "bear" covers every kind of bear.
QUERY_CLASSES: dict[str, tuple[str, ...]] = {
    "bear": ("bear", "polar bear", "brown bear", "black bear"),
    "elephant": ("elephant",),
    "dog": ("dog",),
    "deer": ("deer",),
    "tiger": ("tiger",),
    "leopard": ("leopard",),
    "monkey": ("monkey",),
    "lion": ("lion",),
    "boar": ("boar",),
}

# Names YOLOE is prompted with together, so an animal on screen competes for
# its best-matching name instead of being forced onto the one queried name:
# COCO's ten animal classes plus the query animals (not tuned on WLD).
COCO_ANIMALS = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
DETECTOR_VOCABULARY: tuple[str, ...] = tuple(dict.fromkeys(COCO_ANIMALS + tuple(QUERY_CLASSES)))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilmLabels:
    film_id: str
    frame_count: int
    categories: tuple[str, ...]
    # The 1-based class ids boxed in each frame.
    frame_classes: tuple[frozenset[int], ...]


def parse_labels(annotation: dict[str, Any]) -> FilmLabels:
    film_id = str(annotation["vid"])
    categories = tuple(str(name) for name in annotation["categories"])
    frames = annotation["bboxes"]
    if int(annotation["frame_num"]) != len(frames):
        raise ValueError(f"{film_id}: frame_num is {annotation['frame_num']} but {len(frames)} frames are boxed.")
    frame_classes = []
    for index, boxes in enumerate(frames):
        ids = set()
        for box in boxes:
            if len(box) != 5:
                raise ValueError(f"{film_id} frame {index}: expected [x1, y1, x2, y2, class_id], got {box!r}.")
            class_id = box[4]
            if isinstance(class_id, bool) or not isinstance(class_id, int) or not 1 <= class_id <= len(categories):
                raise ValueError(
                    f"{film_id} frame {index}: class id {class_id!r} is outside 1..{len(categories)}."
                )
            ids.add(class_id)
        frame_classes.append(frozenset(ids))
    return FilmLabels(film_id, len(frames), categories, tuple(frame_classes))


# Corrections to WLD's labels, applied before any truth is computed. Each was
# checked by eye after a model disagreed with the labels, and each fails loudly
# if the annotation file no longer looks the way it did, instead of being
# silently left unfixed.
#
# Categories whose boxes in a film are wrong, removed. Each must be one of the
# film's categories.
# aN3jEIAXvbk "bear": its only boxes (3.1-4.2 s and 7.7-10.1 s) are in the
# opening landscape shots, with no bear on screen (checked by eye, 2026-10-24).
LABEL_REMOVALS: dict[str, tuple[str, ...]] = {
    "aN3jEIAXvbk": ("bear",),
}

# Animals on screen that WLD did not label, added as (category, first frame,
# last frame). None of those frames may already have the category.
# UsnHCvC92mM "leopard": a leopard walks along the upper platform throughout
# the cage shot of frames 8597-8705 (343.88-348.24 s, cut to cut), which WLD
# labels only as "tiger" (checked by eye, 2026-10-25).
LABEL_ADDITIONS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "UsnHCvC92mM": (("leopard", 8597, 8705),),
}


def fix_labels(labels: FilmLabels) -> FilmLabels:
    """Apply this film's ``LABEL_REMOVALS`` and ``LABEL_ADDITIONS``."""
    film = labels.film_id
    categories = list(labels.categories)
    frame_classes = list(labels.frame_classes)
    for name in LABEL_REMOVALS.get(film, ()):
        if name not in categories:
            raise ValueError(f"{film}: label removal names {name!r}, which is not one of its categories.")
        class_id = categories.index(name) + 1
        frame_classes = [classes - {class_id} for classes in frame_classes]
    for name, first, last in LABEL_ADDITIONS.get(film, ()):
        if not 0 <= first <= last < labels.frame_count:
            raise ValueError(f"{film}: label addition {name!r} at frames {first}-{last} is outside 0-{labels.frame_count - 1}.")
        if name not in categories:
            categories.append(name)
        class_id = categories.index(name) + 1
        if any(class_id in frame_classes[f] for f in range(first, last + 1)):
            raise ValueError(f"{film}: {name!r} is already labeled in frames {first}-{last}; drop this label addition.")
        for f in range(first, last + 1):
            frame_classes[f] = frame_classes[f] | {class_id}
    return FilmLabels(film, labels.frame_count, tuple(categories), tuple(frame_classes))


def load_labels(film_id: str, root: Path = WLD_ROOT) -> FilmLabels:
    return fix_labels(parse_labels(read_json(root / "annotations" / f"{film_id}.json")))


def class_frames(labels: FilmLabels, names: Iterable[str]) -> list[int]:
    """Frames that box any of the named categories."""
    wanted = set(names)
    ids = {index + 1 for index, name in enumerate(labels.categories) if name in wanted}
    return [index for index, classes in enumerate(labels.frame_classes) if classes & ids]


def frames_to_intervals(frames: Iterable[int], fps: float, gap: float = MERGE_GAP_SECONDS) -> list[Interval]:
    """Frame ``f`` covers ``[f / fps, (f + 1) / fps)``; merge across gaps up to ``gap``."""
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return merge_intervals(((f / fps, (f + 1) / fps) for f in frames), gap)


def presence(frames: Sequence[int], fps: float, min_seconds: float = MIN_PRESENT_SECONDS) -> str:
    """``"present"``, ``"brief"`` (visible, but under ``min_seconds`` in total) or ``"absent"``."""
    visible = len(frames) / fps
    if visible >= min_seconds:
        return "present"
    return "brief" if frames else "absent"


def localization_truth(
    frames: Iterable[int],
    fps: float,
    gap: float = MERGE_GAP_SECONDS,
    min_seconds: float = MIN_PRESENT_SECONDS,
) -> tuple[list[Interval], list[Interval]]:
    """Return (labeled, ignored) intervals: merged stretches at least ``min_seconds`` long, and the rest."""
    labeled: list[Interval] = []
    ignored: list[Interval] = []
    for start, end in frames_to_intervals(frames, fps, gap):
        (labeled if end - start >= min_seconds else ignored).append((start, end))
    return labeled, ignored


def film_truth(labels: FilmLabels, fps: float) -> dict[str, dict[str, Any]]:
    """Presence and localization truth for every query word in one film."""
    truth: dict[str, dict[str, Any]] = {}
    for query, names in QUERY_CLASSES.items():
        frames = class_frames(labels, names)
        labeled, ignored = localization_truth(frames, fps)
        truth[query] = {
            "presence": presence(frames, fps),
            "presence_any_frame": bool(frames),
            "visible_seconds": len(frames) / fps,
            "first_seconds": frames[0] / fps if frames else None,
            "labeled": labeled,
            "ignored": ignored,
        }
    return truth


# ---------------------------------------------------------------------------
# (film, animal) pairs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One question: is (or when is) ``animal`` on screen in ``film_id``?"""

    film_id: str
    animal: str
    path: str
    duration: float
    fps: float
    frames: int
    truth: str  # "present", "brief" or "absent"
    truth_any_frame: bool
    first_seconds: float | None

    @property
    def pair_id(self) -> str:
        return f"{self.film_id}|{self.animal}"

    @property
    def file_stem(self) -> str:
        return f"{self.film_id}__{self.animal.replace(' ', '_')}"

    @property
    def youtube_url(self) -> str:
        """WLD film ids are YouTube video ids; Gemini reads the film from here."""
        return f"https://www.youtube.com/watch?v={self.film_id}"

    def video(self, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        """The film (or ``[start, end)`` of it) as a Gemini video prompt value.

        Gemini fetches public YouTube videos itself, so nothing is uploaded and
        the project's file storage quota is never used. With a window, Gemini
        sees only that part and reports times from the window's start.
        """
        value: dict[str, Any] = {"type": "video", "uri": self.youtube_url}
        if (start is None) != (end is None):
            raise ValueError("Give both start and end, or neither.")
        if start is not None:
            value.update(start=float(start), end=float(end))
        return value


def load_pairs(root: Path = WLD_ROOT) -> list[Pair]:
    """One pair per prepared film and query animal, in a fixed order."""
    truth = read_json(root / "truth.json")
    pairs = []
    for film in read_json(root / "films.json"):
        if film["status"] != "ok":
            continue
        for animal in QUERY_CLASSES:
            label = truth[film["film_id"]][animal]
            pairs.append(
                Pair(
                    film_id=film["film_id"],
                    animal=animal,
                    path=film["path"],
                    duration=float(film["duration"]),
                    fps=float(film["fps"]),
                    frames=int(film["frames"]),
                    truth=label["presence"],
                    truth_any_frame=bool(label["presence_any_frame"]),
                    first_seconds=label["first_seconds"],
                )
            )
    if not pairs:
        raise SystemExit(f"No prepared films in {root / 'films.json'}; run 'wld prepare' first.")
    return pairs


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

_TIME = r"(\d+):(\d{2}):(\d{2})[,.](\d{3})"
_TIMING = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}\s*$")


def parse_srt(text: str) -> list[dict[str, Any]]:
    """Parse SubRip text into ``{"start", "end", "text"}`` cues in seconds.

    Handles a leading byte-order mark, CRLF line ends, missing cue numbers and
    multi-line cue text (joined with spaces). Cues with no text are skipped.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[dict[str, Any]] = []
    for number, block in enumerate(re.split(r"\n\s*\n", text.strip()), start=1):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timing_at = 1 if len(lines) > 1 and lines[0].strip().isdigit() else 0
        match = _TIMING.match(lines[timing_at])
        if match is None:
            raise ValueError(f"Subtitle block {number}: no timing line in {block!r}.")
        values = [int(part) for part in match.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        if end < start:
            raise ValueError(f"Subtitle block {number}: ends at {end} before it starts at {start}.")
        cue_text = " ".join(line.strip() for line in lines[timing_at + 1:])
        if cue_text:
            cues.append({"start": start, "end": end, "text": cue_text})
    return cues


# Typos in the released subtitle files, fixed before parsing. Each must match
# exactly once, so a changed file fails loudly instead of being half-fixed.
# aN3jEIAXvbk cue 248: the previous cue ends at 00:18:54.599, so "018:54.600"
# is 00:18:54.600 with a character lost in the manual correction.
SUBTITLE_FIXES: dict[str, tuple[tuple[str, str], ...]] = {
    "aN3jEIAXvbk": (("\n018:54.600 -->", "\n00:18:54.600 -->"),),
}


def fix_subtitles(film_id: str, text: str) -> str:
    for wrong, right in SUBTITLE_FIXES.get(film_id, ()):
        count = text.count(wrong)
        if count != 1:
            raise ValueError(f"{film_id}: expected the subtitle typo {wrong!r} once, found it {count} times.")
        text = text.replace(wrong, right)
    return text


def load_subtitles(film_id: str, root: Path = WLD_ROOT) -> list[dict[str, Any]]:
    text = (root / "subtitles" / f"{film_id}.srt").read_text(encoding="utf-8")
    return parse_srt(fix_subtitles(film_id, text))


# ---------------------------------------------------------------------------
# Preparing videos
# ---------------------------------------------------------------------------


# Today's YouTube copies differ from the 2017 annotations by 1-2 frames
# (at most 0.08 s at 25 fps), which is far below the 1 s scoring rules.
MAX_FRAME_DIFF = 2


def check_film(film_id: str, probe: dict[str, Any], label_frames: int) -> dict[str, Any]:
    """Accept a prepared film if its frame count is within ``MAX_FRAME_DIFF`` of the labels."""
    entry = {"film_id": film_id, **probe, "label_frames": label_frames, "frame_diff": probe["frames"] - label_frames}
    if abs(entry["frame_diff"]) <= MAX_FRAME_DIFF:
        entry["status"] = "ok"
    else:
        entry["status"] = "frame_mismatch"
        entry["reason"] = f"video has {probe['frames']} frames, labels have {label_frames}"
    return entry


def film_ids(root: Path = WLD_ROOT) -> list[str]:
    return [line.strip() for line in (root / "filelist.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def download_command(film_id: str, height: int, target_dir: Path, yt_dlp: Sequence[str] = YT_DLP) -> list[str]:
    """yt-dlp command for one film: H.264 video no taller than ``height`` (the
    annotation's resolution) plus English audio, merged into ``{id}.mp4``.
    It prints the format ids it chose."""
    formats = f"bv*[height<={height}][vcodec^=avc1]+ba[language^=en]/bv*[height<={height}]+ba/b[height<={height}]"
    return [
        *yt_dlp, "--no-warnings", "--no-playlist", "--no-progress",
        "-f", formats, "--merge-output-format", "mp4",
        "--print", "after_move:%(format_id)s",
        "-o", str(target_dir / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={film_id}",
    ]


def download(
    root: Path = WLD_ROOT,
    yt_dlp: Sequence[str] = YT_DLP,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, Any]]:
    """Download every film still on YouTube; write what happened to downloads.json.

    Films already downloaded are kept. A film YouTube no longer has is
    reported as unavailable, not an error: 10 of the 15 are gone.
    """
    target_dir = root / "videos"
    target_dir.mkdir(parents=True, exist_ok=True)
    version = run([*yt_dlp, "--version"], capture_output=True, text=True).stdout.strip()
    records = []
    for film_id in film_ids(root):
        target = target_dir / f"{film_id}.mp4"
        if target.exists():
            records.append({"film_id": film_id, "status": "present"})
            continue
        height = int(read_json(root / "annotations" / f"{film_id}.json")["height"])
        print(f"  downloading {film_id} (up to {height}p) ...", flush=True)
        result = run(download_command(film_id, height, target_dir, yt_dlp), capture_output=True, text=True)
        if result.returncode == 0 and target.exists():
            formats = result.stdout.strip().splitlines()
            records.append({"film_id": film_id, "status": "downloaded", "formats": formats[-1] if formats else None})
        else:
            errors = result.stderr.strip().splitlines()
            records.append({"film_id": film_id, "status": "unavailable", "reason": errors[-1] if errors else None})
        print(f"  {film_id}: {records[-1]['status']}", flush=True)
    write_json_atomic(root / "downloads.json", {"yt_dlp_version": version, "films": records})
    return records


def prepare(root: Path = WLD_ROOT, height: int = VIDEO_HEIGHT) -> list[dict[str, Any]]:
    """Re-encode downloaded films, check frame counts, and write films.json and truth.json."""
    out_dir = root / f"videos_{height}p"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads_path = root / "downloads.json"
    downloads = {}
    if downloads_path.exists():
        record = read_json(downloads_path)
        downloads = {film["film_id"]: {**film, "yt_dlp_version": record["yt_dlp_version"]} for film in record["films"]}
    films: list[dict[str, Any]] = []
    truth: dict[str, Any] = {}
    for film_id in film_ids(root):
        source = root / "videos" / f"{film_id}.mp4"
        if not source.exists():
            films.append({"film_id": film_id, "status": "missing", "reason": f"{source} not downloaded"})
            continue
        labels = load_labels(film_id, root)
        target = out_dir / f"{film_id}.mp4"
        encode_seconds = None
        if not target.exists():
            print(f"  encoding {film_id} to {height}p ...", flush=True)
            encode_seconds = encode_video(source, target, height=height)
        entry = check_film(film_id, probe_video(target), labels.frame_count)
        entry.update({
            "path": str(target),
            "source_frames": probe_video(source)["frames"],
            "encode_seconds": encode_seconds,
            "download": downloads.get(film_id),
        })
        films.append(entry)
        if entry["status"] == "ok":
            truth[film_id] = film_truth(labels, entry["fps"])
        print(f"  {film_id}: {entry['status']} ({entry['frames']} frames at {entry['fps']:.3f} fps)", flush=True)
    write_json_atomic(root / "films.json", films)
    write_json_atomic(root / "truth.json", truth)
    return films


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=WLD_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download the films still on YouTube and write downloads.json")
    commands.add_parser("prepare", help="re-encode downloaded films and write films.json and truth.json")
    args = parser.parse_args(argv)
    if args.command == "download":
        records = download(args.root)
        got = sum(1 for record in records if record["status"] != "unavailable")
        print(f"{got} of {len(records)} films on disk; see {args.root / 'downloads.json'}")
    else:
        films = prepare(args.root)
        ok = sum(1 for film in films if film["status"] == "ok")
        print(f"{ok} of {len(films)} films ready; see {args.root / 'films.json'}")


if __name__ == "__main__":
    main()
