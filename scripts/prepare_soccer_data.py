"""Download SoccerNet games and produce data/soccer_goal_clips.jsonl.

Each row is a 60-second VideoView centred on a goal (positive) or a quiet
stretch of regular play (negative). Each row also carries a ``transcript``
field built by concatenating every SoccerNet caption annotation that falls
inside the clip window. This transcript is the cheap text gate used by the
optimized MMDS pipeline (no LLM, no video decoding).

Usage:
    PYTHONPATH=src:. .venv/bin/python scripts/prepare_soccer_data.py [--max-games N]

Environment:
    SOCCERNET_PASSWORD   OwnCloud password  (default: s0cc3rn3t)
    SOCCERNET_DIR        local download dir (default: data/soccernet)
    SOCCERNET_SPLIT      split to pull from (default: valid)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = ROOT / "data" / "soccernet"
DEFAULT_OUTPUT = ROOT / "data" / "soccer_goal_clips.jsonl"
DEFAULT_CLIPS_DIR = ROOT / "data" / "soccer_clips"
CLIP_HALF_LEN_SEC = 30.0   # seconds on each side of event → 60 s total clip
NEGATIVE_STRIDE_SEC = 120  # one negative clip every 2 min of quiet play
NEGATIVE_BUFFER_SEC = 60   # drop negatives within this many s of any event
HALF_DURATION_SEC = 55 * 60  # conservative upper bound per half

REQUIRED_FILES = ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json", "Labels-caption.json"]


# ---------------------------------------------------------------------------
# SoccerNet download helpers
# ---------------------------------------------------------------------------

def _downloader(data_dir: Path, password: str):
    try:
        from SoccerNet.Downloader import SoccerNetDownloader  # type: ignore[import]
    except ImportError:
        raise SystemExit(
            "SoccerNet package not found.\n"
            "Install it with:  uv pip install SoccerNet --python .venv/bin/python"
        )
    dl = SoccerNetDownloader(LocalDirectory=str(data_dir))
    dl.password = password
    return dl


def _target_games(split: str, max_games: int) -> list[str]:
    """Return the first max_games games from the caption-task game list."""
    from SoccerNet.utils import getListGames  # type: ignore[import]
    return getListGames(split, task="caption")[:max_games]


def download(data_dir: Path, split: str, password: str, games: list[str]) -> None:
    """Download only the specific games we need, skipping already-complete files."""
    dl = _downloader(data_dir, password)

    for i, game in enumerate(games, 1):
        game_dir = data_dir / game
        missing = [f for f in REQUIRED_FILES if not (game_dir / f).exists()]
        if not missing:
            print(f"  [{i}/{len(games)}] {game}  ✓ already complete")
            continue
        print(f"  [{i}/{len(games)}] {game}  — fetching {missing}")
        dl.downloadGame(game=game, files=missing, spl=split, verbose=True)


# ---------------------------------------------------------------------------
# Caption-transcript helpers
# ---------------------------------------------------------------------------

def parse_position_sec(position_ms: str | int) -> float:
    return int(position_ms) / 1000.0


def build_transcript_index(
    caption_path: Path,
) -> dict[str, list[tuple[float, str]]]:
    """Return {half_str -> sorted [(position_sec, description)]} from Labels-caption.json."""
    if not caption_path.exists():
        return {}
    data = json.loads(caption_path.read_text(encoding="utf-8"))
    index: dict[str, list[tuple[float, str]]] = {"1": [], "2": []}
    for ann in data.get("annotations", []):
        desc = ann.get("description", "").strip()
        if not desc:
            continue
        game_time = ann.get("gameTime", "")
        if " - " not in game_time:
            continue
        half_str = game_time.split(" - ")[0]
        if half_str not in index:
            continue
        pos_sec = parse_position_sec(ann.get("position", 0))
        index[half_str].append((pos_sec, desc))
    for key in index:
        index[key].sort()
    return index


def get_clip_transcript(
    caption_index: dict[str, list[tuple[float, str]]],
    half: str,
    start_sec: float,
    end_sec: float,
) -> str:
    """Concatenate every caption description whose timestamp falls in [start_sec, end_sec]."""
    parts = [
        desc
        for pos, desc in caption_index.get(half, [])
        if start_sec <= pos <= end_sec
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Clip extraction (MKV → MP4 via ffmpeg)
# ---------------------------------------------------------------------------

def extract_clip_mp4(
    source_mkv: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
) -> Path:
    """Extract [start_sec, end_sec] from source_mkv into an MP4 at out_path.

    Uses stream-copy (no re-encode) so it's fast.  Skips extraction if the
    output file already exists.  Returns out_path.
    """
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(source_mkv),
        "-t", str(duration),
        "-c", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {out_path.name}:\n{result.stderr.decode()}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Clip building
# ---------------------------------------------------------------------------

def build_clips_for_game(
    game_dir: Path,
    labels_path: Path,
    clip_counter: list[int],
    clips_dir: Path,
) -> list[dict]:
    spotting = json.loads(labels_path.read_text(encoding="utf-8"))
    game_name = spotting.get("UrlLocal", game_dir.name)
    annotations = spotting.get("annotations", [])

    caption_index = build_transcript_index(game_dir / "Labels-caption.json")

    # Group spotting annotations by half
    by_half: dict[str, list[dict]] = {"1": [], "2": []}
    for ann in annotations:
        half_str, _ = ann["gameTime"].split(" - ")
        if half_str in by_half:
            by_half[half_str].append(ann)

    rows: list[dict] = []

    def make_clip(
        half: str,
        mkv_path: Path,
        center_sec: float,
        true_label: bool,
        event_type: str | None,
        video_duration: float,
    ) -> dict:
        start = max(0.0, center_sec - CLIP_HALF_LEN_SEC)
        end = min(video_duration, center_sec + CLIP_HALF_LEN_SEC)
        clip_id = f"clip_{clip_counter[0]:05d}"
        clip_counter[0] += 1
        mp4_path = extract_clip_mp4(mkv_path, start, end, clips_dir / f"{clip_id}.mp4")
        return {
            "clip_id": clip_id,
            "video": {
                "type": "VideoView",
                "source": str(mp4_path.resolve()),
            },
            "transcript": get_clip_transcript(caption_index, half, start, end),
            "true_label": true_label,
            "game": game_name,
            "half": int(half),
            "event_type": event_type,
        }

    for half in ("1", "2"):
        mkv_path = game_dir / f"{half}_224p.mkv"
        if not mkv_path.exists():
            print(f"  Warning: {mkv_path} not found; skipping half {half}")
            continue

        # Use actual video duration so we never generate clips past the file end.
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mkv_path)],
            capture_output=True, text=True,
        )
        actual_duration = float(probe.stdout.strip()) if probe.stdout.strip() else HALF_DURATION_SEC

        events = by_half[half]
        goal_times = {
            round(parse_position_sec(e["position"])) for e in events
            if e.get("label") == "Goal" and e.get("visibility", "visible") == "visible"
        }

        # Positive clips: centred on each visible Goal
        for ann in events:
            if ann.get("label") != "Goal" or ann.get("visibility", "visible") != "visible":
                continue
            t = parse_position_sec(ann["position"])
            rows.append(make_clip(half, mkv_path, t, True, "Goal", actual_duration))

        # Negative clips: regular play — only exclude windows that overlap a goal.
        # Corners, fouls, yellow cards etc. are fine; they are not goals.
        t = float(NEGATIVE_STRIDE_SEC)
        while t + CLIP_HALF_LEN_SEC < actual_duration:
            if not any(abs(t - gt) < NEGATIVE_BUFFER_SEC for gt in goal_times):
                rows.append(make_clip(half, mkv_path, t, False, None, actual_duration))
            t += NEGATIVE_STRIDE_SEC

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-games", type=int, default=2,
                        help="Number of games to download and process (default: 2)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download; use whatever is already on disk")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    password = os.environ.get("SOCCERNET_PASSWORD", "s0cc3rn3t")
    data_dir = Path(os.environ.get("SOCCERNET_DIR", str(DEFAULT_DATA_DIR)))
    split = os.environ.get("SOCCERNET_SPLIT", "valid")
    output = DEFAULT_OUTPUT

    data_dir.mkdir(parents=True, exist_ok=True)

    # Always work from the caption-task game list so every game has both
    # Labels-v2.json (for goal timestamps) and Labels-caption.json (for transcripts).
    target_games = _target_games(split, args.max_games)
    print(f"Target games ({len(target_games)}):")
    for g in target_games:
        print(f"  {g}")

    if not args.no_download:
        print()
        download(data_dir, split, password, target_games)

    # Build clips only for our target games
    game_dirs: list[tuple[Path, Path]] = []
    for game in target_games:
        game_dir = data_dir / game
        labels_path = game_dir / "Labels-v2.json"
        has_video = (game_dir / "1_224p.mkv").exists() or (game_dir / "2_224p.mkv").exists()
        if not labels_path.exists() or not has_video:
            print(f"  Skipping {game} — missing files (run without --no-download)")
            continue
        game_dirs.append((game_dir, labels_path))

    if not game_dirs:
        raise SystemExit("No complete games found. Run without --no-download.")

    n_with_captions = sum(1 for gd, _ in game_dirs if (gd / "Labels-caption.json").exists())
    print(f"\nProcessing {len(game_dirs)} game(s) — "
          f"{n_with_captions}/{len(game_dirs)} have caption transcripts.")

    clips_dir = DEFAULT_CLIPS_DIR
    clips_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting MP4 clips → {clips_dir}")

    all_clips: list[dict] = []
    clip_counter = [0]
    for game_dir, labels_path in game_dirs:
        clips = build_clips_for_game(game_dir, labels_path, clip_counter, clips_dir)
        n_pos = sum(c["true_label"] for c in clips)
        n_neg = len(clips) - n_pos
        has_cap = (game_dir / "Labels-caption.json").exists()
        print(f"  {game_dir.name}: {n_pos} goals, {n_neg} non-goals"
              f"{' [+captions]' if has_cap else ' [no captions]'}")
        all_clips.extend(clips)

    random.seed(args.seed)
    random.shuffle(all_clips)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for clip in all_clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    n_pos = sum(c["true_label"] for c in all_clips)
    n_neg = len(all_clips) - n_pos
    n_no_transcript = sum(1 for c in all_clips if not c.get("transcript"))
    print(f"\nWrote {len(all_clips)} clips → {output}")
    print(f"  Goals (positive):    {n_pos}")
    print(f"  Non-goals (negative): {n_neg}")
    if n_no_transcript:
        print(f"  Warning: {n_no_transcript} clips have empty transcripts")
    print(f"\nNext step:\n  GEMINI_API_KEY=<key> PYTHONPATH=src:. "
          f".venv/bin/python examples/run_soccer_benchmark.py")


if __name__ == "__main__":
    main()
