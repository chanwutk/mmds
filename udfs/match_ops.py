"""UDFs for the full-match goal detection pipeline."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Goal-signal vocabulary
# Single words are matched against tokenised (punctuation-stripped) words.
# Phrases are matched as substrings of the lowercased transcript text.
# ---------------------------------------------------------------------------

_SIGNAL_WORDS = frozenset({
    "goal", "scores", "scored", "scoring",
    "equalizer", "equalised", "equalized", "levelled",
})

_SIGNAL_PHRASES = (
    # Ball entering the net
    "into the net", "into the goal", "back of the net",
    "into the corner", "into the top", "into the bottom",
    "went in off", "went into the", "ricocheting in",
    "tucks it in", "tucks it away", "slots it in", "pokes it in",
    "headed in", "puts it away", "sweeps it home",
    "fires home", "drives it home", "drills it home", "nets it",
    "tap in", "tap-in",
    # Finish language
    "a fine finish", "brilliant finish", "superb finish",
    "great finish", "what a finish", "what a goal", "what a strike",
    # Scoreline references
    "onto the scoresheet", "on the scoresheet", "the score is",
    "converts the",
    "it's 1:", "it's 2:", "it's 3:", "it's 4:", "it's 5:", "it's 6:",
    "1-0", "2-0", "3-0", "4-0", "5-0",
    "1-1", "2-1", "3-1", "4-1",
    "2-2", "3-2", "4-2", "3-3",
    # Exclamatory commentary
    "it's in", "he's done it", "she's done it",
    "he has done", "she has done",
    "makes it",
)

# Phrases that contain signal words but describe non-goal situations.
_FALSE_POSITIVE_PHRASES = (
    "on goal", "behind the goal", "goal difference",
    "goal kick", "goal line", "no goal",
)

CLIP_DIR = Path(__file__).resolve().parents[1] / "data" / "match_clips"
CLIP_DURATION_SEC = 15.0   # seconds before the signal timestamp to capture
CLIP_LOOKAHEAD_SEC = 5.0   # seconds after the signal timestamp
NAIVE_CHUNK_SEC = 30.0     # naive baseline: one clip every N seconds
GATHER_WINDOW_SEC = 60.0   # transcript context window on each side of the candidate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_segments(segments: list[dict]) -> list[dict]:
    hits = []
    for seg in segments:
        text = seg["text"].strip().lower()
        if any(fp in text for fp in _FALSE_POSITIVE_PHRASES):
            continue
        tokens = {w.strip(".,!?;:'\"()-") for w in text.split()}
        matched_word = tokens & _SIGNAL_WORDS
        matched_phrase = next((p for p in _SIGNAL_PHRASES if p in text), None)
        if matched_word or matched_phrase:
            hits.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "signal": sorted(matched_word) if matched_word else [matched_phrase],
            })
    return hits


def _cluster_hits(hits: list[dict], gap_sec: float = 30.0) -> list[dict]:
    """Merge nearby hits into single candidate events."""
    if not hits:
        return []
    clusters = []
    cur = dict(hits[0])
    for h in hits[1:]:
        if h["start"] - cur["end"] <= gap_sec:
            cur["end"] = max(cur["end"], h["end"])
            cur["signal"] = sorted(set(cur["signal"]) | set(h["signal"]))
            cur["text"] = cur["text"] + " | " + h["text"]
        else:
            clusters.append(cur)
            cur = dict(h)
    clusters.append(cur)
    return clusters


# ---------------------------------------------------------------------------
# UDFs
# ---------------------------------------------------------------------------

def find_goal_candidates(row: dict) -> dict:
    """Scan the Whisper transcript and return the row with a `candidates` list.

    Each candidate: {"start": float, "end": float, "text": str, "signal": list[str]}
    """
    transcript_path = Path(row["transcript_cache"])
    if not transcript_path.exists():
        return {**row, "candidates": []}
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    hits = _scan_segments(data.get("segments", []))
    return {**row, "candidates": _cluster_hits(hits)}


def extract_candidate_clip(row: dict) -> dict:
    """Extract a 20s MP4 clip centred on a goal candidate timestamp.

    Window: [signal_time - CLIP_DURATION_SEC, signal_time + CLIP_LOOKAHEAD_SEC]
    The look-back captures the goal itself; commentators react after the fact.
    """
    candidate = row["candidates"]
    signal_time = candidate["start"]
    game_safe = row["game"].replace("/", "_").replace(" ", "_").replace("-", "_")
    clip_name = f"{game_safe}_half{row['half']}_{int(signal_time):05d}s.mp4"

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLIP_DIR / clip_name

    if not out_path.exists():
        start = max(0.0, signal_time - CLIP_DURATION_SEC)
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(row["source"]),
            "-t", str(CLIP_DURATION_SEC + CLIP_LOOKAHEAD_SEC), "-c", "copy", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {clip_name}:\n{result.stderr.decode()}")

    return {
        **row,
        "video": {"type": "VideoView", "source": str(out_path.resolve())},
        "candidate_time_sec": signal_time,
    }


def gather_transcript_context(row: dict) -> str:
    """Return broadcast commentary text from ±GATHER_WINDOW_SEC around the candidate.

    Attaches surrounding transcript text to each clip so Gemini can reason about
    both the video and what the commentator said around that moment.
    """
    candidate = row["candidates"]
    signal_time = candidate["start"]
    transcript_path = Path(row["transcript_cache"])
    if not transcript_path.exists():
        return ""
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    window = [
        s["text"].strip()
        for s in data.get("segments", [])
        if signal_time - GATHER_WINDOW_SEC <= s["start"] <= signal_time + GATHER_WINDOW_SEC
    ]
    return " ".join(window)


def chunk_half(row: dict) -> dict:
    """Divide a match half into fixed-size 30s chunks (naive baseline).

    Uses ffprobe for actual duration; falls back to 45 min if unavailable.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(row["source"])],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 45 * 60.0
    chunks = []
    t = 0.0
    while t + NAIVE_CHUNK_SEC <= duration:
        chunks.append({"start": t, "end": t + NAIVE_CHUNK_SEC, "mid": t + NAIVE_CHUNK_SEC / 2})
        t += NAIVE_CHUNK_SEC
    return {**row, "chunks": chunks}


def extract_chunk_clip(row: dict) -> dict:
    """Extract a 30s chunk clip for the naive baseline."""
    chunk = row["chunks"]
    game_safe = row["game"].replace("/", "_").replace(" ", "_").replace("-", "_")
    clip_name = f"naive_{game_safe}_half{row['half']}_{int(chunk['start']):05d}s.mp4"

    naive_dir = CLIP_DIR / "naive"
    naive_dir.mkdir(parents=True, exist_ok=True)
    out_path = naive_dir / clip_name

    if not out_path.exists():
        cmd = [
            "ffmpeg", "-y", "-ss", str(chunk["start"]), "-i", str(row["source"]),
            "-t", str(NAIVE_CHUNK_SEC), "-c", "copy", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {clip_name}:\n{result.stderr.decode()}")

    return {
        **row,
        "video": {"type": "VideoView", "source": str(out_path.resolve())},
        "candidate_time_sec": chunk["mid"],
    }
