"""Transcribe all downloaded SoccerNet matches in batch.

Skips halves that already have a cached transcript.

Usage:
    PYTHONPATH=src:. .venv/bin/python scripts/transcribe_all_matches.py
    PYTHONPATH=src:. .venv/bin/python scripts/transcribe_all_matches.py --model small
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SOCCERNET_DIR = ROOT / "data" / "soccernet"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"


def _safe_name(game: str) -> str:
    return game.replace("/", "_").replace(" ", "_").replace("-", "_")


def _find_games() -> list[str]:
    """Walk data/soccernet/ and return all game paths that have at least one MKV."""
    games = []
    for mkv in sorted(SOCCERNET_DIR.rglob("1_224p.mkv")):
        game_dir = mkv.parent
        rel = game_dir.relative_to(SOCCERNET_DIR)
        games.append(str(rel))
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    games = _find_games()
    print(f"Found {len(games)} games.\n")
    print(f"Loading Whisper model '{args.model}'…")
    import whisper
    model = whisper.load_model(args.model)
    print("Model ready.\n")

    total_halves = 0
    skipped = 0

    for game in games:
        game_dir = SOCCERNET_DIR / game
        safe = _safe_name(game)
        print(f"Game: {Path(game).name}")

        for half in ("1", "2"):
            mkv_path = game_dir / f"{half}_224p.mkv"
            out_path = TRANSCRIPT_DIR / f"{safe}_half{half}.json"

            if not mkv_path.exists():
                continue

            if out_path.exists():
                print(f"  Half {half}: already transcribed — skipping")
                skipped += 1
                continue

            print(f"  Half {half}: transcribing…", flush=True)
            t0 = time.time()
            result = model.transcribe(str(mkv_path), fp16=False, language="en",
                                      verbose=False, condition_on_previous_text=True)
            elapsed = time.time() - t0

            segments = [
                {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
                for s in result["segments"]
            ]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "game": game,
                "half": half,
                "language": result.get("language", "?"),
                "duration_s": round(elapsed, 1),
                "segments": segments,
            }, indent=2, ensure_ascii=False))

            print(f"    Done in {elapsed:.0f}s — {len(segments)} segments → {out_path.name}")
            total_halves += 1

        print()

    print(f"Transcribed {total_halves} halves. Skipped {skipped} already-done.")
    print(f"\nNext step:")
    print(f"  PYTHONPATH=src:. .venv/bin/python scripts/prepare_match_data.py --all")


if __name__ == "__main__":
    main()
