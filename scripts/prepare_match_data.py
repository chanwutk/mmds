"""Build data/matches.jsonl — one row per match half for the full-match pipeline.

Expects transcripts to already exist in data/transcripts/ (run transcribe_match.py first).

Usage:
    PYTHONPATH=src:. .venv/bin/python scripts/prepare_match_data.py \\
        --games "england_epl/2015-2016/2016-02-13 - 20-30 Chelsea 5 - 1 Newcastle Utd"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOCCERNET_DIR = ROOT / "data" / "soccernet"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
OUTPUT = ROOT / "data" / "matches.jsonl"


def _safe_name(game: str) -> str:
    return game.replace("/", "_").replace(" ", "_").replace("-", "_")


def _load_ground_truth(game_dir: Path) -> dict[str, list[float]]:
    labels_path = game_dir / "Labels-v2.json"
    if not labels_path.exists():
        return {"1": [], "2": []}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    result: dict[str, list[float]] = {"1": [], "2": []}
    for ann in data.get("annotations", []):
        if ann.get("label") != "Goal" or ann.get("visibility", "visible") != "visible":
            continue
        half = ann["gameTime"].split(" - ")[0]
        if half in result:
            result[half].append(int(ann["position"]) / 1000.0)
    return result


def _find_all_games() -> list[str]:
    games = []
    for mkv in sorted(SOCCERNET_DIR.rglob("1_224p.mkv")):
        rel = mkv.parent.relative_to(SOCCERNET_DIR)
        games.append(str(rel))
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", nargs="+",
        default=None,
        help="Game paths relative to data/soccernet/ (default: all downloaded games)",
    )
    parser.add_argument("--all", action="store_true",
                        help="Include all downloaded games")
    args = parser.parse_args()

    games = args.games or _find_all_games()

    rows: list[dict] = []
    for game in games:
        game_dir = SOCCERNET_DIR / game
        if not game_dir.exists():
            print(f"  Skipping {game} — directory not found")
            continue

        ground_truth = _load_ground_truth(game_dir)
        safe = _safe_name(game)

        for half in ("1", "2"):
            mkv_path = game_dir / f"{half}_224p.mkv"
            transcript_path = TRANSCRIPT_DIR / f"{safe}_half{half}.json"

            if not mkv_path.exists():
                print(f"  Skipping {game} half {half} — MKV not found")
                continue
            if not transcript_path.exists():
                print(f"  Warning: no transcript for {game} half {half} — "
                      f"run transcribe_match.py first")

            rows.append({
                "game": game,
                "half": half,
                "source": str(mkv_path.resolve()),
                "transcript_cache": str(transcript_path.resolve()),
                "ground_truth_goals": ground_truth.get(half, []),
            })
            gt_count = len(ground_truth.get(half, []))
            print(f"  {Path(game).name}  half {half}  ({gt_count} goals)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(f"\nWrote {len(rows)} rows → {OUTPUT}")
    print(f"\nNext step:")
    print(f"  GEMINI_API_KEY=<key> PYTHONPATH=src:. .venv/bin/python "
          f"scripts/run_match_benchmark.py")


if __name__ == "__main__":
    main()
