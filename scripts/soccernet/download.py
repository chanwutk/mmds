"""Plan, download, validate, and inventory a bounded SoccerNet sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, Sequence

from .common import (
    DEFAULT_LIMIT,
    DEFAULT_RESOLUTIONS,
    DEFAULT_ROOT,
    DEFAULT_SPLIT,
    LABEL_FILENAME,
    SCHEMA_VERSION,
    VIDEO_FILENAMES,
    SoccerNetDataError,
    atomic_write_json,
    ensure_free_space,
    game_directory,
    load_json_object,
    load_official_games,
    load_secret,
    probe_media,
    select_first_games,
    serialize_probe,
    validate_labels,
    validate_resolution,
)


SELECTION_FILENAME = "selection.json"
MANIFEST_FILENAME = "manifest.json"


class DownloadBackend(Protocol):
    def download_file(
        self,
        *,
        game_id: str,
        filename: str,
        split: str,
        destination_root: Path,
        password: str,
        verbose: bool,
    ) -> None: ...


class OfficialSoccerNetBackend:
    """Small adapter around the official SoccerNet downloader."""

    def download_file(
        self,
        *,
        game_id: str,
        filename: str,
        split: str,
        destination_root: Path,
        password: str,
        verbose: bool,
    ) -> None:
        try:
            from SoccerNet.Downloader import SoccerNetDownloader
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SoccerNetDataError(
                "SoccerNet is not installed. Install scripts/soccernet/requirements.txt."
            ) from exc
        downloader = SoccerNetDownloader(LocalDirectory=str(destination_root))
        downloader.password = password
        downloader.downloadGame(
            game=game_id,
            files=[filename],
            spl=split,
            verbose=verbose,
        )


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def make_selection(
    *,
    games: Sequence[str],
    split: str,
    limit: int,
    resolutions: Sequence[str],
) -> dict[str, Any]:
    unknown = sorted(set(resolutions) - set(VIDEO_FILENAMES))
    if unknown:
        raise ValueError(f"Unsupported resolutions: {', '.join(unknown)}")
    normalized_resolutions = list(dict.fromkeys(resolutions))
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "SoccerNet Action Spotting",
        "task": "spotting",
        "split": split,
        "strategy": "first_in_official_registry",
        "limit": limit,
        "resolutions": normalized_resolutions,
        "label_filename": LABEL_FILENAME,
        "soccernet_version": _package_version("SoccerNet"),
        "games": select_first_games(games, limit),
    }


def _selection_signature(selection: dict[str, Any]) -> tuple[Any, ...]:
    return (
        selection.get("schema_version"),
        selection.get("split"),
        selection.get("strategy"),
        selection.get("limit"),
        tuple(selection.get("resolutions", [])),
        tuple(selection.get("games", [])),
    )


def freeze_selection(root: Path, requested: dict[str, Any]) -> dict[str, Any]:
    """Create an immutable sample definition, or verify the existing one."""
    path = root / SELECTION_FILENAME
    if path.exists():
        existing = load_json_object(path)
        if _selection_signature(existing) != _selection_signature(requested):
            raise SoccerNetDataError(
                f"{path} defines a different sample. Use another --root or remove it explicitly."
            )
        return existing
    atomic_write_json(path, requested)
    return requested


def expected_files(selection: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """Return ``(game_id, filename, resolution)`` in deterministic order."""
    result: list[tuple[str, str, str | None]] = []
    for game_id in selection["games"]:
        result.append((game_id, LABEL_FILENAME, None))
        for resolution in selection["resolutions"]:
            for filename in VIDEO_FILENAMES[resolution]:
                result.append((game_id, filename, resolution))
    return result


def _validate_file(path: Path, resolution: str | None) -> dict[str, Any]:
    if resolution is None:
        return {"status": "valid", **vars(validate_labels(path))}
    probe = probe_media(path)
    validate_resolution(probe, resolution, path)
    return {"status": "valid", **vars(probe)}


def _staged_path(root: Path, game_id: str, filename: str) -> Path:
    return game_directory(root / ".partial", game_id) / filename


def _promote_valid_staged_file(
    *, root: Path, game_id: str, filename: str, resolution: str | None
) -> bool:
    staged = _staged_path(root, game_id, filename)
    if not staged.exists():
        return False
    try:
        _validate_file(staged, resolution)
    except SoccerNetDataError:
        # This path is owned exclusively by this downloader and is safe to retry.
        staged.unlink()
        return False
    destination = game_directory(root, game_id) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)
    return True


def _download_one(
    *,
    root: Path,
    game_id: str,
    filename: str,
    resolution: str | None,
    split: str,
    password: str,
    backend: DownloadBackend,
    verbose: bool,
) -> str:
    destination = game_directory(root, game_id) / filename
    if destination.exists():
        try:
            _validate_file(destination, resolution)
        except SoccerNetDataError as exc:
            raise SoccerNetDataError(
                f"Existing destination is invalid and was not replaced: {exc}"
            ) from exc
        return "reused"

    if _promote_valid_staged_file(
        root=root,
        game_id=game_id,
        filename=filename,
        resolution=resolution,
    ):
        return "recovered"

    partial_root = root / ".partial"
    backend.download_file(
        game_id=game_id,
        filename=filename,
        split=split,
        destination_root=partial_root,
        password=password,
        verbose=verbose,
    )
    staged = _staged_path(root, game_id, filename)
    if not staged.exists():
        raise SoccerNetDataError(
            f"SoccerNet downloader did not produce {filename} for {game_id}"
        )
    _validate_file(staged, resolution)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)
    return "downloaded"


def build_manifest(root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    total_bytes = {resolution: 0 for resolution in selection["resolutions"]}
    complete_games = 0

    for game_id in selection["games"]:
        directory = game_directory(root, game_id)
        label_path = directory / LABEL_FILENAME
        game_payload: dict[str, Any] = {
            "game_id": game_id,
            "split": selection["split"],
            "labels": {"path": label_path.relative_to(root).as_posix()},
            "videos": {},
        }
        complete = True
        try:
            game_payload["labels"].update(vars(validate_labels(label_path)))
            game_payload["labels"]["status"] = "valid"
        except SoccerNetDataError as exc:
            complete = False
            game_payload["labels"].update(status="invalid", error=str(exc))

        for resolution in selection["resolutions"]:
            resolution_payload: dict[str, Any] = {}
            for half, filename in enumerate(VIDEO_FILENAMES[resolution], start=1):
                path = directory / filename
                try:
                    probe = probe_media(path)
                    validate_resolution(probe, resolution, path)
                    serialized = serialize_probe(probe, path, root)
                    serialized["status"] = "valid"
                    resolution_payload[str(half)] = serialized
                    total_bytes[resolution] += serialized["size_bytes"]
                except SoccerNetDataError as exc:
                    complete = False
                    resolution_payload[str(half)] = {
                        "path": path.relative_to(root).as_posix(),
                        "status": "invalid",
                        "error": str(exc),
                    }
            game_payload["videos"][resolution] = resolution_payload

        game_payload["status"] = "complete" if complete else "incomplete"
        complete_games += int(complete)
        games.append(game_payload)

    comparison: dict[str, Any] = {}
    if {"224p", "720p"}.issubset(total_bytes):
        low_bytes = total_bytes["224p"]
        high_bytes = total_bytes["720p"]
        paired_halves = [
            (game["videos"]["224p"][half], game["videos"]["720p"][half])
            for game in games
            for half in ("1", "2")
            if game["videos"]["224p"][half].get("status") == "valid"
            and game["videos"]["720p"][half].get("status") == "valid"
        ]
        comparison = {
            "paired_half_count": len(paired_halves),
            "720p_to_224p_size_ratio": high_bytes / low_bytes if low_bytes else None,
            "maximum_duration_difference_seconds": max(
                (
                    abs(high["duration_seconds"] - low["duration_seconds"])
                    for low, high in paired_halves
                ),
                default=None,
            ),
            "audio_stream_count_mismatch_count": sum(
                low["audio_stream_count"] != high["audio_stream_count"]
                for low, high in paired_halves
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "selection": selection,
        "summary": {
            "game_count": len(games),
            "complete_game_count": complete_games,
            "total_bytes_by_resolution": total_bytes,
            "resolution_comparison": comparison,
        },
        "games": games,
    }


def download_selection(
    *,
    root: Path,
    selection: dict[str, Any],
    password: str,
    minimum_free_gib: float,
    backend: DownloadBackend | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    backend = backend or OfficialSoccerNetBackend()
    files = expected_files(selection)
    for index, (game_id, filename, resolution) in enumerate(files, start=1):
        ensure_free_space(root, minimum_free_gib)
        if verbose:
            print(f"[{index}/{len(files)}] {game_id}/{filename}")
        outcome = _download_one(
            root=root,
            game_id=game_id,
            filename=filename,
            resolution=resolution,
            split=selection["split"],
            password=password,
            backend=backend,
            verbose=verbose,
        )
        if verbose:
            print(f"  {outcome}")

    manifest = build_manifest(root, selection)
    atomic_write_json(root / MANIFEST_FILENAME, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=("train", "valid", "test"))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        choices=tuple(VIDEO_FILENAMES),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform network downloads. Without this flag, print the frozen plan only.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selection = make_selection(
            games=load_official_games(args.split),
            split=args.split,
            limit=args.limit,
            resolutions=args.resolutions,
        )
        print(json.dumps(selection, indent=2))
        if not args.execute:
            print("\nPreview only: no files were written and no network request was made.")
            return 0

        frozen = freeze_selection(args.root, selection)
        password = load_secret("SOCCERNET_PASSWORD", args.env_file)
        manifest = download_selection(
            root=args.root,
            selection=frozen,
            password=password,
            minimum_free_gib=args.minimum_free_gib,
            verbose=not args.quiet,
        )
        complete = manifest["summary"]["complete_game_count"]
        print(f"Validated {complete}/{args.limit} complete games in {args.root}")
        return 0 if complete == args.limit else 2
    except (SoccerNetDataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
