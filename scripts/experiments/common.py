"""Explicit filesystem, secret, and hashing helpers for experiment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar


class ExperimentDataError(RuntimeError):
    """Raised when durable experiment inputs or artifacts are inconsistent."""


ErrorT = TypeVar("ErrorT", bound=Exception)
_ENV_LINE_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)


def _raise(error_type: type[ErrorT], message: str) -> None:
    raise error_type(message)


def _parse_dotenv_value(
    raw_value: str,
    *,
    key: str,
    error_type: type[ErrorT] = ExperimentDataError,
) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            _raise(error_type, f"Malformed quoted value for {key} in .env")
        return value[1:-1]
    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    if any(character in value for character in ("$", "`")):
        _raise(
            error_type,
            f"Unsupported expansion syntax for {key} in .env; "
            "set it in the process environment",
        )
    return value


def load_secret(
    key: str,
    env_file: Path | None = Path(".env"),
    *,
    error_type: type[ErrorT] = ExperimentDataError,
) -> str:
    """Read one secret without ever evaluating dotenv shell syntax."""
    process_value = os.environ.get(key)
    if process_value:
        return process_value
    if env_file is None or not env_file.exists():
        _raise(
            error_type,
            f"Missing {key}; set it in the environment or provide an existing --env-file",
        )
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = _ENV_LINE_RE.match(line)
        if match and match.group("key") == key:
            value = _parse_dotenv_value(
                match.group("value"), key=key, error_type=error_type
            )
            if value:
                return value
            break
    _raise(error_type, f"{key} is missing or empty in {env_file}")


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    def write(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _atomic_write(path, write)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    def write(handle: Any) -> None:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

    _atomic_write(path, write)


def atomic_write_text(path: Path, text: str) -> None:
    def write(handle: Any) -> None:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")

    _atomic_write(path, write)


def load_json_object(
    path: Path, *, error_type: type[ErrorT] = ExperimentDataError
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _raise(error_type, f"Expected a JSON object in {path}")
    return payload


def load_jsonl_objects(
    path: Path, *, error_type: type[ErrorT] = ExperimentDataError
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise error_type(f"Cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise error_type(
                f"Invalid JSON on line {line_number} of {path}"
            ) from exc
        if not isinstance(row, dict):
            _raise(error_type, f"Expected an object on line {line_number} of {path}")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    normalized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256_text(normalized)


def require_files(
    paths: Sequence[Path],
    *,
    message: str,
    error_type: type[ErrorT] = ExperimentDataError,
) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        _raise(error_type, f"{message} Missing: {', '.join(missing)}")
