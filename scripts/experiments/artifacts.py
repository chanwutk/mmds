"""Content-addressed completion markers shared by experiment workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    load_json_object,
    sha256_file,
)


ErrorT = TypeVar("ErrorT", bound=Exception)


def write_completion(
    path: Path,
    *,
    stage_name: str,
    directory: Path,
    artifact_names: Sequence[str],
    dependencies: Mapping[str, Path],
    error_type: type[ErrorT] = ExperimentDataError,
) -> None:
    artifacts: dict[str, str] = {}
    for name in artifact_names:
        artifact = directory / name
        if not artifact.is_file():
            raise error_type(f"Cannot complete stage; artifact missing: {artifact}")
        artifacts[name] = sha256_file(artifact)
    dependency_hashes: dict[str, str] = {}
    for name, dependency in dependencies.items():
        if not dependency.is_file():
            raise error_type(f"Cannot complete stage; dependency missing: {dependency}")
        dependency_hashes[name] = sha256_file(dependency)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "stage_name": stage_name,
            "artifacts_sha256": artifacts,
            "dependencies_sha256": dependency_hashes,
        },
    )


def validate_completion(
    path: Path,
    *,
    stage_name: str,
    directory: Path,
    artifact_names: Sequence[str],
    dependencies: Mapping[str, Path],
    error_type: type[ErrorT] = ExperimentDataError,
) -> None:
    completion = load_json_object(path, error_type=error_type)
    if completion.get("stage_name") != stage_name:
        raise error_type(f"Completion marker has wrong stage name: {path}")
    artifacts = completion.get("artifacts_sha256")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(artifact_names):
        raise error_type(f"Completion artifact set changed: {path}")
    for name in artifact_names:
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != artifacts[name]:
            raise error_type(f"Completed artifact changed: {artifact}")
    recorded_dependencies = completion.get("dependencies_sha256")
    if (
        not isinstance(recorded_dependencies, Mapping)
        or set(recorded_dependencies) != set(dependencies)
    ):
        raise error_type(f"Completion dependency set changed: {path}")
    for name, dependency in dependencies.items():
        if (
            not dependency.is_file()
            or sha256_file(dependency) != recorded_dependencies[name]
        ):
            raise error_type(f"Completed dependency changed: {dependency}")


def refuse_completed(
    stage_directory: Path,
    *,
    error_type: type[ErrorT] = ExperimentDataError,
) -> None:
    if (stage_directory / "completion.json").exists():
        raise error_type(f"Stage is already complete and immutable: {stage_directory}")


__all__ = ["refuse_completed", "validate_completion", "write_completion"]
