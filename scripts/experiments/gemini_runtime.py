"""Resumable, measured Gemini execution shared by paper experiments."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds.execution.llm.gemini import GeminiPromptExecutor
from mmds.model import PromptSpec, ResolvedPrompt

from .common import ExperimentDataError, atomic_write_json, load_secret


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = {str(key): _jsonable(item) for key, item in value.items()}
        media_type = str(value.get("type", "")).casefold()
        if media_type in {"video", "videoview"}:
            source = value.get("path") or value.get("source")
            content_hash = value.get("sha256")
            if isinstance(content_hash, str) and len(content_hash) == 64:
                normalized["_local_file_fingerprint"] = {"sha256": content_hash}
            elif isinstance(source, str):
                path = Path(source)
                if path.is_file():
                    stat = path.stat()
                    normalized["_local_file_fingerprint"] = {
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def cache_key(
    *,
    model: str,
    op_type: str,
    prompt: PromptSpec,
    resolved_prompt: ResolvedPrompt,
) -> str:
    request = {
        "model": model,
        "op_type": op_type,
        "prompt_parts": _jsonable(resolved_prompt.parts),
        "output_schema": _jsonable(prompt.output_schema),
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return _jsonable(value)


class ApiCallRecorder:
    """Append one fsynced JSONL record for every real provider call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._context = threading.local()

    def set_cache_key(self, value: str | None) -> None:
        self._context.cache_key = value

    def record(
        self,
        *,
        model: str,
        elapsed_seconds: float,
        response: Any | None,
        error: BaseException | None,
    ) -> None:
        payload = {
            "cache_key": getattr(self._context, "cache_key", None),
            "model": model,
            "elapsed_seconds": elapsed_seconds,
            "status": "error" if error is not None else "ok",
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "response_id": getattr(response, "response_id", None),
            "model_version": getattr(response, "model_version", None),
            "usage_metadata": _model_dump(getattr(response, "usage_metadata", None)),
        }
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._write_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


class MediaUploadRecorder:
    """Append one durable record for every media upload or stage-local reuse."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def record(
        self,
        *,
        path: str,
        size_bytes: int | None,
        elapsed_seconds: float,
        cache_hit: bool,
        error: BaseException | None,
    ) -> None:
        payload = {
            "path": path,
            "size_bytes": size_bytes,
            "elapsed_seconds": elapsed_seconds,
            "status": (
                "error" if error is not None else "reused" if cache_hit else "ok"
            ),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
        }
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._write_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


class _RecordingModelsProxy:
    def __init__(self, models: Any, recorder: ApiCallRecorder) -> None:
        self._models = models
        self._recorder = recorder

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", "unknown"))
        started = time.perf_counter()
        response = None
        error = None
        try:
            response = self._models.generate_content(*args, **kwargs)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._recorder.record(
                model=model,
                elapsed_seconds=time.perf_counter() - started,
                response=response,
                error=error,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _RecordingClientProxy:
    def __init__(self, client: Any, recorder: ApiCallRecorder) -> None:
        self._client = client
        self.models = _RecordingModelsProxy(client.models, recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class CachingGeminiPromptExecutor(GeminiPromptExecutor):
    """Serialize calls, cache parsed results, and expose cache statistics."""

    def __init__(
        self,
        *,
        cache_directory: Path,
        recorder: ApiCallRecorder,
        model: str,
        client: Any,
        types_module: Any,
        media_recorder: MediaUploadRecorder | None = None,
        error_type: type[Exception] = ExperimentDataError,
    ) -> None:
        super().__init__(
            model=model,
            client=_RecordingClientProxy(client, recorder),
            types_module=types_module,
        )
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.recorder = recorder
        self.media_recorder = media_recorder or MediaUploadRecorder(
            cache_directory.parent / "media_uploads.jsonl"
        )
        self.error_type = error_type
        self.cache_hits = 0
        self.cache_misses = 0
        self._execution_lock = threading.Lock()

    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any:
        key = cache_key(
            model=self.model,
            op_type=op_type,
            prompt=prompt,
            resolved_prompt=resolved_prompt,
        )
        path = self.cache_directory / f"{key}.json"
        with self._execution_lock:
            if path.is_file():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise self.error_type(f"Corrupt API cache entry: {path}") from exc
                if not isinstance(cached, Mapping) or cached.get("cache_key") != key or "result" not in cached:
                    raise self.error_type(f"Invalid API cache entry: {path}")
                self.cache_hits += 1
                return cached["result"]

            self.cache_misses += 1
            self.recorder.set_cache_key(key)
            try:
                result = super().execute(
                    op_type, prompt, resolved_prompt, payload, context
                )
            finally:
                self.recorder.set_cache_key(None)
            atomic_write_json(path, {"cache_key": key, "result": result})
            return result

    def _upload_video_file(self, path_value: Any, client: Any) -> tuple[str, str | None]:
        path = str(Path(path_value))
        cache_hit = path in self._uploaded_files
        local_path = Path(path)
        size_bytes = local_path.stat().st_size if local_path.is_file() else None
        started = time.perf_counter()
        error: BaseException | None = None
        try:
            return super()._upload_video_file(path_value, client)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self.media_recorder.record(
                path=path,
                size_bytes=size_bytes,
                elapsed_seconds=time.perf_counter() - started,
                cache_hit=cache_hit,
                error=error,
            )


def create_experiment_executor(
    *,
    stage_directory: Path,
    model: str,
    env_file: Path = Path(".env"),
    error_type: type[Exception] = ExperimentDataError,
) -> CachingGeminiPromptExecutor:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise error_type("google-genai is required for Gemini execution") from exc

    api_key = load_secret("GEMINI_API_KEY", env_file, error_type=error_type)
    recorder = ApiCallRecorder(stage_directory / "api_calls.jsonl")
    media_recorder = MediaUploadRecorder(stage_directory / "media_uploads.jsonl")
    return CachingGeminiPromptExecutor(
        cache_directory=stage_directory / "api_cache",
        recorder=recorder,
        media_recorder=media_recorder,
        model=model,
        client=genai.Client(api_key=api_key),
        types_module=types,
        error_type=error_type,
    )


def aggregate_media_uploads(path: Path) -> dict[str, Any]:
    """Summarize stage-local media uploads without treating reuse as an upload."""
    records: list[dict[str, Any]] = []
    if path.is_file():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentDataError(
                    f"Invalid media-upload JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ExperimentDataError(
                    f"Media-upload line {line_number} of {path} must be an object"
                )
            records.append(record)
    uploads = [item for item in records if item.get("status") == "ok"]
    failures = [item for item in records if item.get("status") == "error"]
    reused = [item for item in records if item.get("status") == "reused"]
    return {
        "media_reference_count": len(records),
        "upload_attempt_count": len(uploads) + len(failures),
        "successful_upload_count": len(uploads),
        "failed_upload_count": len(failures),
        "reused_upload_count": len(reused),
        "unique_uploaded_media_count": len(
            {str(item.get("path")) for item in uploads}
        ),
        "uploaded_bytes": sum(
            int(item["size_bytes"])
            for item in uploads
            if isinstance(item.get("size_bytes"), int)
            and not isinstance(item.get("size_bytes"), bool)
        ),
        "media_upload_elapsed_seconds": sum(
            float(item.get("elapsed_seconds", 0.0))
            for item in (*uploads, *failures)
        ),
    }
