from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from .model import MMDSValidationError, PromptSpec, ResolvedPrompt

logger = logging.getLogger(__name__)


class GeminiPromptExecutor:
    """Prompt executor backed by the Gemini API."""

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-lite-preview",
        api_key: str | None = None,
        client: Any | None = None,
        types_module: Any | None = None,
        poll_interval_seconds: float = 5.0,
        file_ready_timeout_seconds: float = 300.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._client = client
        self._types = types_module
        self.poll_interval_seconds = poll_interval_seconds
        self.file_ready_timeout_seconds = file_ready_timeout_seconds
        self._uploaded_files: dict[str, tuple[str, str | None]] = {}

    def execute(
        self,
        op_type: str,
        prompt: PromptSpec,
        resolved_prompt: ResolvedPrompt,
        payload: Any,
        context: Mapping[str, Any],
    ) -> Any:
        client = self._get_client()
        types = self._get_types()
        parts = self._build_parts(resolved_prompt.parts, client, types)
        contents = types.Content(parts=parts)
        config = self._build_config(op_type, prompt)
        logger.debug(
            "Sending Gemini prompt for %s:\n%s",
            op_type,
            _format_debug_parts(parts),
        )
        response = client.models.generate_content(model=self.model, contents=contents, config=config)
        text = getattr(response, "text", None)
        if not text:
            raise MMDSValidationError("Gemini returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MMDSValidationError(f"Gemini returned invalid JSON: {text!r}") from exc

    def _build_config(self, op_type: str, prompt: PromptSpec) -> dict[str, Any]:
        if op_type == "filter":
            schema: dict[str, Any] = {"type": "boolean"}
        else:
            if prompt.output_schema is None:
                raise MMDSValidationError(f"Prompt-backed {op_type} operations require an output schema.")
            schema = prompt.output_schema
        return {
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        }

    def _build_parts(self, values: tuple[Any, ...], client: Any, types: Any) -> list[Any]:
        parts: list[Any] = []
        text_buffer: list[str] = []

        def flush_text() -> None:
            if text_buffer:
                parts.append(types.Part(text="".join(text_buffer)))
                text_buffer.clear()

        for value in values:
            if _is_video_payload(value):
                flush_text()
                parts.append(self._build_video_part(value, client, types))
                continue
            text_buffer.append(_stringify_prompt_value(value))

        flush_text()
        return parts

    def _build_video_part(self, value: Mapping[str, Any], client: Any, types: Any) -> Any:
        metadata_kwargs = {}
        for key in ("start_offset", "end_offset", "fps"):
            if key in value:
                metadata_kwargs[key] = value[key]
        metadata = types.VideoMetadata(**metadata_kwargs) if metadata_kwargs else None

        if "bytes" in value:
            mime_type = value.get("mime_type")
            if not isinstance(mime_type, str) or not mime_type:
                raise MMDSValidationError("Inline video bytes require a mime_type.")
            return types.Part(
                inline_data=types.Blob(data=value["bytes"], mime_type=mime_type),
                video_metadata=metadata,
            )

        file_uri: str | None = None
        mime_type: str | None = value.get("mime_type")
        if "path" in value:
            file_uri, mime_type = self._upload_video_file(value["path"], client)
        elif "uri" in value:
            file_uri = value["uri"]
        elif "source" in value:
            file_uri, mime_type = self._resolve_source_video(value["source"], mime_type, client)
        else:
            raise MMDSValidationError(
                "Video prompt values must include one of 'path', 'uri', 'source', or 'bytes'."
            )

        file_data_kwargs = {"file_uri": file_uri}
        if mime_type is not None:
            file_data_kwargs["mime_type"] = mime_type
        return types.Part(
            file_data=types.FileData(**file_data_kwargs),
            video_metadata=metadata,
        )

    def _upload_video_file(self, path_value: Any, client: Any) -> tuple[str, str | None]:
        path = str(Path(path_value))
        cached = self._uploaded_files.get(path)
        if cached is not None:
            return cached

        uploaded = client.files.upload(file=path)
        deadline = time.monotonic() + self.file_ready_timeout_seconds
        while True:
            state = getattr(getattr(uploaded, "state", None), "name", None)
            if state in {None, "ACTIVE"}:
                break
            if time.monotonic() >= deadline:
                raise MMDSValidationError(f"Timed out waiting for Gemini to process {path!r}.")
            time.sleep(self.poll_interval_seconds)
            uploaded = client.files.get(name=getattr(uploaded, "name"))

        file_uri = getattr(uploaded, "uri", None)
        mime_type = getattr(uploaded, "mime_type", None) or getattr(uploaded, "mimeType", None)
        if not file_uri:
            raise MMDSValidationError(f"Gemini upload for {path!r} did not return a file URI.")
        self._uploaded_files[path] = (file_uri, mime_type)
        return self._uploaded_files[path]

    def _resolve_source_video(
        self,
        source_value: Any,
        mime_type: str | None,
        client: Any,
    ) -> tuple[str, str | None]:
        if not isinstance(source_value, str) or not source_value:
            raise MMDSValidationError("Video 'source' values must be non-empty strings.")

        parsed = urlparse(source_value)
        if parsed.scheme == "file":
            local_path = url2pathname(parsed.path)
            return self._upload_video_file(local_path, client)
        if parsed.scheme:
            return source_value, mime_type
        return self._upload_video_file(source_value, client)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise MMDSValidationError(
                "GeminiPromptExecutor requires the google-genai package to be installed."
            ) from exc
        kwargs = {"api_key": self.api_key} if self.api_key is not None else {}
        self._client = genai.Client(**kwargs)
        return self._client

    def _get_types(self) -> Any:
        if self._types is not None:
            return self._types
        try:
            from google.genai import types
        except ImportError as exc:
            raise MMDSValidationError(
                "GeminiPromptExecutor requires the google-genai package to be installed."
            ) from exc
        self._types = types
        return self._types


def _is_video_payload(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("type") == "Video"


def _stringify_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return json.dumps(value)
    if isinstance(value, bytes):
        raise MMDSValidationError("Binary prompt values must be wrapped in a video field descriptor.")
    if isinstance(value, Mapping) or isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _format_debug_parts(parts: list[Any]) -> str:
    lines: list[str] = []
    for index, part in enumerate(parts, start=1):
        lines.append(f"Part {index}: {part!r}")
    return "\n".join(lines)
