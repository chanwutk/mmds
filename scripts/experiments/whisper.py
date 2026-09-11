"""Generic OpenAI Whisper wrapper and normalized segment contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .common import ExperimentDataError


NORMALIZATION_CONTRACT_VERSION = 2
FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS = 0.1


class TranscriptionModel(Protocol):
    def transcribe(self, video_path: Path, *, language: str) -> Mapping[str, Any]: ...


class WhisperTranscriptionModel:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ExperimentDataError(
                "openai-whisper is required for transcription"
            ) from exc
        kwargs = {"device": self.device} if self.device else {}
        self._model = whisper.load_model(self.model_name, **kwargs)
        return self._model

    def transcribe(self, video_path: Path, *, language: str) -> Mapping[str, Any]:
        return self._load_model().transcribe(
            str(video_path),
            language=language,
            task="transcribe",
            verbose=False,
            condition_on_previous_text=True,
            fp16=False,
        )


def normalize_transcription(
    result: Mapping[str, Any],
    *,
    source_id: str,
    source_path: str,
    source_sha256: str,
    model_name: str,
    source_duration_seconds: float | None = None,
    final_segment_duration_tolerance_seconds: float = (
        FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS
    ),
) -> dict[str, Any]:
    raw_segments = result.get("segments") or []
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ExperimentDataError("Whisper segments must be a non-empty list")
    duration = _optional_positive_finite(
        source_duration_seconds, "source_duration_seconds"
    )
    tolerance = _nonnegative_finite(
        final_segment_duration_tolerance_seconds,
        "final_segment_duration_tolerance_seconds",
    )
    segments: list[dict[str, Any]] = []
    boundary_adjustments: list[dict[str, Any]] = []
    previous_start = -math.inf
    previous_end = -math.inf
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise ExperimentDataError(f"Whisper segment {index} must be an object")
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentDataError(
                f"Whisper segment {index} has invalid boundaries"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ExperimentDataError(
                f"Whisper segment {index} must satisfy finite 0 <= start < end"
            )
        if start < previous_start or end < previous_end:
            raise ExperimentDataError(
                f"Whisper segment {index} is not chronologically ordered"
            )
        normalized_end = end
        adjustment: dict[str, Any] | None = None
        if duration is not None and end > duration:
            overrun = end - duration
            if (
                index != len(raw_segments) - 1
                or start >= duration
                or overrun > tolerance
            ):
                raise ExperimentDataError(
                    f"Whisper segment {index} exceeds the source duration by "
                    f"{overrun:.6f} seconds"
                )
            normalized_end = duration
            adjustment = {
                "segment_id": index,
                "field": "end_seconds",
                "raw_value": end,
                "normalized_value": normalized_end,
                "overrun_seconds": overrun,
                "reason": "final_segment_timestamp_quantization",
            }
            boundary_adjustments.append(adjustment)
        no_speech_probability = _finite_float(
            raw.get("no_speech_prob", 0.0), f"Whisper segment {index} no_speech_prob"
        )
        average_log_probability = _finite_float(
            raw.get("avg_logprob", 0.0), f"Whisper segment {index} avg_logprob"
        )
        normalized_segment = {
            "segment_id": index,
            "start_seconds": start,
            "end_seconds": normalized_end,
            "text": str(raw.get("text", "")).strip(),
            "no_speech_probability": no_speech_probability,
            "average_log_probability": average_log_probability,
        }
        if adjustment is not None:
            normalized_segment["raw_end_seconds"] = end
            normalized_segment["boundary_adjustment"] = adjustment["reason"]
        segments.append(normalized_segment)
        previous_start, previous_end = start, end
    return {
        "schema_version": 1,
        "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
        "source_id": source_id,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "model": model_name,
        "source_duration_seconds": duration,
        "final_segment_duration_tolerance_seconds": tolerance,
        "boundary_adjustments": boundary_adjustments,
        "language": str(result.get("language", "unknown")),
        "text": str(result.get("text", "")).strip(),
        "segments": segments,
    }


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentDataError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ExperimentDataError(f"{label} must be finite")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0:
        raise ExperimentDataError(f"{label} must be non-negative")
    return result


def _optional_positive_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, label)
    if result <= 0:
        raise ExperimentDataError(f"{label} must be positive")
    return result
