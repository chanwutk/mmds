"""Pure UDFs shared by the SoccerNet goal-pushdown experiment plans."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from numbers import Real
from typing import Any


MIN_INCLUDE_SIGNAL_COUNT = 30
MAX_INCLUDE_SIGNAL_COUNT = 160
MIN_EXCLUDE_SIGNAL_COUNT = 10
MAX_EXCLUDE_SIGNAL_COUNT = 100
MAX_SIGNAL_TOKENS = 12
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalize_signal_lexicon(row: dict[str, Any]) -> dict[str, Any]:
    """Mechanically validate and normalize an LLM-generated signal lexicon."""
    raw_include = row.get("include_phrases")
    raw_exclude = row.get("exclude_phrases")
    if not isinstance(raw_include, list) or not isinstance(raw_exclude, list):
        raise ValueError("include_phrases and exclude_phrases must be lists")

    include, rejected_include = _normalize_phrase_list(raw_include)
    exclude, rejected_exclude = _normalize_phrase_list(raw_exclude)
    if not MIN_INCLUDE_SIGNAL_COUNT <= len(include) <= MAX_INCLUDE_SIGNAL_COUNT:
        raise ValueError(
            "normalized include_phrases must contain between "
            f"{MIN_INCLUDE_SIGNAL_COUNT} and {MAX_INCLUDE_SIGNAL_COUNT} entries"
        )
    if not MIN_EXCLUDE_SIGNAL_COUNT <= len(exclude) <= MAX_EXCLUDE_SIGNAL_COUNT:
        raise ValueError(
            "normalized exclude_phrases must contain between "
            f"{MIN_EXCLUDE_SIGNAL_COUNT} and {MAX_EXCLUDE_SIGNAL_COUNT} entries"
        )
    return {
        "goal_signal_lexicon": {
            "source": "llm_generated",
            "include_phrases": include,
            "exclude_phrases": exclude,
            "overlap_phrases": sorted(set(include) & set(exclude)),
            "rejected_include_phrases": rejected_include,
            "rejected_exclude_phrases": rejected_exclude,
            "matching_policy": "suppress only if every include span is inside an exclude span",
        }
    }


def signal_goal_candidates(row: dict[str, Any]) -> dict[str, Any]:
    """Filter transcript segments with one frozen LLM-generated lexicon."""
    segments = row.get("transcript_segments")
    lexicon = row.get("goal_signal_lexicon")
    if not isinstance(segments, list):
        return {"goal_candidates": [], "excluded_signal_matches": []}
    if not isinstance(lexicon, Mapping):
        raise ValueError("goal_signal_lexicon must be an object")
    include_rules = _phrase_rules(lexicon.get("include_phrases"), "include_phrases")
    exclude_rules = _phrase_rules(lexicon.get("exclude_phrases"), "exclude_phrases")

    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        text = segment.get("text")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(text, str) or not isinstance(start, Real):
            continue
        tokens = _tokens(text)
        include_spans = _matching_spans(tokens, include_rules)
        if not include_spans:
            continue
        exclude_spans = _matching_spans(tokens, exclude_rules)
        independent_includes = [
            include
            for include in include_spans
            if not any(
                exclude_start <= include[0] and include[1] <= exclude_end
                for exclude_start, exclude_end, _ in exclude_spans
            )
        ]
        if not independent_includes:
            excluded.append(
                {
                    "time_seconds": float(start),
                    "segment_end_seconds": (
                        float(end) if isinstance(end, Real) else float(start)
                    ),
                    "evidence": text.strip(),
                    "contained_include_phrases": sorted(
                        {phrase for _, _, phrase in include_spans}
                    ),
                    "matched_exclude_phrases": sorted(
                        {phrase for _, _, phrase in exclude_spans}
                    ),
                }
            )
            continue
        candidates.append(
            {
                "time_seconds": float(start),
                "segment_end_seconds": float(end) if isinstance(end, Real) else float(start),
                "evidence": text.strip(),
                "signal_type": "llm_generated_lexicon",
                "matched_include_phrases": sorted(
                    {phrase for _, _, phrase in independent_includes}
                ),
                "matched_exclude_phrases": sorted(
                    {phrase for _, _, phrase in exclude_spans}
                ),
                "confidence": 1.0,
            }
        )
    return {
        "goal_candidates": candidates,
        "excluded_signal_matches": excluded,
    }


def merge_candidate_windows(row: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate points, expand them, and merge overlapping windows."""
    duration = row.get("duration_seconds")
    radius = row.get("window_radius_seconds")
    candidates = row.get("goal_candidates")
    if not isinstance(duration, Real) or float(duration) <= 0:
        raise ValueError("duration_seconds must be a positive number")
    if not isinstance(radius, Real) or float(radius) <= 0:
        raise ValueError("window_radius_seconds must be a positive number")
    if not isinstance(candidates, list):
        raise ValueError("goal_candidates must be a list")

    duration_seconds = float(duration)
    radius_seconds = float(radius)
    expanded: list[dict[str, Any]] = []
    seen_times: set[float] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        time_value = candidate.get("time_seconds")
        if not isinstance(time_value, Real):
            continue
        time_seconds = float(time_value)
        if not 0 <= time_seconds <= duration_seconds:
            continue
        rounded_time = round(time_seconds, 3)
        if rounded_time in seen_times:
            continue
        seen_times.add(rounded_time)
        expanded.append(
            {
                "start_seconds": max(0.0, time_seconds - radius_seconds),
                "end_seconds": min(duration_seconds, time_seconds + radius_seconds),
                "candidate_times_seconds": [time_seconds],
                "evidence": [str(candidate.get("evidence", ""))],
                "signal_types": [str(candidate.get("signal_type", "unknown"))],
                "matched_signals": [
                    str(signal)
                    for signal in candidate.get("matched_include_phrases", [])
                    if isinstance(signal, str)
                ],
            }
        )

    expanded.sort(key=lambda window: (window["start_seconds"], window["end_seconds"]))
    merged: list[dict[str, Any]] = []
    for window in expanded:
        if not merged or window["start_seconds"] > merged[-1]["end_seconds"]:
            merged.append(window)
            continue
        current = merged[-1]
        current["end_seconds"] = max(current["end_seconds"], window["end_seconds"])
        current["candidate_times_seconds"].extend(window["candidate_times_seconds"])
        current["evidence"].extend(window["evidence"])
        current["signal_types"].extend(window["signal_types"])
        current["matched_signals"].extend(window["matched_signals"])

    for index, window in enumerate(merged):
        window["window_id"] = index
        window["duration_seconds"] = window["end_seconds"] - window["start_seconds"]
    return {"candidate_windows": merged}


def normalize_naive_goals(row: dict[str, Any]) -> dict[str, Any]:
    """Convert explicit clip-minute/second components for a full half-video."""
    duration = row.get("duration_seconds")
    if not isinstance(duration, Real) or isinstance(duration, bool) or float(duration) <= 0:
        raise ValueError("duration_seconds must be a positive number")
    return {
        "goals": _normalize_video_goals(
            row.get("goals"),
            clip_start=0.0,
            clip_end=float(duration),
        )
    }


def normalize_candidate_clip_goals(row: dict[str, Any]) -> dict[str, Any]:
    """Map standalone-clip offsets back to the original half exactly once."""
    window = row.get("candidate_window")
    context = row.get("materialized_video_context")
    video = row.get("candidate_video")
    goals = row.get("goals")
    if not isinstance(window, Mapping) or not isinstance(context, Mapping):
        raise ValueError("candidate_window and materialized_video_context must be objects")
    if (
        not isinstance(video, Mapping)
        or video.get("type") != "Video"
        or not isinstance(video.get("path"), str)
        or not video.get("path")
    ):
        raise ValueError("candidate_video must be a standalone Video")
    if row.get("candidate_strategy") != "llm_high_recall":
        raise ValueError("candidate_strategy must be llm_high_recall")
    if not isinstance(goals, list):
        return {"transcript_video_goals": []}
    start = window.get("start_seconds")
    end = window.get("end_seconds")
    duration = row.get("duration_seconds")
    if any(
        not isinstance(value, Real) or isinstance(value, bool)
        for value in (start, end, duration)
    ):
        raise ValueError("candidate and half durations must be numeric")

    window_start = float(start)
    window_end = float(end)
    half_duration = float(duration)
    if not all(math.isfinite(value) for value in (window_start, window_end, half_duration)):
        raise ValueError("candidate and half durations must be finite")
    if not 0 <= window_start < window_end <= half_duration:
        raise ValueError("candidate window must be inside the source half")
    window_id = window.get("window_id")
    if (
        not isinstance(window_id, int)
        or isinstance(window_id, bool)
        or window_id < 0
    ):
        raise ValueError("candidate window_id must be a non-negative integer")
    if context.get("game_id") != row.get("game_id") or context.get("half") != row.get("half"):
        raise ValueError("materialized video context has the wrong source identity")
    origin = context.get("timeline_origin_seconds")
    clip_duration = context.get("duration_seconds")
    if (
        not isinstance(origin, Real)
        or isinstance(origin, bool)
        or float(origin) != 0.0
        or not isinstance(clip_duration, Real)
        or isinstance(clip_duration, bool)
        or not math.isfinite(float(clip_duration))
        or float(clip_duration) <= 0
        or context.get("media_extent") != "standalone_candidate_clip"
    ):
        raise ValueError("materialized video context violates the zero-origin clip contract")
    return {
        "transcript_video_goals": _normalize_video_goals(
            goals,
            clip_start=window_start,
            clip_end=window_end,
            window_id=window_id,
        )
    }


def normalize_transcript_goals(row: dict[str, Any]) -> dict[str, Any]:
    """Validate final transcript-only predictions without repairing timestamps."""
    duration = row.get("duration_seconds")
    segments = row.get("transcript_segments")
    goals = row.get("transcript_goals")
    if (
        not isinstance(duration, Real)
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError("duration_seconds must be a finite positive number")
    if not isinstance(segments, list):
        raise ValueError("transcript_segments must be a list")
    if not isinstance(goals, list):
        return {"transcript_goals": []}

    segment_starts: set[float] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"transcript segment {index} must be an object")
        start = segment.get("start")
        if (
            not isinstance(start, Real)
            or isinstance(start, bool)
            or not math.isfinite(float(start))
            or float(start) < 0
        ):
            raise ValueError(
                f"transcript segment {index} start must be finite and non-negative"
            )
        segment_starts.add(round(float(start), 3))

    duration_seconds = float(duration)
    normalized: list[dict[str, Any]] = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, Mapping):
            continue
        time_value = goal.get("time_seconds")
        if (
            not isinstance(time_value, Real)
            or isinstance(time_value, bool)
            or not math.isfinite(float(time_value))
        ):
            raise ValueError(
                f"transcript goal {index} time_seconds must be a finite number"
            )
        time_seconds = float(time_value)
        timestamp_error = None
        if not 0 <= time_seconds <= duration_seconds:
            timestamp_error = "outside_half_duration"
        elif round(time_seconds, 3) not in segment_starts:
            timestamp_error = "not_transcript_segment_start"
        normalized.append(
            {
                "time_seconds": time_seconds,
                "timestamp_valid": timestamp_error is None,
                "timestamp_error": timestamp_error,
                "timestamp_source": "transcript_segment_start",
                "confidence": _bounded_confidence(goal.get("confidence")),
                "evidence": str(goal.get("evidence", "")),
            }
        )
    return {"transcript_goals": normalized}


def _bounded_confidence(value: Any) -> float:
    if not isinstance(value, Real):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _normalize_video_goals(
    goals: Any,
    *,
    clip_start: float,
    clip_end: float,
    window_id: Any = None,
) -> list[dict[str, Any]]:
    """Normalize one shared VLM timestamp contract without label-based repair."""
    if not isinstance(goals, list):
        return []
    clip_duration = clip_end - clip_start
    normalized: list[dict[str, Any]] = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, Mapping):
            continue
        clip_minute, clip_second, offset_seconds = _clip_timestamp(goal, index=index)
        timestamp_valid = offset_seconds <= clip_duration
        item = {
            "time_seconds": clip_start + offset_seconds,
            "offset_seconds": offset_seconds,
            "clip_minute": clip_minute,
            "clip_second": clip_second,
            "timestamp_valid": timestamp_valid,
            "timestamp_error": None if timestamp_valid else "outside_clip_duration",
            "confidence": _bounded_confidence(goal.get("confidence")),
            "evidence": str(goal.get("evidence", "")),
        }
        if window_id is not None:
            item["window_id"] = window_id
        normalized.append(item)
    return normalized


def _clip_timestamp(
    goal: Mapping[str, Any], *, index: int
) -> tuple[int, float, float]:
    minute = goal.get("clip_minute")
    second = goal.get("clip_second")
    if (
        not isinstance(minute, Real)
        or isinstance(minute, bool)
        or not float(minute).is_integer()
        or float(minute) < 0
    ):
        raise ValueError(f"goal {index} clip_minute must be a non-negative integer")
    if (
        not isinstance(second, Real)
        or isinstance(second, bool)
        or not 0 <= float(second) < 60
    ):
        raise ValueError(f"goal {index} clip_second must satisfy 0 <= value < 60")
    normalized_minute = int(minute)
    normalized_second = float(second)
    return (
        normalized_minute,
        normalized_second,
        normalized_minute * 60.0 + normalized_second,
    )


def _normalize_phrase_list(
    values: list[Any],
) -> tuple[list[str], list[dict[str, str]]]:
    normalized: list[str] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            rejected.append({"value": repr(value), "reason": "not_a_string"})
            continue
        tokens = _tokens(value)
        if not tokens:
            rejected.append({"value": value, "reason": "no_word_tokens"})
            continue
        if len(tokens) > MAX_SIGNAL_TOKENS:
            rejected.append({"value": value, "reason": "too_many_tokens"})
            continue
        if len(tokens) == 1 and len(tokens[0]) < 4:
            rejected.append({"value": value, "reason": "short_single_token"})
            continue
        canonical = " ".join(tokens)
        if canonical in seen:
            rejected.append({"value": value, "reason": "duplicate"})
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized, rejected


def _phrase_rules(value: Any, field_name: str) -> list[tuple[str, tuple[str, ...]]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"goal_signal_lexicon {field_name} must be a non-empty list")
    rules: list[tuple[str, tuple[str, ...]]] = []
    for phrase in value:
        if not isinstance(phrase, str):
            raise ValueError(f"goal_signal_lexicon {field_name} entries must be strings")
        tokens = tuple(_tokens(phrase))
        if not tokens:
            raise ValueError(f"goal_signal_lexicon {field_name} entries cannot be empty")
        rules.append((phrase, tokens))
    return rules


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _WORD_PATTERN.findall(normalized)


def _matching_spans(
    tokens: list[str],
    rules: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[int, int, str]]:
    matches: set[tuple[int, int, str]] = set()
    for phrase, phrase_tokens in rules:
        width = len(phrase_tokens)
        for start in range(len(tokens) - width + 1):
            end = start + width
            if tuple(tokens[start:end]) == phrase_tokens:
                matches.add((start, end, phrase))
    return sorted(matches)
