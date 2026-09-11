"""MMDS logical plans and frozen prompts for the SoccerNet goal experiment."""

from __future__ import annotations

from typing import Any

from mmds import Input, Map, Record, Unnest
from mmds.model import DatasetExpr
from udfs.soccernet_goal_ops import (
    merge_candidate_windows,
    normalize_candidate_clip_goals,
    normalize_naive_goals,
    normalize_signal_lexicon,
    normalize_transcript_goals,
    signal_goal_candidates,
)


TIMESTAMP_CONTRACT_VERSION = 2


VIDEO_GOAL_LOCALIZATION_PROMPT = """Analyze the supplied soccer broadcast video
using both its visual stream and native audio stream. Identify every moment where
an actual goal is scored during live play inside the supplied video.

Rules:
- Use only the elapsed position from the START OF THE SUPPLIED VIDEO. Its local
  timeline always starts at zero, whether it is a complete half or a shorter clip.
- Do not use the broadcast match clock, a containing-video clock, outside
  knowledge, or an inferred source-video offset.
- For each goal, return clip_minute as whole elapsed minutes and clip_second as
  seconds within that minute, where 0 <= clip_second < 60.
- Example: 34.5 seconds after this video starts must be returned as
  clip_minute=0 and clip_second=34.5. Never encode it as 0.345 or 0345.
- Do not return total elapsed seconds; the system converts the two fields.
- Do not count near misses, saves, shots off target, goal kicks, or disallowed goals.
- Do not count a replay, celebration, or discussion when the original live-play
  scoring moment itself is absent from the supplied video.
- If both live play and one or more replays appear, return only the original
  live-play goal.
- Return an empty goals list if no goal is scored.
"""


TRANSCRIPT_CANDIDATE_PROMPT = """You are a high-recall candidate generator for soccer goals.
Read the complete timestamped English commentary transcript for one match half.
Return moments that plausibly correspond to an actual goal being scored.

Include explicit and implicit signals such as a goal call, a score change, taking
the lead, an equalizer, doubling a lead, or pulling a goal back. The commentary
may announce the event shortly before or after the visual scoring moment.

Exclude generic discussion of earlier goals, hypothetical chances, near misses,
saves, goal kicks, goalkeeper discussion, and clearly disallowed goals. Prefer
high recall when uncertain; the video verifier will reject false positives.

For each candidate, return its transcript timestamp in seconds, short evidence,
a concise signal type, and confidence between 0 and 1. Do not invent timestamps.

Timestamped transcript:
"""


TRANSCRIPT_ONLY_GOAL_PROMPT = """You are given the complete timestamped English
commentary transcript for one half of a professional soccer match. Produce the
final list of every newly scored, valid goal asserted by the commentary.

This is a transcript-only decision. Do not assume access to video, evaluator
annotations, team scores, or any information outside the supplied transcript.

Rules:
- Count a goal only when the commentary asserts that a new goal has actually
  been scored during live play.
- Exclude near misses, saves, shots off target, goal kicks, hypothetical goals,
  historical statistics, recaps of earlier goals, and clearly disallowed goals.
- Do not count repeated discussion of the same goal more than once.
- For each goal, copy the START timestamp from the bracket of the transcript
  segment where the commentary first announces or clearly recognizes that goal.
- Return that copied value in time_seconds. Do not invent, interpolate, repair,
  or convert timestamps, and do not use a broadcast match clock.
- Return short transcript evidence and confidence between 0 and 1.
- Return an empty transcript_goals list if the transcript does not establish a
  newly scored valid goal.

Timestamped transcript:
"""


SIGNAL_LEXICON_PROMPT = """Synthesize a reusable lexical filter for English-language
professional soccer broadcast commentary. The target event is a newly scored,
valid goal during live play.

Generate the lexicon from generic domain knowledge only. You are not given and
must not assume any evaluation games, transcripts, teams, players, scores,
timestamps, or labels.

Return two plain-text phrase lists:
- include_phrases: 60 to 120 unique words or short phrases whose occurrence is
  evidence that a goal has just been scored. Cover explicit calls, implicit
  scoring language, and score-state changes. Enumerate useful morphological and
  commentary variants explicitly. Prefer multiword phrases; use a single word
  only when it is strongly indicative.
- exclude_phrases: 20 to 60 unique phrases that can contain or overlap goal-like
  vocabulary but clearly describe no newly scored valid goal, including restarts,
  negation, hypothetical or historical discussion, and invalidated events.

Rules:
- Emit literal phrases only: no regular expressions, placeholders, annotations,
  team names, player names, scorelines, timestamps, or explanations.
- Avoid duplicates and function words that would match ordinary commentary.
- The downstream matcher is deterministic and will not reinterpret your output.
"""


def _video_goal_list_schema(field_name: str) -> dict[str, Any]:
    """Build the shared explicit clip-timestamp contract for video plans."""
    return {
        field_name: {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip_minute": {"type": "integer", "minimum": 0},
                    "clip_second": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 59.999999,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "clip_minute",
                    "clip_second",
                    "confidence",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        }
    }


GOAL_LIST_SCHEMA = _video_goal_list_schema("goals")


CANDIDATE_LIST_SCHEMA = {
    "goal_candidates": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "time_seconds": {"type": "number"},
                "evidence": {"type": "string"},
                "signal_type": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["time_seconds", "evidence", "signal_type", "confidence"],
            "additionalProperties": False,
        },
    }
}


TRANSCRIPT_GOAL_LIST_SCHEMA = {
    "transcript_goals": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "time_seconds": {"type": "number", "minimum": 0},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
            },
            "required": ["time_seconds", "confidence", "evidence"],
            "additionalProperties": False,
        },
    }
}


SIGNAL_LEXICON_SCHEMA = {
    "include_phrases": {
        "type": "array",
        "items": {"type": "string"},
    },
    "exclude_phrases": {
        "type": "array",
        "items": {"type": "string"},
    },
}


def _video_goal_prompt(video_reference: Any) -> list[Any]:
    """Build the identical audiovisual localization request for either video shape."""
    return [
        video_reference,
        "\n",
        VIDEO_GOAL_LOCALIZATION_PROMPT,
    ]


def build_naive_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    predicted = Map(
        source,
        _video_goal_prompt(Record["video"]),
        schema=GOAL_LIST_SCHEMA,
        name="full_video_goal_detection",
    )
    normalized = Map(
        predicted,
        normalize_naive_goals,
        name="convert_clip_components_to_half_timestamps",
    )
    return Unnest(normalized, "goals", name="one_row_per_predicted_goal")


def build_llm_candidate_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    candidates = Map(
        source,
        [TRANSCRIPT_CANDIDATE_PROMPT, Record["timestamped_transcript"]],
        schema=CANDIDATE_LIST_SCHEMA,
        name="transcript_goal_candidate_generation",
    )
    return Map(
        candidates,
        merge_candidate_windows,
        name="expand_and_merge_candidate_windows",
    )


def build_transcript_only_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    predicted = Map(
        source,
        [TRANSCRIPT_ONLY_GOAL_PROMPT, Record["timestamped_transcript"]],
        schema=TRANSCRIPT_GOAL_LIST_SCHEMA,
        name="transcript_only_final_goal_detection",
    )
    normalized = Map(
        predicted,
        normalize_transcript_goals,
        name="validate_transcript_segment_timestamps",
    )
    return Unnest(
        normalized,
        "transcript_goals",
        name="one_row_per_transcript_goal",
    )


def build_signal_generation_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    generated = Map(
        source,
        SIGNAL_LEXICON_PROMPT,
        schema=SIGNAL_LEXICON_SCHEMA,
        name="generate_reusable_goal_signal_lexicon",
    )
    return Map(
        generated,
        normalize_signal_lexicon,
        name="normalize_and_validate_signal_lexicon",
    )


def build_signal_candidate_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    candidates = Map(
        source,
        signal_goal_candidates,
        name="llm_signal_lexicon_candidate_generation",
    )
    return Map(
        candidates,
        merge_candidate_windows,
        name="expand_and_merge_candidate_windows",
    )


def build_transcript_video_plan(materialized_input_path: str) -> DatasetExpr:
    """Localize goals in physically materialized standalone candidate clips."""
    source = Input(materialized_input_path)
    verified = Map(
        source,
        _video_goal_prompt(Record["candidate_video"]),
        schema=GOAL_LIST_SCHEMA,
        name="localize_goal_in_materialized_candidate_clip",
    )
    normalized = Map(
        verified,
        normalize_candidate_clip_goals,
        name="convert_materialized_clip_offsets_to_half_timestamps",
    )
    return Unnest(
        normalized,
        "transcript_video_goals",
        name="one_row_per_transcript_video_goal",
    )
