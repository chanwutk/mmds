"""Frozen logical and physical plans for the three paper-eval methods."""

from __future__ import annotations

from typing import Any

from mmds import Input, Map, Record, Unnest
from mmds.model import DatasetExpr

from udfs.lecture_paper_eval_ops import (
    candidate_segment_ranges_to_windows,
    normalize_candidate_video_events,
    normalize_full_video_events,
    normalize_transcript_events,
)


VIDEO_LOCALIZATION_CONTRACT_VERSION = 1
TRANSCRIPT_ONLY_CONTRACT_VERSION = 1
TRANSCRIPT_CANDIDATE_CONTRACT_VERSION = 1


VIDEO_LOCALIZATION_PROMPT = """Analyze the supplied lecture video using both its
visual stream and native audio stream. Find every interval that fully satisfies
the query below.

Return the smallest continuous interval containing the direct audiovisual
evidence required by the query. Do not widen an interval to include discussion,
setup, adjustment, warnings, an already-completed static result, or reactions
that the query excludes. Return distinct repetitions as distinct events. Return
an empty events list if the requested event does not occur or is ambiguous.

All boundaries must be elapsed from the start of the SUPPLIED VIDEO, whose
timeline starts at zero. Represent each boundary as a whole elapsed minute plus
seconds within that minute, where 0 <= second < 60. Do not use a lecture clock,
transcript timestamp, outside knowledge, or an inferred source-video offset.

Query:
"""


TRANSCRIPT_ONLY_PROMPT = """Answer the query using only the supplied timestamped
lecture transcript. Find every interval where the transcript itself establishes
that the exact requested event is currently occurring.

Return inclusive start_segment_id/end_segment_id pairs copied exactly from the
supplied transcript. Use the smallest segment range that contains the event, and
return distinct repetitions separately. Exclude discussion, plans, setup,
historical descriptions, analogies, frequency adjustment, already-completed
static results, and reactions when the query excludes them. Do not assume access
to video, audio, lecture summaries, labels, or outside knowledge. Return an empty
transcript_event_ranges list when the transcript does not establish the event.
Never invent or modify a segment ID.

Query:
"""


TRANSCRIPT_CANDIDATE_PROMPT = """Act as a HIGH-RECALL transcript filter for the
query below. Select every transcript region that could plausibly contain the
requested event or its immediately surrounding setup. A later audiovisual model
will inspect the corresponding video, so uncertainty must favor recall.

Return inclusive start_segment_id/end_segment_id pairs copied exactly from the
supplied transcript. Include enough transcript context to capture the full event,
especially when the exact visual action or sound is not narrated. Retain a
topically relevant current-lecture demonstration even when the transcript alone
cannot prove the visual or audio outcome. Historical or external references may
be omitted only when the transcript clearly distinguishes them from a current
lecture event. Do not rank, cap, or discard candidates to save work. Return an
empty candidate_ranges list only when no transcript region is plausibly relevant.
Never invent or modify a segment ID.

Query:
"""


def _range_schema(field: str) -> dict[str, Any]:
    properties = {
        "start_segment_id": {"type": "integer"},
        "end_segment_id": {"type": "integer"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string", "minLength": 1},
    }
    return {
        field: {
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        }
    }


VIDEO_EVENT_SCHEMA = {
    "events": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "start_minute": {"type": "integer", "minimum": 0},
                "start_second": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 59.999999,
                },
                "end_minute": {"type": "integer", "minimum": 0},
                "end_second": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 59.999999,
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "minLength": 1},
            },
            "required": [
                "start_minute",
                "start_second",
                "end_minute",
                "end_second",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
    }
}
TRANSCRIPT_EVENT_SCHEMA = _range_schema("transcript_event_ranges")
CANDIDATE_RANGE_SCHEMA = _range_schema("candidate_ranges")


def _video_prompt(video_reference: Any) -> list[Any]:
    return [
        video_reference,
        "\n",
        VIDEO_LOCALIZATION_PROMPT,
        Record["query_text"],
    ]


def build_naive_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    localized = Map(
        source,
        _video_prompt(Record["video"]),
        schema=VIDEO_EVENT_SCHEMA,
        name="paper_eval_full_video_localization",
    )
    normalized = Map(
        localized,
        normalize_full_video_events,
        name="paper_eval_normalize_full_video_events",
    )
    return Unnest(normalized, "events", name="paper_eval_one_naive_event_per_row")


def build_transcript_only_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    localized = Map(
        source,
        [
            TRANSCRIPT_ONLY_PROMPT,
            Record["query_text"],
            "\n\nTimestamped transcript:\n",
            Record["timestamped_transcript"],
        ],
        schema=TRANSCRIPT_EVENT_SCHEMA,
        name="paper_eval_transcript_only_localization",
    )
    normalized = Map(
        localized,
        normalize_transcript_events,
        name="paper_eval_normalize_transcript_only_events",
    )
    return Unnest(
        normalized,
        "events",
        name="paper_eval_one_transcript_only_event_per_row",
    )


def build_candidate_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    selected = Map(
        source,
        [
            TRANSCRIPT_CANDIDATE_PROMPT,
            Record["query_text"],
            "\n\nTimestamped transcript:\n",
            Record["timestamped_transcript"],
        ],
        schema=CANDIDATE_RANGE_SCHEMA,
        name="paper_eval_high_recall_transcript_filter",
    )
    return Map(
        selected,
        candidate_segment_ranges_to_windows,
        name="paper_eval_validate_pad_and_merge_candidates",
    )


def build_transcript_video_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    localized = Map(
        source,
        _video_prompt(Record["candidate_video"]),
        schema=VIDEO_EVENT_SCHEMA,
        name="paper_eval_candidate_video_localization",
    )
    normalized = Map(
        localized,
        normalize_candidate_video_events,
        name="paper_eval_normalize_candidate_video_events",
    )
    return Unnest(
        normalized,
        "events",
        name="paper_eval_one_transcript_video_event_per_row",
    )

