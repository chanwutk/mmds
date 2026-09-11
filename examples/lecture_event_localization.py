"""Paper-aligned VRA queries for lecture event localization.

The input is JSON/JSONL with one tuple per (lecture, query) pair and fields:
``lecture_id``, ``query_id``, ``video``, ``timestamped_transcript``,
``duration_seconds``, and ``query_text``. Transcripts are materialized inputs
produced offline. For the paper's one-frame-per-second setting, ``video`` is a
``Video`` mapping with ``fps: 1.0``; materialized clips inherit that setting.
"""

from __future__ import annotations

from mmds import (
    CrossModalTemporalPushdown,
    Input,
    Map,
    ModalitySubstitution,
    PromptSpec,
    Record,
    Unnest,
)
from mmds.model import DatasetExpr


VIDEO_LOCALIZER_NAME = "lecture_video_event_localizer"

VIDEO_LOCALIZATION_INSTRUCTIONS = """Use the supplied lecture video's visual
and audio streams to find every occurrence of the requested physical event.
Return the smallest continuous interval containing the direct audiovisual
evidence. Exclude discussion, setup, adjustment, and reactions unless the query
explicitly requests them. Return distinct repetitions separately and return an
empty list when the event does not occur. Timestamps must be seconds from the
beginning of the supplied video.

Event description:
"""

TRANSCRIPT_LOCALIZATION_INSTRUCTIONS = """Use only the supplied source-aligned
lecture transcript to find every occurrence of the requested event. Return
source-video start and end times from the transcript. Exclude discussion,
setup, and historical references that do not establish that the event occurs.
Return distinct occurrences separately and an empty list when the transcript
does not establish the event.

Event description:
"""

HIGH_RECALL_CANDIDATE_INSTRUCTIONS = """Use the source-aligned lecture transcript
as a high-recall filter for the requested event. Return every source-video
interval that could plausibly contain the event or its immediate setup. A video
model will inspect the selected intervals, so include uncertain but relevant
regions. Do not cap or rank the candidates, and return an empty list only when
no transcript region is plausibly relevant.

Event description:
"""

INTERVAL_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "start_seconds": {"type": "number", "minimum": 0},
        "end_seconds": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["start_seconds", "end_seconds", "confidence", "evidence"],
    "additionalProperties": False,
}
EVENT_SCHEMA = {
    "events": {"type": "array", "items": INTERVAL_ITEM_SCHEMA},
}
CANDIDATE_SCHEMA = {
    "candidate_intervals": {"type": "array", "items": INTERVAL_ITEM_SCHEMA},
}


def build_video_only_query(input_path: str) -> DatasetExpr:
    """Baseline: apply the video event localizer to every complete lecture."""
    source = Input(input_path)
    localized = Map(
        source,
        [
            Record["video"],
            "\n",
            VIDEO_LOCALIZATION_INSTRUCTIONS,
            Record["query_text"],
        ],
        schema=EVENT_SCHEMA,
        name=VIDEO_LOCALIZER_NAME,
    )
    return Unnest(localized, "events", name="one_event_per_tuple")


def build_transcript_only_query(input_path: str) -> DatasetExpr:
    """O1: substitute source-aligned transcript localization for video."""
    baseline = build_video_only_query(input_path)
    transcript_prompt = PromptSpec(
        parts=(
            TRANSCRIPT_LOCALIZATION_INSTRUCTIONS,
            Record["query_text"],
            "\n\nSource-aligned transcript:\n",
            Record["timestamped_transcript"],
        ),
        output_schema=EVENT_SCHEMA,
    )
    return ModalitySubstitution(
        target_name=VIDEO_LOCALIZER_NAME,
        replacement_prompt=transcript_prompt,
    ).rewrite(baseline)


def build_transcript_to_video_query(input_path: str) -> DatasetExpr:
    """O2: select transcript candidates, materialize clips, then localize."""
    baseline = build_video_only_query(input_path)
    candidate_prompt = PromptSpec(
        parts=(
            HIGH_RECALL_CANDIDATE_INSTRUCTIONS,
            Record["query_text"],
            "\n\nSource-aligned transcript:\n",
            Record["timestamped_transcript"],
        ),
        output_schema=CANDIDATE_SCHEMA,
    )
    return CrossModalTemporalPushdown(
        target_name=VIDEO_LOCALIZER_NAME,
        candidate_prompt=candidate_prompt,
        group_by=("lecture_id", "query_id"),
        preserve_fields=(
            "video",
            "timestamped_transcript",
            "duration_seconds",
            "query_text",
        ),
        padding_seconds=30.0,
    ).rewrite(baseline)


__all__ = [
    "CANDIDATE_SCHEMA",
    "EVENT_SCHEMA",
    "HIGH_RECALL_CANDIDATE_INSTRUCTIONS",
    "TRANSCRIPT_LOCALIZATION_INSTRUCTIONS",
    "VIDEO_LOCALIZATION_INSTRUCTIONS",
    "VIDEO_LOCALIZER_NAME",
    "build_transcript_only_query",
    "build_transcript_to_video_query",
    "build_video_only_query",
]
