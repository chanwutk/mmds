"""Lecture event localization using each complete video."""

from mmds import Input, Map, Record, Unnest


INTERVALS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
        },
        "required": ["start", "end"],
        "additionalProperties": False,
    },
}


def build_query(input_path: str = "data/lectures.jsonl"):
    lectures = Input(input_path)
    localized = Map(
        lectures,
        [
            Record["video"],
            "Inspect this complete lecture video and find every interval that fully "
            "satisfies the following event query. Exclude setup, discussion, reactions, "
            "and failed attempts. Return an empty list if the event does not occur. "
            "Return absolute seconds from the beginning of the complete lecture.\n\n"
            "Event query:\n",
            Record["query_text"],
        ],
        schema={"events": INTERVALS_SCHEMA},
        name="localize_full_video",
    )
    return Unnest(localized, "events", name="one_event_per_row")


output = build_query()
