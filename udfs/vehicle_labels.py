"""Shared vehicle attribute vocabularies for Detect–Track–Join and semantic baselines.

Kept free of OpenCV / torch imports so examples can build prompts without pulling
the CV stack.
"""

from __future__ import annotations

VEHICLE_CLASSES: frozenset[str] = frozenset({"sedan", "suv", "truck"})

VEHICLE_COLOR_VOCAB: frozenset[str] = frozenset(
    {
        "white",
        "silver",
        "gray",
        "black",
        "beige",
        "yellow",
        "red",
        "green",
        "brown",
        "blue",
    }
)

VEHICLE_CAR_SUBTYPES: frozenset[str] = frozenset({"coupe", "suv", "sedan"})
VEHICLE_TRUCK_SUBTYPES: frozenset[str] = frozenset(
    {"tractor_trailer", "flatbed_truck", "box_truck", "pickup"}
)
VEHICLE_SUBTYPES: frozenset[str] = VEHICLE_CAR_SUBTYPES | VEHICLE_TRUCK_SUBTYPES


def format_attribute_vocab_for_prompt() -> str:
    """Return the shared class / color / subtype / timeline instructions for LLM prompts."""
    classes = ", ".join(sorted(VEHICLE_CLASSES))
    colors = ", ".join(sorted(VEHICLE_COLOR_VOCAB))
    truck_subtypes = ", ".join(sorted(VEHICLE_TRUCK_SUBTYPES))
    car_subtypes = ", ".join(sorted(VEHICLE_CAR_SUBTYPES))
    return (
        f"Vehicle class must be one of: {classes}. "
        f"Colors must be one of: {colors}. "
        f"Truck subtypes must be one of: {truck_subtypes}. "
        f"Car subtypes must be one of: {car_subtypes}. "
        "For each vehicle, report entered/exited times as seconds from the start "
        "of the underlying source video (absolute Detect frame timeline: "
        "time = frame_idx / fps; 0 is the first frame of the file), plus "
        "a match_score in [0, 1]."
    )


__all__ = [
    "VEHICLE_CAR_SUBTYPES",
    "VEHICLE_CLASSES",
    "VEHICLE_COLOR_VOCAB",
    "VEHICLE_SUBTYPES",
    "VEHICLE_TRUCK_SUBTYPES",
    "format_attribute_vocab_for_prompt",
]
