"""Human-frozen labels, kept outside every prediction input and plan module."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from typing import Final

from .catalog import EvaluationProfile


ANNOTATION_VERSION: Final[int] = 2
ANNOTATOR: Final[str] = "Sultan Muratbek"
ANNOTATION_POLICY: Final[str] = (
    "Smallest continuous interval satisfying the complete frozen query; discussion, "
    "setup, and reactions are excluded. Distinct Chladni formations are separate."
)

# Annotation version 2 corrects a clerical omission of the leading one-hour
# component from three Lecture 15 timestamps and adds the separately confirmed
# first Chladni formation. The amendment is evaluator-only: prediction inputs,
# plans, prompts, and completed prediction artifacts do not depend on this file.
POSITIVE_INTERVALS: Final[dict[tuple[str, str], tuple[tuple[float, float], ...]]] = {
    ("mit_8_03sc_lecture_03", "tone_shatters_glass"): ((4427.0, 4432.0),),
    ("mit_8_03sc_lecture_07", "hand_driven_spring_waves"): ((2793.0, 2805.0),),
    ("mit_8_03sc_lecture_09", "heat_device_resonant_sound"): ((1643.0, 1654.0),),
    ("mit_8_03sc_lecture_15", "speaker_driven_chladni_formation"): (
        (4195.0, 4225.0),
        (4289.0, 4300.0),
        (4338.0, 4360.0),
        (4375.0, 4395.0),
    ),
}


SUPERSEDED_ANNOTATION_VERSION: Final[int] = 1
SUPERSEDED_POSITIVE_INTERVALS: Final[
    dict[tuple[str, str], tuple[tuple[float, float], ...]]
] = {
    ("mit_8_03sc_lecture_03", "tone_shatters_glass"): ((4427.0, 4432.0),),
    ("mit_8_03sc_lecture_07", "hand_driven_spring_waves"): ((2793.0, 2805.0),),
    ("mit_8_03sc_lecture_09", "heat_device_resonant_sound"): ((1643.0, 1654.0),),
    ("mit_8_03sc_lecture_15", "speaker_driven_chladni_formation"): (
        (689.0, 700.0),
        (738.0, 760.0),
        (775.0, 795.0),
    ),
}

ANNOTATION_AMENDMENT_ID: Final[str] = "lecture_15_leading_hour_v1_to_v2"
ANNOTATION_AMENDMENT_DATE: Final[str] = "2026-07-20"
ANNOTATION_AMENDMENT_REASON: Final[str] = (
    "The human annotator clarified that three Lecture 15 timestamps omitted the "
    "leading one-hour component and confirmed an additional earlier Chladni formation."
)


VERIFIED_ANNOTATION_VERSION: Final[int] = 1
VERIFIED_ANNOTATOR: Final[str] = "Sultan Muratbek"
VERIFIED_ANNOTATION_POLICY: Final[str] = (
    "Smallest continuous interval satisfying the complete frozen query. The soap "
    "bubble interval starts when a large intact film first spans the separated "
    "sticks and ends when it visibly ruptures; partial or failed attempts are excluded."
)
VERIFIED_POSITIVE_INTERVALS: Final[
    dict[tuple[str, str], tuple[tuple[float, float], ...]]
] = {
    ("mit_8_03sc_lecture_03", "tone_shatters_glass"): ((4427.0, 4432.0),),
    ("mit_8_03sc_lecture_09", "heat_device_resonant_sound"): ((1643.0, 1654.0),),
    ("mit_8_03sc_lecture_20", "successful_large_soap_bubble"): (
        (435.0, 441.0),
        (460.0, 466.0),
    ),
}


@dataclass(frozen=True)
class AnnotationDefinition:
    version: int
    annotator: str
    policy: str
    positive_intervals: Mapping[
        tuple[str, str], tuple[tuple[float, float], ...]
    ]
    superseded_version: int | None = None
    superseded_positive_intervals: Mapping[
        tuple[str, str], tuple[tuple[float, float], ...]
    ] | None = None


FROZEN_ANNOTATIONS = AnnotationDefinition(
    version=ANNOTATION_VERSION,
    annotator=ANNOTATOR,
    policy=ANNOTATION_POLICY,
    positive_intervals=MappingProxyType(POSITIVE_INTERVALS),
    superseded_version=SUPERSEDED_ANNOTATION_VERSION,
    superseded_positive_intervals=MappingProxyType(
        SUPERSEDED_POSITIVE_INTERVALS
    ),
)
VERIFIED_ANNOTATIONS = AnnotationDefinition(
    version=VERIFIED_ANNOTATION_VERSION,
    annotator=VERIFIED_ANNOTATOR,
    policy=VERIFIED_ANNOTATION_POLICY,
    positive_intervals=MappingProxyType(VERIFIED_POSITIVE_INTERVALS),
)


def annotations_for(profile: EvaluationProfile) -> AnnotationDefinition:
    if profile.profile_id == "frozen-4x4":
        return FROZEN_ANNOTATIONS
    if profile.profile_id == "verified-3pair":
        return VERIFIED_ANNOTATIONS
    raise ValueError(f"No annotations are registered for profile {profile.profile_id!r}")
