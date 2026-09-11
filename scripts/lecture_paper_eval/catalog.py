"""Label-free sources, queries, pairings, and settings for paper evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.experiments.whisper import FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS
from scripts.lectures.catalog import LectureQuery, LectureSource


SCHEMA_VERSION = 1
MODEL = "gemini-3.1-flash-lite"
WHISPER_MODEL = "small"
LANGUAGE = "en"
VIDEO_FPS = 1.0
CANDIDATE_PADDING_SECONDS = 30.0
TIOU_THRESHOLDS = (0.1, 0.3, 0.5)
PRIMARY_TIOU_THRESHOLD = 0.3
DEDUPLICATION_TIOU_THRESHOLD = 0.8
INPUT_USD_PER_MILLION_TOKENS = 0.25
AUDIO_INPUT_USD_PER_MILLION_TOKENS = 0.50
OUTPUT_USD_PER_MILLION_TOKENS = 1.50
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"


FROZEN_LECTURES = (
    LectureSource(
        lecture_id="mit_8_03sc_lecture_03",
        title="Driven Oscillators, Transient Phenomena, Resonance",
        filename="MIT8_03SCF16_lec03_300k.mp4",
        page_url=(
            "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-"
            "fall-2016/pages/part-i-mechanical-vibrations-and-waves/lecture-3/"
        ),
        download_url=(
            "https://archive.org/download/MIT8.03SCF16/"
            "MIT8_03SCF16_lec03_300k.mp4"
        ),
    ),
    LectureSource(
        lecture_id="mit_8_03sc_lecture_07",
        title="Symmetry, Infinite Number of Coupled Oscillators",
        filename="MIT8_03SCF16_lec07_300k.mp4",
        page_url=(
            "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-"
            "fall-2016/pages/part-i-mechanical-vibrations-and-waves/lecture-7/"
        ),
        download_url=(
            "https://archive.org/download/MIT8.03SCF16/"
            "MIT8_03SCF16_lec07_300k.mp4"
        ),
    ),
    LectureSource(
        lecture_id="mit_8_03sc_lecture_09",
        title="Wave Equation, Standing Waves, Fourier Series",
        filename="MIT8_03SCF16_lec09_300k.mp4",
        page_url=(
            "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-"
            "fall-2016/pages/part-i-mechanical-vibrations-and-waves/lecture-9/"
        ),
        download_url=(
            "https://archive.org/download/MIT8.03SCF16/"
            "MIT8_03SCF16_lec09_300k.mp4"
        ),
    ),
    LectureSource(
        lecture_id="mit_8_03sc_lecture_15",
        title="Uncertainty Principle, 2D Waves",
        filename="MIT8_03SCF16_lec15_300k.mp4",
        page_url=(
            "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-"
            "fall-2016/pages/part-ii-electromagnetic-waves/lecture-15/"
        ),
        download_url=(
            "https://archive.org/download/MIT8.03SCF16/"
            "MIT8_03SCF16_lec15_300k.mp4"
        ),
    ),
)


FROZEN_QUERIES = (
    LectureQuery(
        query_id="tone_shatters_glass",
        text=(
            "Find every interval where a sustained tone physically causes a drinking "
            "glass to visibly shatter. Return only the physical shattering event. "
            "Discussion, setup, frequency adjustment, warnings, and reactions do not "
            "count."
        ),
    ),
    LectureQuery(
        query_id="heat_device_resonant_sound",
        text=(
            "Find every continuous interval where the lecturer visibly applies fire to "
            "a device and the device produces the resulting resonant sound. Discussion, "
            "preparation, and later explanation do not count."
        ),
    ),
    LectureQuery(
        query_id="hand_driven_spring_waves",
        text=(
            "Find every continuous interval where the lecturer physically manipulates a "
            "long spring by hand to create one or more visible traveling waves. "
            "Discussion or merely holding a stationary spring does not count."
        ),
    ),
    LectureQuery(
        query_id="speaker_driven_chladni_formation",
        text=(
            "Find every distinct interval where a speaker-driven plate causes particles "
            "to visibly rearrange into a Chladni nodal pattern. Discussion, setup, and "
            "showing an already-formed static pattern do not count."
        ),
    ),
)


VERIFIED_LECTURES = (
    FROZEN_LECTURES[0],
    FROZEN_LECTURES[2],
    LectureSource(
        lecture_id="mit_8_03sc_lecture_20",
        title="Interference, Soap Bubble",
        filename="MIT8_03SCF16_lec20_300k.mp4",
        page_url=(
            "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-"
            "fall-2016/pages/part-iii-optics/lecture-20/"
        ),
        download_url=(
            "https://archive.org/download/MIT8.03SCF16/"
            "MIT8_03SCF16_lec20_300k.mp4"
        ),
    ),
)


VERIFIED_QUERIES = (
    FROZEN_QUERIES[0],
    FROZEN_QUERIES[1],
    LectureQuery(
        query_id="successful_large_soap_bubble",
        text=(
            "Find every interval where the lecturer separates two soap-coated sticks "
            "and visibly creates a large, intact soap film or bubble that remains "
            "suspended between the sticks until it ruptures. Return the complete "
            "visible lifetime from formation through rupture. Partial or failed "
            "creation attempts, setup, and verbal discussion do not count."
        ),
    ),
)


@dataclass(frozen=True)
class EvaluationProfile:
    """Immutable label-free definition of one evaluation input workload."""

    profile_id: str
    experiment_name: str
    default_root: Path
    lectures: tuple[LectureSource, ...]
    queries: tuple[LectureQuery, ...]
    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        lecture_ids = [lecture.lecture_id for lecture in self.lectures]
        query_ids = [query.query_id for query in self.queries]
        if len(lecture_ids) != len(set(lecture_ids)) or not lecture_ids:
            raise ValueError("Evaluation profile lectures must be unique and nonempty")
        if len(query_ids) != len(set(query_ids)) or not query_ids:
            raise ValueError("Evaluation profile queries must be unique and nonempty")
        if len(self.pairs) != len(set(self.pairs)) or not self.pairs:
            raise ValueError("Evaluation profile pairs must be unique and nonempty")
        allowed = {
            (lecture_id, query_id)
            for lecture_id in lecture_ids
            for query_id in query_ids
        }
        if not set(self.pairs).issubset(allowed):
            raise ValueError("Evaluation profile contains an unknown lecture-query pair")


FROZEN_4X4_PROFILE = EvaluationProfile(
    profile_id="frozen-4x4",
    experiment_name="cross_modal_lecture_4x4_v1",
    default_root=Path("data/lecture_paper_eval"),
    lectures=FROZEN_LECTURES,
    queries=FROZEN_QUERIES,
    pairs=tuple(
        (lecture.lecture_id, query.query_id)
        for lecture in FROZEN_LECTURES
        for query in FROZEN_QUERIES
    ),
)


VERIFIED_3PAIR_PROFILE = EvaluationProfile(
    profile_id="verified-3pair",
    experiment_name="cross_modal_lecture_verified_3pair_v1",
    default_root=Path("data/lecture_verified_eval"),
    lectures=VERIFIED_LECTURES,
    queries=VERIFIED_QUERIES,
    pairs=(
        ("mit_8_03sc_lecture_03", "tone_shatters_glass"),
        ("mit_8_03sc_lecture_09", "heat_device_resonant_sound"),
        ("mit_8_03sc_lecture_20", "successful_large_soap_bubble"),
    ),
)


PROFILES = {
    profile.profile_id: profile
    for profile in (FROZEN_4X4_PROFILE, VERIFIED_3PAIR_PROFILE)
}
DEFAULT_PROFILE = FROZEN_4X4_PROFILE

# Backward-compatible aliases keep the completed four-by-four experiment as the
# default API and preserve its catalog payload byte-for-byte.
EXPERIMENT_NAME = DEFAULT_PROFILE.experiment_name
DEFAULT_ROOT = DEFAULT_PROFILE.default_root
LECTURES = DEFAULT_PROFILE.lectures
QUERIES = DEFAULT_PROFILE.queries


def get_profile(profile_id: str) -> EvaluationProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown lecture paper-eval profile: {profile_id}") from exc


def source_catalog_payload(
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": profile.experiment_name,
        "lectures": [asdict(source) for source in profile.lectures],
    }


def catalog_payload(
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    payload = {
        **source_catalog_payload(profile),
        "queries": [asdict(query) for query in profile.queries],
        "method_settings": {
            "model": MODEL,
            "whisper_model": WHISPER_MODEL,
            "language": LANGUAGE,
            "video_fps": VIDEO_FPS,
            "candidate_padding_seconds": CANDIDATE_PADDING_SECONDS,
            "final_segment_duration_tolerance_seconds": (
                FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS
            ),
            "tiou_thresholds": list(TIOU_THRESHOLDS),
            "primary_tiou_threshold": PRIMARY_TIOU_THRESHOLD,
            "deduplication_tiou_threshold": DEDUPLICATION_TIOU_THRESHOLD,
            "max_in_flight_provider_calls": 1,
        },
        "pricing": {
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
            "audio_input_usd_per_million_tokens": (
                AUDIO_INPUT_USD_PER_MILLION_TOKENS
            ),
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
            "source": PRICING_SOURCE,
        },
    }
    if profile.pairs != tuple(
        (lecture.lecture_id, query.query_id)
        for lecture in profile.lectures
        for query in profile.queries
    ):
        payload["input_pairs"] = [
            {"lecture_id": lecture_id, "query_id": query_id}
            for lecture_id, query_id in profile.pairs
        ]
    return payload
