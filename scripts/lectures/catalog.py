"""Frozen source and query catalog for the three-lecture pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.experiments.whisper import FINAL_SEGMENT_DURATION_TOLERANCE_SECONDS


SCHEMA_VERSION = 1
EXPERIMENT_NAME = "cross_modal_lecture_3x3_v1"
DEFAULT_ROOT = Path("data/lectures")
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


@dataclass(frozen=True)
class LectureSource:
    lecture_id: str
    title: str
    filename: str
    page_url: str
    download_url: str


@dataclass(frozen=True)
class LectureQuery:
    query_id: str
    text: str


LECTURES = (
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
        lecture_id="mit_8_03sc_lecture_11",
        title="Sound Waves",
        filename="MIT8_03SCF16_lec11_300k.mp4",
        page_url=(
            "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-"
            "fall-2016/pages/part-i-mechanical-vibrations-and-waves/lecture-11/"
        ),
        download_url=(
            "https://archive.org/download/MIT8.03SCF16/"
            "MIT8_03SCF16_lec11_300k.mp4"
        ),
    ),
)


QUERIES = (
    LectureQuery(
        query_id="heat_device_resonant_sound",
        text=(
            "Find every interval where the lecturer actually heats a device with "
            "fire and the device produces a sustained resonant sound. Mere verbal "
            "discussion or a historical reference does not count."
        ),
    ),
    LectureQuery(
        query_id="tone_shatters_glass",
        text=(
            "Find every interval where a sustained tone visibly causes a drinking "
            "glass to shatter. Discussion, setup, or a tone without an actual visible "
            "shattering event does not count."
        ),
    ),
    LectureQuery(
        query_id="measure_speed_of_sound",
        text=(
            "Find every interval where the lecturer physically measures the speed of "
            "sound using a tube, speaker, and microphone. Explanation without the "
            "physical measurement does not count."
        ),
    ),
)


def catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "lectures": [asdict(source) for source in LECTURES],
        "queries": [asdict(query) for query in QUERIES],
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
        },
        "pricing": {
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
            "audio_input_usd_per_million_tokens": AUDIO_INPUT_USD_PER_MILLION_TOKENS,
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
            "source": PRICING_SOURCE,
        },
    }


def source_catalog_payload() -> dict[str, Any]:
    """Return only fields that determine the downloaded source corpus.

    Query, model, and pricing changes must not make already validated source
    bytes look corrupt.  The complete experiment configuration is frozen later
    by :func:`catalog_payload` during ``prepare``.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "lectures": [asdict(source) for source in LECTURES],
    }
