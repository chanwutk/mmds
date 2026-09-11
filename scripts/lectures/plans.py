"""Frozen prompts, schemas, and MMDS plans for lecture retrieval."""

from __future__ import annotations

from typing import Any

from mmds import Filter, Input, Map, Record, Unnest
from mmds.model import DatasetExpr
from udfs.lecture_ops import (
    attach_candidate_video_view,
    candidate_segment_ranges_to_windows,
    has_events,
    is_event_present,
    normalize_naive_events,
    normalize_naive_episodes,
    normalize_condition_evidence_to_event,
    normalize_required_conditions,
    normalize_role_aware_condition_evidence_to_event,
    normalize_role_aware_required_conditions,
    normalize_transcript_grounded_refinement,
    normalize_transcript_events,
    normalize_verified_events,
    normalize_verified_episodes,
    transcript_episode_proposals_to_windows,
    validate_binary_verification,
)


INTERVAL_CONTRACT_VERSION = 1
BINARY_VERIFICATION_CONTRACT_VERSION = 1
EPISODE_LOCALIZATION_CONTRACT_VERSION = 1
TRANSCRIPT_EPISODE_PROPOSAL_CONTRACT_VERSION = 1
TRANSCRIPT_GROUNDED_REFINEMENT_CONTRACT_VERSION = 1
QUERY_CONDITION_CONTRACT_VERSION = 1
PREDICATE_GROUNDING_CONTRACT_VERSION = 1
ROLE_AWARE_QUERY_CONDITION_CONTRACT_VERSION = 1
ROLE_AWARE_PREDICATE_GROUNDING_CONTRACT_VERSION = 1


VERIFIER_PROMPT = """Analyze the supplied lecture video and answer the query below.
Return every interval where the requested physical or audible event ACTUALLY
occurs inside this supplied video. Discussion, prediction, setup, analogy,
historical reference, replay, or description without the event itself does not
count. Use both the visual stream and the native audio stream.

All returned times must be elapsed from the START OF THE SUPPLIED VIDEO PART,
not a lecture clock shown on screen. Represent each boundary as whole elapsed
minutes plus seconds within that minute, where 0 <= second < 60. Return the
smallest interval that contains enough direct audiovisual evidence to establish
the event. Return an empty events list when the event does not occur.

Query:
"""


BINARY_VERIFIER_PROMPT = """Analyze the supplied lecture video and decide whether
the requested physical or audible event ACTUALLY OCCURS anywhere inside this
supplied video. Use both the visual stream and the native audio stream.

Set event_present=true only when the video directly establishes the requested
event. Discussion, prediction, setup without the outcome, analogy, historical
reference, replay, or description does not count. Set event_present=false when
the evidence is absent or ambiguous. This stage performs only binary
verification: do not estimate or discuss timestamps.

Query:
"""


EPISODE_LOCALIZATION_PROMPT = """A previous binary audiovisual stage confirmed
that the requested physical or audible event occurs in the supplied lecture
video. Locate every COMPLETE CONTIGUOUS EPISODE of that event using both the
visual stream and native audio stream.

For a physical demonstration or experiment, start when the lecturer begins
physically performing the relevant experiment—not earlier verbal setup—and end
when the demonstrated outcome and its immediate physical activity conclude.
Return the complete episode, not isolated decisive moments or individual
sub-actions inside one continuing experiment. Separate genuinely distinct
repetitions into distinct episodes.

All returned times must be elapsed from the START OF THE SUPPLIED VIDEO PART,
not a lecture clock shown on screen. Represent each boundary as whole elapsed
minutes plus seconds within that minute, where 0 <= second < 60. The confirmed
event must yield at least one episode; never invent an episode outside the
supplied video.

Query:
"""


TRANSCRIPT_CANDIDATE_PROMPT = """You are the high-recall transcript pushdown stage
for a cross-modal lecture query. Select transcript segment ranges that plausibly
surround the requested ACTUAL physical or audible event. The transcript can
locate discussion and setup but cannot prove that the visual or audio outcome
occurred; a later multimodal verifier will decide that.

Return inclusive start_segment_id/end_segment_id ranges copied from the supplied
transcript. Include enough adjacent transcript context to cover the relevant
discussion or demonstration. Prefer recall when uncertain. Historical references,
examples, and verbal descriptions may be retained as candidates because the
verifier will reject them. Never invent a segment id.

Query:
"""


TRANSCRIPT_ONLY_PROMPT = """Answer the lecture query using ONLY the supplied
timestamped transcript. Return the inclusive transcript segment ranges where you
believe the requested actual event occurs. Do not assume access to video, audio,
ground truth, lecture summaries, or outside knowledge.

Count only an event that the transcript itself asserts is currently occurring.
Exclude mere plans, setup, hypothetical discussion, historical references, and
descriptions of events outside the current lecture. Never invent segment ids.
Return an empty transcript_event_ranges list if the transcript does not establish
the event.

Query:
"""


TRANSCRIPT_EPISODE_PROPOSAL_PROMPT = """Use the timestamped transcript to propose
every plausible COMPLETE PHYSICAL OR AUDIBLE EPISODE requested by the query.
These proposals are the timestamp hypotheses for a later audiovisual refinement
stage, not merely topical search windows.

For each hypothesis, copy an inclusive start_segment_id/end_segment_id pair from
the transcript. Start at the earliest segment indicating that physical
performance of the relevant demonstration or experiment has begun. End at the
final segment indicating that the same demonstrated outcome and its immediate
physical activity have concluded. Keep one continuing experiment as one range;
do not fragment it into setup actions, measurements, reactions, or other
sub-actions. A multi-minute experiment should remain one multi-minute proposal.

Prefer a modestly inclusive complete range over truncating the episode, but do
not widen a range to the entire surrounding topical discussion. Because the
transcript cannot always prove visual evidence, retain a plausible current
lecture event when uncertain; the later audiovisual stage may reject it. Exclude
purely historical or external descriptions when the transcript clearly says the
event is not happening in this lecture. Never invent segment ids.

Query:
"""


TRANSCRIPT_GROUNDED_REFINEMENT_PROMPT = """Perform timestamp detection for ONE
transcript-proposed lecture episode. The supplied video is padded context around
the proposal. The aligned transcript excerpt lists the only segment ids you may
use as boundaries.

First determine from the visual stream and native audio whether the requested
physical or audible event actually occurs in this clip. If it does not occur,
return an empty episode_refinements list. If it occurs, return exactly one
start_segment_id/end_segment_id range that bounds the COMPLETE CONTIGUOUS
EPISODE. You may accept, contract, or expand the proposal within the supplied
transcript excerpt.

Start when physical performance of the relevant experiment begins, rather than
at earlier verbal setup. End when its demonstrated outcome and immediate
physical activity conclude. Do not split one continuing experiment into its
individual actions or decisive moments. Use the transcript as the timestamp
coordinate system and the audiovisual evidence to choose the correct boundary
segments. Never output or infer free-form minute/second timestamps, and never use
a segment id absent from the aligned excerpt.

Query:
"""


QUERY_CONDITION_PROMPT = """Compile the supplied cross-modal retrieval query into
the smallest complete set of MANDATORY, DIRECTLY OBSERVABLE conditions. A video
interval satisfies the query only if every returned condition is established by
its visual stream, native audio, or their relationship.

Preserve every explicit object, action, modality, measurement requirement,
temporal relationship, and causal relationship in the query. Do not weaken a
condition to a related topic, substitute a similar experiment, or add facts not
required by the query. Combine words that form one observable relation into one
condition when separating them would lose meaning. Each description must be a
self-contained yes/no statement that can later be grounded to audiovisual
evidence. Do not mention particular lectures, timestamps, segment IDs, expected
answers, or implementation details.

Query:
"""


PREDICATE_GROUNDING_PROMPT = """Perform precise timestamp detection for ONE
transcript-proposed lecture interval. The supplied video is padded context. The
query has already been compiled into mandatory conditions, and the aligned
transcript excerpt lists the only segment IDs available as timestamp anchors.

Evaluate every required condition independently using the visual stream and
native audio. Return one condition_evidence item only when that exact condition
is directly established. Copy its condition_id exactly and select the smallest
inclusive transcript-segment range containing its direct audiovisual evidence.
Omit a condition when it is absent, ambiguous, merely discussed, only planned,
or replaced by a related but different object, action, measurement, modality,
temporal relation, or causal relation. Never treat topical similarity as
satisfaction.

The deterministic executor will accept the interval only if evidence is
returned for EVERY required condition. It will compute the final timestamp as
the smallest span covering all condition evidence. Therefore do not return one
broad range for the whole experiment, setup, or later explanation, and do not
invent a final interval yourself. Never output free-form minute/second
timestamps or use a segment ID absent from the aligned excerpt.

Query:
"""


ROLE_AWARE_QUERY_CONDITION_PROMPT = """Compile the supplied cross-modal retrieval
query into the smallest NONREDUNDANT set of mandatory, directly observable
conditions. A video interval satisfies the query only if every condition is
established by its visual stream, native audio, or their relationship.

Preserve every EXPLICIT object, action, modality, measurement, temporal
relationship, and causal relationship. Do not weaken a requirement to a related
topic, and do not introduce simultaneity, causality, actor identity, ordering, or
other constraints absent from the query. Omit conditions logically entailed by
another condition. Do not create a separate actor-role condition unless that
identity itself is essential to the requested event.

Assign each condition one role:
- anchor: its direct occurrence defines the requested event's temporal
  boundaries; use this for the target action, outcome, measurement, or explicit
  temporal/causal relation
- gate: it must be true for acceptance but its evidence must not widen the final
  timestamp; use this only for necessary qualifications, objects, modalities, or
  exclusions whose duration is not the target

Return at least one anchor. When one self-contained relation already preserves
several requirements, prefer that single condition over redundant component
conditions. Each description must be an independently auditable yes/no statement.
Do not mention lectures, timestamps, segment IDs, expected answers, examples, or
implementation details.

Query:
"""


ROLE_AWARE_PREDICATE_GROUNDING_PROMPT = """Perform precise timestamp detection
for ONE transcript-proposed lecture interval. The supplied video is padded
context. The query has been compiled into mandatory gate and anchor conditions;
the aligned transcript excerpt lists the only segment IDs available as
timestamp anchors.

Evaluate every condition independently using the visual stream and native
audio. Return one condition_evidence item only when that exact condition is
directly established. Copy its condition_id exactly and select the smallest
inclusive transcript-segment range containing direct evidence for that
condition. Omit a condition when it is absent, ambiguous, merely discussed,
planned, historical, or replaced by a related but different requirement. Never
treat topical similarity as satisfaction and never infer a temporal or causal
relation that is not directly established.

The deterministic executor rejects the proposal unless EVERY gate and anchor
has evidence. It derives the final timestamp solely from anchor evidence; gate
evidence cannot expand the result. Do not return broad setup or explanatory
ranges, do not invent a final interval, and never output free-form minute/second
timestamps or segment IDs absent from the aligned excerpt.

Query:
"""


def _video_event_schema(
    field_name: str = "events", *, require_nonempty: bool = False
) -> dict[str, Any]:
    boundary_properties = {
        "start_minute": {"type": "integer", "minimum": 0},
        "start_second": {"type": "number", "minimum": 0, "maximum": 59.999999},
        "end_minute": {"type": "integer", "minimum": 0},
        "end_second": {"type": "number", "minimum": 0, "maximum": 59.999999},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    items = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": boundary_properties,
            "required": list(boundary_properties),
            "additionalProperties": False,
        },
    }
    if require_nonempty:
        items["minItems"] = 1
    return {field_name: items}


def _segment_range_schema(field_name: str) -> dict[str, Any]:
    properties = {
        "start_segment_id": {"type": "integer", "minimum": 0},
        "end_segment_id": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    return {
        field_name: {
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        }
    }


VIDEO_EVENT_SCHEMA = _video_event_schema()
EPISODE_SCHEMA = _video_event_schema("episodes", require_nonempty=True)
CANDIDATE_RANGE_SCHEMA = _segment_range_schema("candidate_ranges")
TRANSCRIPT_EVENT_RANGE_SCHEMA = _segment_range_schema("transcript_event_ranges")
TRANSCRIPT_EPISODE_PROPOSAL_SCHEMA = _segment_range_schema("episode_proposals")
TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA = _segment_range_schema(
    "episode_refinements"
)
TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA["episode_refinements"]["maxItems"] = 1
QUERY_CONDITION_SCHEMA = {
    "condition_descriptions": {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "items": {"type": "string", "minLength": 1},
    }
}
PREDICATE_GROUNDING_SCHEMA = {
    "condition_evidence": {
        "type": "array",
        "maxItems": 8,
        "items": {
            "type": "object",
            "properties": {
                "condition_id": {"type": "string", "minLength": 1},
                "start_segment_id": {"type": "integer", "minimum": 0},
                "end_segment_id": {"type": "integer", "minimum": 0},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "minLength": 1},
            },
            "required": [
                "condition_id",
                "start_segment_id",
                "end_segment_id",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
    }
}
ROLE_AWARE_QUERY_CONDITION_SCHEMA = {
    "condition_specs": {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "items": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "minLength": 1},
                "role": {"type": "string", "enum": ["gate", "anchor"]},
            },
            "required": ["description", "role"],
            "additionalProperties": False,
        },
    }
}
BINARY_VERIFICATION_SCHEMA = {
    "verification": {
        "type": "object",
        "properties": {
            "event_present": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string", "minLength": 1},
        },
        "required": ["event_present", "confidence", "evidence"],
        "additionalProperties": False,
    }
}


def _verifier_parts(video_reference: Any, context_reference: Any) -> list[Any]:
    return [
        video_reference,
        "\n",
        VERIFIER_PROMPT,
        Record["query_text"],
        "\n\nSupplied-video context metadata:\n",
        context_reference,
    ]


def _binary_verifier_parts(
    video_reference: Any, context_reference: Any
) -> list[Any]:
    return [
        video_reference,
        "\n",
        BINARY_VERIFIER_PROMPT,
        Record["query_text"],
        "\n\nSupplied-video context metadata:\n",
        context_reference,
    ]


def _episode_localizer_parts(
    video_reference: Any, context_reference: Any
) -> list[Any]:
    return [
        video_reference,
        "\n",
        EPISODE_LOCALIZATION_PROMPT,
        Record["query_text"],
        "\n\nSupplied-video context metadata:\n",
        context_reference,
    ]


def build_naive_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    predicted = Map(
        source,
        _verifier_parts(Record["video"], Record["full_video_context"]),
        schema=VIDEO_EVENT_SCHEMA,
        name="full_lecture_audiovisual_verification",
    )
    normalized = Map(
        predicted,
        normalize_naive_events,
        name="normalize_full_lecture_event_intervals",
    )
    return Unnest(normalized, "events", name="one_row_per_naive_event")


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
        name="transcript_candidate_range_selection",
    )
    return Map(
        selected,
        candidate_segment_ranges_to_windows,
        name="ground_pad_and_merge_candidate_ranges",
    )


def build_transcript_only_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    predicted = Map(
        source,
        [
            TRANSCRIPT_ONLY_PROMPT,
            Record["query_text"],
            "\n\nTimestamped transcript:\n",
            Record["timestamped_transcript"],
        ],
        schema=TRANSCRIPT_EVENT_RANGE_SCHEMA,
        name="transcript_only_event_selection",
    )
    normalized = Map(
        predicted,
        normalize_transcript_events,
        name="ground_transcript_only_event_intervals",
    )
    return Unnest(normalized, "events", name="one_row_per_transcript_only_event")


def build_v3_transcript_proposal_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    proposed = Map(
        source,
        [
            TRANSCRIPT_EPISODE_PROPOSAL_PROMPT,
            Record["query_text"],
            "\n\nTimestamped transcript:\n",
            Record["timestamped_transcript"],
        ],
        schema=TRANSCRIPT_EPISODE_PROPOSAL_SCHEMA,
        name="v3_transcript_complete_episode_proposals",
    )
    return Map(
        proposed,
        transcript_episode_proposals_to_windows,
        name="ground_v3_episode_proposals_and_add_video_context",
    )


def build_v3_refinement_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    refined = Map(
        source,
        [
            Record["candidate_video"],
            "\n",
            TRANSCRIPT_GROUNDED_REFINEMENT_PROMPT,
            Record["query_text"],
            "\n\nTranscript-proposed episode:\n",
            Record["proposed_episode"],
            "\n\nSupplied-video and source-timeline context:\n",
            Record["v3_video_context"],
            "\n\nAligned allowed transcript excerpt:\n",
            Record["timestamped_transcript_context"],
        ],
        schema=TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA,
        name="v3_audiovisual_transcript_boundary_refinement",
    )
    normalized = Map(
        refined,
        normalize_transcript_grounded_refinement,
        name="normalize_v3_refined_segment_boundaries",
    )
    nonempty = Filter(normalized, has_events, name="keep_v3_accepted_proposals")
    return Unnest(nonempty, "events", name="one_row_per_v3_refined_episode")


def build_v4_query_condition_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    compiled = Map(
        source,
        [QUERY_CONDITION_PROMPT, Record["query_text"]],
        schema=QUERY_CONDITION_SCHEMA,
        name="v4_compile_mandatory_query_conditions",
    )
    return Map(
        compiled,
        normalize_required_conditions,
        name="assign_v4_condition_identifiers",
    )


def build_v4_predicate_grounding_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    grounded = Map(
        source,
        [
            Record["candidate_video"],
            "\n",
            PREDICATE_GROUNDING_PROMPT,
            Record["query_text"],
            "\n\nMandatory observable conditions:\n",
            Record["required_conditions"],
            "\n\nTranscript proposal used only for high-recall context:\n",
            Record["proposed_episode"],
            "\n\nSupplied-video and source-timeline context:\n",
            Record["v3_video_context"],
            "\n\nAligned allowed transcript excerpt:\n",
            Record["timestamped_transcript_context"],
        ],
        schema=PREDICATE_GROUNDING_SCHEMA,
        name="v4_ground_every_required_condition",
    )
    normalized = Map(
        grounded,
        normalize_condition_evidence_to_event,
        name="v4_require_all_conditions_and_derive_interval",
    )
    accepted = Filter(normalized, has_events, name="keep_v4_complete_condition_sets")
    return Unnest(accepted, "events", name="one_row_per_v4_grounded_interval")


def build_v5_query_condition_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    compiled = Map(
        source,
        [ROLE_AWARE_QUERY_CONDITION_PROMPT, Record["query_text"]],
        schema=ROLE_AWARE_QUERY_CONDITION_SCHEMA,
        name="v5_compile_nonredundant_gate_anchor_conditions",
    )
    return Map(
        compiled,
        normalize_role_aware_required_conditions,
        name="assign_v5_condition_identifiers_and_validate_roles",
    )


def build_v5_predicate_grounding_plan(input_path: str) -> DatasetExpr:
    source = Input(input_path)
    grounded = Map(
        source,
        [
            Record["candidate_video"],
            "\n",
            ROLE_AWARE_PREDICATE_GROUNDING_PROMPT,
            Record["query_text"],
            "\n\nMandatory gate and anchor conditions:\n",
            Record["required_conditions"],
            "\n\nTranscript proposal used only for high-recall context:\n",
            Record["proposed_episode"],
            "\n\nSupplied-video and source-timeline context:\n",
            Record["v3_video_context"],
            "\n\nAligned allowed transcript excerpt:\n",
            Record["timestamped_transcript_context"],
        ],
        schema=PREDICATE_GROUNDING_SCHEMA,
        name="v5_ground_every_gate_and_anchor_condition",
    )
    normalized = Map(
        grounded,
        normalize_role_aware_condition_evidence_to_event,
        name="v5_require_all_conditions_and_derive_anchor_interval",
    )
    accepted = Filter(normalized, has_events, name="keep_v5_complete_condition_sets")
    return Unnest(accepted, "events", name="one_row_per_v5_grounded_interval")


def build_verification_plan(candidate_input_path: str) -> DatasetExpr:
    source = Input(candidate_input_path)
    windows = Unnest(source, "candidate_windows", name="one_row_per_candidate_window")
    with_view = Map(
        windows,
        attach_candidate_video_view,
        name="attach_candidate_video_view",
    )
    return _build_verification_pipeline(
        with_view,
        Record["candidate_video"],
        Record["candidate_windows"],
        verification_name="candidate_audiovisual_verification",
        normalization_name="normalize_candidate_event_intervals",
        filter_name="keep_nonempty_candidate_results",
        unnest_name="one_row_per_verified_event",
    )


def build_materialized_verification_plan(materialized_input_path: str) -> DatasetExpr:
    """Verify standalone zero-origin clips while retaining source offsets for grounding."""
    return _build_verification_pipeline(
        Input(materialized_input_path),
        Record["candidate_video"],
        Record["materialized_video_context"],
        verification_name="materialized_candidate_audiovisual_verification",
        normalization_name="normalize_materialized_candidate_event_intervals",
        filter_name="keep_nonempty_materialized_candidate_results",
        unnest_name="one_row_per_materialized_verified_event",
    )


def build_naive_v2_binary_plan(input_path: str) -> DatasetExpr:
    return _build_binary_verification_plan(
        Input(input_path),
        Record["video"],
        Record["full_video_context"],
        verification_name="v2_full_lecture_binary_verification",
        validation_name="validate_v2_full_lecture_binary_verification",
    )


def build_materialized_v2_binary_plan(materialized_input_path: str) -> DatasetExpr:
    return _build_binary_verification_plan(
        Input(materialized_input_path),
        Record["candidate_video"],
        Record["materialized_video_context"],
        verification_name="v2_materialized_candidate_binary_verification",
        validation_name="validate_v2_materialized_candidate_binary_verification",
    )


def build_naive_v2_localization_plan(decision_input_path: str) -> DatasetExpr:
    return _build_episode_localization_plan(
        Input(decision_input_path),
        Record["video"],
        Record["full_video_context"],
        normalize_naive_episodes,
        filter_name="keep_v2_positive_full_lecture_decisions",
        localization_name="v2_full_lecture_complete_episode_localization",
        normalization_name="normalize_v2_full_lecture_complete_episodes",
        unnest_name="one_row_per_v2_naive_complete_episode",
    )


def build_materialized_v2_localization_plan(decision_input_path: str) -> DatasetExpr:
    return _build_episode_localization_plan(
        Input(decision_input_path),
        Record["candidate_video"],
        Record["materialized_video_context"],
        normalize_verified_episodes,
        filter_name="keep_v2_positive_materialized_candidate_decisions",
        localization_name="v2_materialized_candidate_complete_episode_localization",
        normalization_name="normalize_v2_materialized_complete_episodes",
        unnest_name="one_row_per_v2_optimized_complete_episode",
    )


def _build_binary_verification_plan(
    source: DatasetExpr,
    video_reference: Any,
    context_reference: Any,
    *,
    verification_name: str,
    validation_name: str,
) -> DatasetExpr:
    verified = Map(
        source,
        _binary_verifier_parts(video_reference, context_reference),
        schema=BINARY_VERIFICATION_SCHEMA,
        name=verification_name,
    )
    return Map(verified, validate_binary_verification, name=validation_name)


def _build_episode_localization_plan(
    source: DatasetExpr,
    video_reference: Any,
    context_reference: Any,
    normalizer: Any,
    *,
    filter_name: str,
    localization_name: str,
    normalization_name: str,
    unnest_name: str,
) -> DatasetExpr:
    positive = Filter(source, is_event_present, name=filter_name)
    localized = Map(
        positive,
        _episode_localizer_parts(video_reference, context_reference),
        schema=EPISODE_SCHEMA,
        name=localization_name,
    )
    normalized = Map(localized, normalizer, name=normalization_name)
    return Unnest(normalized, "events", name=unnest_name)


def _build_verification_pipeline(
    source: DatasetExpr,
    video_reference: Any,
    context_reference: Any,
    *,
    verification_name: str,
    normalization_name: str,
    filter_name: str,
    unnest_name: str,
) -> DatasetExpr:
    verified = Map(
        source,
        _verifier_parts(video_reference, context_reference),
        schema=VIDEO_EVENT_SCHEMA,
        name=verification_name,
    )
    normalized = Map(
        verified,
        normalize_verified_events,
        name=normalization_name,
    )
    nonempty = Filter(normalized, has_events, name=filter_name)
    return Unnest(nonempty, "events", name=unnest_name)
