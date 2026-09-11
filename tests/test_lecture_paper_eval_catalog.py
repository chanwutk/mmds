from __future__ import annotations

import unittest

from mmds.model import PromptSpec

from scripts.lecture_paper_eval.annotations import (
    POSITIVE_INTERVALS,
    VERIFIED_POSITIVE_INTERVALS,
)
from scripts.lecture_paper_eval.catalog import (
    LECTURES,
    QUERIES,
    VERIFIED_3PAIR_PROFILE,
    catalog_payload,
)
from scripts.lecture_paper_eval.plans import (
    VIDEO_EVENT_SCHEMA,
    VIDEO_LOCALIZATION_PROMPT,
    build_naive_plan,
    build_transcript_video_plan,
)


class LecturePaperEvalCatalogTests(unittest.TestCase):
    def test_frozen_four_by_four_definition_and_human_labels(self) -> None:
        self.assertEqual(
            [lecture.lecture_id for lecture in LECTURES],
            [
                "mit_8_03sc_lecture_03",
                "mit_8_03sc_lecture_07",
                "mit_8_03sc_lecture_09",
                "mit_8_03sc_lecture_15",
            ],
        )
        self.assertNotIn("mit_8_03sc_lecture_11", {item.lecture_id for item in LECTURES})
        self.assertEqual(len(QUERIES), 4)
        self.assertEqual(len({item.query_id for item in QUERIES}), 4)
        self.assertEqual(len(LECTURES) * len(QUERIES), 16)
        self.assertEqual(sum(len(events) for events in POSITIVE_INTERVALS.values()), 7)
        self.assertEqual(len(POSITIVE_INTERVALS), 4)
        self.assertEqual(
            POSITIVE_INTERVALS[("mit_8_03sc_lecture_03", "tone_shatters_glass")],
            ((4427.0, 4432.0),),
        )
        self.assertEqual(
            POSITIVE_INTERVALS[("mit_8_03sc_lecture_07", "hand_driven_spring_waves")],
            ((2793.0, 2805.0),),
        )
        self.assertEqual(
            POSITIVE_INTERVALS[("mit_8_03sc_lecture_09", "heat_device_resonant_sound")],
            ((1643.0, 1654.0),),
        )
        self.assertEqual(
            POSITIVE_INTERVALS[
                ("mit_8_03sc_lecture_15", "speaker_driven_chladni_formation")
            ],
            (
                (4195.0, 4225.0),
                (4289.0, 4300.0),
                (4338.0, 4360.0),
                (4375.0, 4395.0),
            ),
        )

    def test_catalog_contains_no_labels(self) -> None:
        payload = catalog_payload()
        serialized = repr(payload).casefold()
        for forbidden in ("ground_truth", "annotation", "expected", "label"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(payload["method_settings"]["video_fps"], 1.0)
        self.assertEqual(
            payload["method_settings"]["max_in_flight_provider_calls"], 1
        )
        self.assertNotIn("input_pairs", payload)

    def test_verified_profile_has_three_matched_pairs_and_four_events(self) -> None:
        profile = VERIFIED_3PAIR_PROFILE
        self.assertEqual(
            [lecture.lecture_id for lecture in profile.lectures],
            [
                "mit_8_03sc_lecture_03",
                "mit_8_03sc_lecture_09",
                "mit_8_03sc_lecture_20",
            ],
        )
        self.assertEqual(
            profile.pairs,
            (
                ("mit_8_03sc_lecture_03", "tone_shatters_glass"),
                ("mit_8_03sc_lecture_09", "heat_device_resonant_sound"),
                ("mit_8_03sc_lecture_20", "successful_large_soap_bubble"),
            ),
        )
        self.assertEqual(
            VERIFIED_POSITIVE_INTERVALS[
                ("mit_8_03sc_lecture_20", "successful_large_soap_bubble")
            ],
            ((435.0, 441.0), (460.0, 466.0)),
        )
        self.assertEqual(
            sum(len(events) for events in VERIFIED_POSITIVE_INTERVALS.values()),
            4,
        )
        payload = catalog_payload(profile)
        self.assertEqual(payload["experiment"], profile.experiment_name)
        self.assertEqual(payload["input_pairs"], [
            {"lecture_id": lecture_id, "query_id": query_id}
            for lecture_id, query_id in profile.pairs
        ])
        serialized = repr(payload).casefold()
        for forbidden in ("ground_truth", "annotation", "expected", "label"):
            self.assertNotIn(forbidden, serialized)

    def test_naive_and_transcript_video_share_exact_model_contract(self) -> None:
        naive = _prompt_node(build_naive_plan("input.jsonl"))
        optimized = _prompt_node(build_transcript_video_plan("clips.jsonl"))
        self.assertEqual(naive.spec.output_schema, VIDEO_EVENT_SCHEMA)
        self.assertEqual(naive.spec.output_schema, optimized.spec.output_schema)
        self.assertEqual(naive.spec.parts[1:], optimized.spec.parts[1:])
        self.assertEqual(naive.spec.parts[2], VIDEO_LOCALIZATION_PROMPT)
        self.assertEqual(naive.spec.parts[0].path, ("video",))
        self.assertEqual(optimized.spec.parts[0].path, ("candidate_video",))
        record_paths = [
            part.path for part in optimized.spec.parts if hasattr(part, "path")
        ]
        self.assertEqual(record_paths, [("candidate_video",), ("query_text",)])


def _prompt_node(plan: object) -> object:
    nodes = [
        node
        for node in plan.walk_postorder()  # type: ignore[attr-defined]
        if isinstance(node.spec, PromptSpec)
    ]
    if len(nodes) != 1:
        raise AssertionError(nodes)
    return nodes[0]


if __name__ == "__main__":
    unittest.main()
