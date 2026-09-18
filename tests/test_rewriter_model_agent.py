from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    JointTemporalPushdown,
    MMDSRewriteError,
    ModelRewriteAgent,
    ModalitySubstitution,
    Record,
    parse_query,
    rewrite_once,
)
from mmds.model import PromptSpec  # noqa: E402


def _program():
    return parse_query(
        '''
from mmds import Input, Map, Record
rows = Input("data.jsonl")
output = Map(
    rows,
    ["Answer from ", Record["video"]],
    schema={"answer": "string"},
)
'''
    )


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Unexpected model call.")
        return self.responses.pop(0)


def _selection_response(
    directive: str = "modality_substitution",
    match_id: str = "modality_substitution:output",
) -> str:
    return (
        '{"selections": [{"directive": "'
        + directive
        + '", "match_id": "'
        + match_id
        + '"}]}'
    )


def _instantiation_response() -> str:
    return (
        '{"selections": [{"directive": "modality_substitution", '
        '"match_id": "modality_substitution:output", "params": '
        '{"video_field": "video", "transcript_field": "transcript"}}]}'
    )


class ModelRewriteAgentTests(unittest.TestCase):
    def test_uses_two_calls_with_progressive_disclosure(self) -> None:
        client = SequenceClient(
            [_selection_response(), _instantiation_response()]
        )
        agent = ModelRewriteAgent(client, model_name="test-model")

        result = rewrite_once(
            _program(),
            agent=agent,
            directives=[ModalitySubstitution()],
        )

        self.assertEqual(len(client.prompts), 2)
        self.assertNotIn("Parameter contracts", client.prompts[0])
        self.assertNotIn('"properties"', client.prompts[0])
        self.assertIn("Parameter contracts", client.prompts[1])
        self.assertIn("transcript_field", client.prompts[1])
        self.assertIn("Generate semantically valid alternative plans", client.prompts[0])
        self.assertIn('"plan"', client.prompts[0])
        self.assertIn("Answer from {video}", client.prompts[0])
        self.assertIn("Answer from {video}", client.prompts[1])

        spec = result.program.output_expr.spec
        self.assertIsInstance(spec, PromptSpec)
        self.assertIn(Record["transcript"], spec.parts)
        self.assertEqual(result.trace.agent_details["model"], "test-model")
        self.assertEqual(
            result.trace.agent_details["selection_response"],
            _selection_response(),
        )

    def test_each_call_receives_only_relevant_compact_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "item_id": "item-1",
                        "video": {"type": "Video", "source": "video.mp4"},
                        "transcript": [
                            {"start": 1, "end": 2, "text": "spoken words"}
                        ],
                        "download_url": "https://example.com/video.mp4",
                        "transcript_sha256": "abc123",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            program = parse_query(
                f'''
from mmds import Input, Map, Record
rows = Input({str(path)!r})
output = Map(rows, ["Answer from ", Record["video"]], schema={{"answer": "string"}})
'''
            )
            client = SequenceClient(
                [_selection_response(), _instantiation_response()]
            )

            rewrite_once(
                program,
                agent=ModelRewriteAgent(client),
                directives=[ModalitySubstitution(), JointTemporalPushdown()],
            )

        selection_prompt, instantiation_prompt = client.prompts
        self.assertIn('"transcript"', selection_prompt)
        self.assertNotIn("download_url", selection_prompt)
        self.assertNotIn("transcript_sha256", selection_prompt)
        self.assertIn("temporal_pushdown_joint", selection_prompt)
        self.assertNotIn("temporal_pushdown_joint", instantiation_prompt)
        self.assertIn('"policy"', selection_prompt)
        self.assertNotIn('"policy"', instantiation_prompt)
        self.assertNotIn('"title"', instantiation_prompt)
        self.assertNotIn('"minLength"', instantiation_prompt)
        self.assertLess(len(selection_prompt), 3000)
        self.assertLess(len(instantiation_prompt), 2000)

    def test_temporal_instantiation_exposes_only_semantic_parameters(self) -> None:
        selection = (
            '{"selections": [{"directive": "temporal_pushdown_joint", '
            '"match_id": "temporal_pushdown_joint:output"}]}'
        )
        instantiation = (
            '{"selections": [{"directive": "temporal_pushdown_joint", '
            '"match_id": "temporal_pushdown_joint:output", "params": {'
            '"video_field": "video", "transcript_field": "transcript", '
            '"query_field": "question", '
            '"candidate_prompt": "Find high-recall candidate intervals."}}]}'
        )
        client = SequenceClient([selection, instantiation])
        agent = ModelRewriteAgent(client)
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("data.jsonl")
output = Map(
    rows,
    ["Find ", Record["question"], " in ", Record["video"]],
    schema={"answer": "string"},
)
'''
        )

        result = rewrite_once(
            program,
            agent=agent,
            directives=[
                JointTemporalPushdown(
                    identity_fields="item_id",
                    padding_seconds=10,
                    max_views=4,
                    max_total_video_seconds=120,
                )
            ],
        )

        parameter_prompt = client.prompts[1]
        self.assertIn('"candidate_prompt"', parameter_prompt)
        self.assertIn('"query_field"', parameter_prompt)
        self.assertIn("do not ask this stage to extract video", parameter_prompt)
        self.assertNotIn('"group_by"', parameter_prompt)
        self.assertNotIn('"views_field"', parameter_prompt)
        self.assertNotIn('"padding_seconds"', parameter_prompt)
        self.assertNotIn('"max_views"', parameter_prompt)
        self.assertNotIn('"max_total_video_seconds"', parameter_prompt)
        self.assertEqual(
            result.program.output_expr.spec.group_by,
            ("item_id", "question"),
        )
        self.assertEqual(result.program.output_expr.spec.padding_time, 10.0)
        self.assertEqual(result.program.output_expr.spec.max_views, 4)
        self.assertEqual(
            result.program.output_expr.spec.max_total_video_seconds,
            120.0,
        )

    def test_empty_selection_uses_only_one_call(self) -> None:
        client = SequenceClient(['{"selections": []}'])
        agent = ModelRewriteAgent(client)

        with self.assertRaisesRegex(MMDSRewriteError, "did not select"):
            rewrite_once(
                _program(),
                agent=agent,
                directives=[ModalitySubstitution()],
            )

        self.assertEqual(len(client.prompts), 1)
        self.assertIsNone(agent.last_trace["instantiation_prompt"])

    def test_rejects_unoffered_selection(self) -> None:
        client = SequenceClient([_selection_response(match_id="not-offered")])

        with self.assertRaisesRegex(MMDSRewriteError, "unoffered"):
            rewrite_once(
                _program(),
                agent=ModelRewriteAgent(client),
                directives=[ModalitySubstitution()],
            )

        self.assertEqual(len(client.prompts), 1)

    def test_rejects_invalid_json(self) -> None:
        client = SequenceClient(["not JSON"])

        with self.assertRaisesRegex(MMDSRewriteError, "invalid JSON"):
            rewrite_once(
                _program(),
                agent=ModelRewriteAgent(client),
                directives=[ModalitySubstitution()],
            )

    def test_rejects_missing_instantiation(self) -> None:
        client = SequenceClient(
            [_selection_response(), '{"selections": []}']
        )

        with self.assertRaisesRegex(MMDSRewriteError, "every selected rewrite"):
            rewrite_once(
                _program(),
                agent=ModelRewriteAgent(client),
                directives=[ModalitySubstitution()],
            )

    def test_rejects_more_than_requested_candidates(self) -> None:
        selection = (
            '{"selections": ['
            '{"directive": "modality_substitution", '
            '"match_id": "modality_substitution:output"}, '
            '{"directive": "modality_substitution", '
            '"match_id": "modality_substitution:output"}'
            "]}"
        )
        client = SequenceClient([selection])

        with self.assertRaisesRegex(MMDSRewriteError, "more than 1"):
            rewrite_once(
                _program(),
                agent=ModelRewriteAgent(client),
                directives=[ModalitySubstitution()],
            )


if __name__ == "__main__":
    unittest.main()
