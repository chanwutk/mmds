from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import parse_query  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    GeminiRewriteModel,
    MMDSRewriteError,
    PerViewTemporalPushdown,
    rewrite_once,
)


PARAMETERS = {
    "video_field": "video",
    "transcript_field": "transcript",
    "query_field": "query_text",
    "candidate_prompt": (
        "Find broad source-time transcript intervals relevant to the event query."
    ),
    "video_prompt": (
        "Inspect this selected video view for every event matching the query. "
        "Return seconds relative to the beginning of the supplied view."
    ),
}


class SequenceModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Unexpected model call")
        return self.responses.pop(0)


def _program(path: str):
    return parse_query(
        f'''
from mmds import Input, Map, Record, Unnest
rows = Input({path!r})
events = Map(
    rows,
    [
        Record["video"],
        "Find every interval satisfying the event query. Return source-time seconds.",
        Record["query_text"],
    ],
    schema={{"events": {{"type": "array", "items": {{"type": "object"}}}}}},
)
output = Unnest(events, "events")
'''
    )


def _write_dataset():
    temp_dir = tempfile.TemporaryDirectory()
    path = Path(temp_dir.name) / "lectures.jsonl"
    path.write_text(
        json.dumps(
            {
                "lecture_id": "lecture-1",
                "query_text": "SECRET QUERY VALUE",
                "video": {"type": "Video", "source": "SECRET VIDEO URL"},
                "transcript": [
                    {"start": 10, "end": 20, "text": "SECRET TRANSCRIPT"}
                ],
                "ground_truth_events": [{"start": 12, "end": 15}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return temp_dir, str(path)


class RewriteModelFlowTests(unittest.TestCase):
    def test_two_calls_select_validate_instantiate_and_apply(self) -> None:
        temp_dir, path = _write_dataset()
        self.addCleanup(temp_dir.cleanup)
        model = SequenceModel(
            '{"option_id": "0"}',
            json.dumps(PARAMETERS),
        )

        result = rewrite_once(
            _program(path),
            directives=[
                PerViewTemporalPushdown(
                    identity_fields="lecture_id",
                    padding_seconds=5,
                )
            ],
            model=model,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.directive, "temporal_pushdown_per_view")
        self.assertEqual(str(result.path), "output.source")
        self.assertEqual(dict(result.parameters), PARAMETERS)
        self.assertEqual(result.program.output_expr.kind, "unnest")
        self.assertEqual(result.program.output_expr.source.kind, "reduce")
        self.assertEqual(len(model.prompts), 2)

        selection_prompt, parameter_prompt = model.prompts
        self.assertIn('"options"', selection_prompt)
        self.assertIn('"plan"', selection_prompt)
        self.assertIn("temporal_pushdown_per_view", selection_prompt)
        self.assertNotIn('"candidate_prompt"', selection_prompt)
        self.assertIn('"candidate_prompt"', parameter_prompt)
        self.assertIn('"video_prompt"', parameter_prompt)
        self.assertIn("Find every interval", parameter_prompt)
        for secret in (
            "SECRET QUERY VALUE",
            "SECRET VIDEO URL",
            "SECRET TRANSCRIPT",
            "ground_truth_events",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, selection_prompt)
                self.assertNotIn(secret, parameter_prompt)

    def test_null_selection_returns_original_after_one_call(self) -> None:
        program = _program("missing.jsonl")
        model = SequenceModel('{"option_id": null}')

        result = rewrite_once(
            program,
            directives=[PerViewTemporalPushdown(identity_fields="lecture_id")],
            model=model,
        )

        self.assertFalse(result.applied)
        self.assertIs(result.program, program)
        self.assertEqual(len(model.prompts), 1)

    def test_invalid_selection_stops_before_parameter_call(self) -> None:
        model = SequenceModel('{"option_id": "99"}')

        with self.assertRaisesRegex(MMDSRewriteError, "unknown option"):
            rewrite_once(
                _program("missing.jsonl"),
                directives=[
                    PerViewTemporalPushdown(identity_fields="lecture_id")
                ],
                model=model,
            )

        self.assertEqual(len(model.prompts), 1)

    def test_invalid_parameters_are_rejected_after_second_call(self) -> None:
        model = SequenceModel(
            '{"option_id": "0"}',
            '{"video_field": "video"}',
        )

        with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
            rewrite_once(
                _program("missing.jsonl"),
                directives=[
                    PerViewTemporalPushdown(identity_fields="lecture_id")
                ],
                model=model,
            )

        self.assertEqual(len(model.prompts), 2)

    def test_no_matches_returns_without_calling_model(self) -> None:
        program = parse_query(
            '''
from mmds import Input
output = Input("missing.jsonl")
'''
        )
        model = SequenceModel()

        result = rewrite_once(
            program,
            directives=[PerViewTemporalPushdown(identity_fields="lecture_id")],
            model=model,
        )

        self.assertFalse(result.applied)
        self.assertEqual(model.prompts, [])

    def test_duplicate_directive_names_fail_before_model_call(self) -> None:
        model = SequenceModel()

        with self.assertRaisesRegex(MMDSRewriteError, "names must be unique"):
            rewrite_once(
                _program("missing.jsonl"),
                directives=[
                    PerViewTemporalPushdown(identity_fields="lecture_id"),
                    PerViewTemporalPushdown(identity_fields="lecture_id"),
                ],
                model=model,
            )

        self.assertEqual(model.prompts, [])


class GeminiRewriteModelTests(unittest.TestCase):
    def test_uses_json_response_mode(self) -> None:
        calls: list[dict] = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(text='{"option_id": null}')

        adapter = GeminiRewriteModel(
            model="test-model",
            client=SimpleNamespace(models=Models()),
        )

        result = adapter.generate("choose")

        self.assertEqual(result, '{"option_id": null}')
        self.assertEqual(calls[0]["model"], "test-model")
        self.assertEqual(calls[0]["contents"], "choose")
        self.assertEqual(
            calls[0]["config"],
            {"response_mime_type": "application/json"},
        )

    def test_empty_response_fails_explicitly(self) -> None:
        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text="")

        adapter = GeminiRewriteModel(
            model="test-model",
            client=SimpleNamespace(models=Models()),
        )

        with self.assertRaisesRegex(MMDSRewriteError, "empty response"):
            adapter.generate("choose")


if __name__ == "__main__":
    unittest.main()
