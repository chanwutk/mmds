from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    Filter,
    ForEach,
    GeminiPromptExecutor,
    Input,
    MMDSValidationError,
    Map,
    PromptSpec,
    Record,
    Reduce,
    ResolvedPrompt,
    StaticLLMClient,
    StaticPromptExecutor,
    Unnest,
    discover_udfs,
    execute,
    load_query,
    optimize,
    program_from_plan,
    render_query,
)
from mmds.llm_optimizer import build_rewrite_prompt, rewrite  # noqa: E402
from udfs.test_ops import add_bucket, annotate, keep_large, summarize_group  # noqa: E402


class QueryRoundTripTests(unittest.TestCase):
    def test_parse_render_parse_round_trip_for_structured_prompts(self) -> None:
        query = """
from mmds import Input, Map, Reduce, Record, ForEach

docs = Input("docs")
mapped = Map(
    docs,
    ["Summarize ", Record["title"], " from ", Record["video"]],
    schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
)
output = Reduce(
    mapped,
    "_all",
    ["Summaries:\\n", ForEach(["- ", Record["summary"], "\\n"])],
    schema={"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]},
)
"""
        program = load_query(query)
        rendered = render_query(program)
        reparsed = load_query(rendered)

        self.assertEqual(program.input_names(), ("docs",))
        self.assertEqual(rendered, render_query(reparsed))
        self.assertIn("Record", rendered)
        self.assertIn("ForEach", rendered)

    def test_program_from_runtime_plan_renders_normalized_python(self) -> None:
        plan = Filter(Map(Input("docs"), annotate), ["Keep label ", Record["label"]])
        program = program_from_plan(plan)
        rendered = render_query(program)

        self.assertIn("from udfs.test_ops import annotate", rendered)
        self.assertIn('output = Filter(step_1, ["Keep label ", Record["label"]])', rendered)


class ExecutionTests(unittest.TestCase):
    def test_execute_udf_pipeline(self) -> None:
        rows = [
            {"value": 1, "tags": ["a", "b"]},
            {"value": 2, "tags": ["c"]},
            {"value": 4, "tags": []},
        ]
        plan = Reduce(
            Unnest(Filter(Map(Input("docs"), add_bucket), keep_large), "tags", keep_empty=True),
            ["bucket"],
            summarize_group,
        )

        result = execute(plan, {"docs": rows})
        self.assertEqual(
            sorted(result, key=lambda row: row["bucket"]),
            [
                {"bucket": 1, "count": 1, "total": 2},
                {"bucket": 2, "count": 1, "total": 4},
            ],
        )

    def test_execute_structured_prompt_pipeline_with_static_executor(self) -> None:
        rows = [
            {"title": "Clip A", "video": {"type": "Video", "uri": "https://youtube.com/watch?v=1"}},
            {"title": "Clip B", "video": {"type": "Video", "uri": "https://youtube.com/watch?v=2"}},
        ]
        map_plan = Map(
            Input("docs"),
            ["Summarize ", Record["title"], " from ", Record["video"]],
            schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        )
        reduce_plan = Reduce(
            map_plan,
            "_all",
            ["Bullets:\n", ForEach(["- ", Record["summary"], "\n"])],
            schema={"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]},
        )

        map_key = map_plan.spec.cache_key()
        reduce_key = reduce_plan.spec.cache_key()
        executor = StaticPromptExecutor(
            {
                ("map", map_key): lambda prompt, payload, ctx: {
                    "summary": f"{prompt.parts[1]}::{prompt.parts[3]['uri']}"
                },
                ("reduce", reduce_key): lambda prompt, payload, ctx: {
                    "report": "".join(str(part) for part in prompt.parts)
                },
            }
        )

        result = execute(reduce_plan, {"docs": rows}, prompt_executor=executor)
        self.assertEqual(
            result,
            [
                {
                    "report": "Bullets:\n- Clip A::https://youtube.com/watch?v=1\n- Clip B::https://youtube.com/watch?v=2\n"
                }
            ],
        )

    def test_unnest_scalar_and_empty_behavior(self) -> None:
        rows = [
            {"value": 1, "tags": "solo"},
            {"value": 2, "tags": []},
            {"value": 3},
        ]
        plan = Unnest(Input("docs"), "tags", keep_empty=True)
        result = execute(plan, {"docs": rows})

        self.assertEqual(
            result,
            [
                {"value": 1, "tags": "solo"},
                {"value": 2, "tags": None},
                {"value": 3, "tags": None},
            ],
        )


class ValidationTests(unittest.TestCase):
    def test_lambda_specs_are_rejected(self) -> None:
        with self.assertRaises(MMDSValidationError):
            Map(Input("docs"), lambda row: row)

    def test_parser_rejects_unsupported_python(self) -> None:
        query = """
from mmds import Input, Map

docs = Input("docs")
for row in []:
    docs = Map(docs, "noop", schema={"type": "object"})
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_parser_rejects_non_udf_callable_specs(self) -> None:
        query = """
from mmds import Input, Map
from math import ceil

docs = Input("docs")
output = Map(docs, ceil)
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_reduce_requires_foreach_for_record_access(self) -> None:
        query = """
from mmds import Input, Reduce, Record

docs = Input("docs")
output = Reduce(
    docs,
    "_all",
    ["Summary for ", Record["title"]],
    schema={"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]},
)
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_foreach_is_rejected_outside_reduce(self) -> None:
        with self.assertRaises(MMDSValidationError):
            Map(
                Input("docs"),
                [ForEach(["- ", Record["title"]])],
                schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
            )


class OptimizerTests(unittest.TestCase):
    def test_rule_optimizer_preserves_results(self) -> None:
        rows = [{"value": 2}, {"value": 4}]
        plan = Map(Input("docs"), annotate)

        original = execute(plan, {"docs": rows})
        optimized = execute(optimize(plan), {"docs": rows})

        self.assertEqual(original, optimized)

    def test_llm_optimizer_rewrites_and_validates_inputs(self) -> None:
        query = """
from mmds import Input, Map, Filter, Record
from udfs.test_ops import annotate

docs = Input("docs")
tagged = Map(docs, annotate)
output = Filter(tagged, ["keep ", Record["label"]])
"""
        client = StaticLLMClient(
            """
```python
from mmds import Input, Map, Filter, Record
from udfs.test_ops import annotate

docs = Input("docs")
mapped = Map(docs, annotate)
output = Filter(mapped, ["keep ", Record["label"]])
```
"""
        )

        rewritten = rewrite(query, client, objective="latency")
        self.assertIn('output = Filter(mapped, ["keep ", Record["label"]])', rewritten)
        self.assertIn("Record, and ForEach", build_rewrite_prompt(query, objective="latency"))

    def test_llm_optimizer_rejects_changed_inputs(self) -> None:
        query = """
from mmds import Input, Map
from udfs.test_ops import annotate

docs = Input("docs")
output = Map(docs, annotate)
"""
        client = StaticLLMClient(
            """
from mmds import Input, Map
from udfs.test_ops import annotate

other = Input("other")
output = Map(other, annotate)
"""
        )
        with self.assertRaises(MMDSValidationError):
            rewrite(query, client)


class GeminiExecutorTests(unittest.TestCase):
    def test_gemini_executor_builds_video_uri_parts_and_schema(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Summarize ", Record["video"]),
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        )
        resolved = ResolvedPrompt(parts=("Summarize ", {"type": "Video", "uri": "https://youtube.com/watch?v=demo"}), output_schema=prompt.output_schema)

        result = executor.execute("map", prompt, resolved, payload={}, context={})
        self.assertEqual(result, {"summary": "done"})
        self.assertEqual(fake_client.models.calls[0]["config"]["response_mime_type"], "application/json")
        self.assertEqual(
            fake_client.models.calls[0]["config"]["response_json_schema"],
            prompt.output_schema,
        )
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[0].text, "Summarize ")
        self.assertEqual(content.parts[1].file_data.file_uri, "https://youtube.com/watch?v=demo")

    def test_gemini_executor_uploads_local_video_files(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        )
        resolved = ResolvedPrompt(parts=("Watch ", {"type": "Video", "path": "/tmp/demo.mp4"}), output_schema=prompt.output_schema)

        executor.execute("map", prompt, resolved, payload={}, context={})
        self.assertEqual(fake_client.files.upload_calls, ["/tmp/demo.mp4"])
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[1].file_data.file_uri, "uploaded://video")


class UdfCatalogTests(unittest.TestCase):
    def test_discover_udfs_finds_python_and_stub_entries(self) -> None:
        catalog = discover_udfs(ROOT / "udfs")
        implemented = catalog.get("udfs.test_ops", "annotate")
        planned = catalog.get("udfs.planned_only", "synthesize_summary")

        self.assertIsNotNone(implemented)
        self.assertTrue(implemented.implemented)
        self.assertIsNotNone(planned)
        self.assertFalse(planned.implemented)
        self.assertIn("def synthesize_summary", planned.signature)


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakePart:
    def __init__(self, text=None, inline_data=None, file_data=None, video_metadata=None):
        self.text = text
        self.inline_data = inline_data
        self.file_data = file_data
        self.video_metadata = video_metadata


class _FakeBlob:
    def __init__(self, data, mime_type):
        self.data = data
        self.mime_type = mime_type


class _FakeFileData:
    def __init__(self, file_uri, mime_type=None):
        self.file_uri = file_uri
        self.mime_type = mime_type


class _FakeVideoMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTypes:
    Content = _FakeContent
    Part = _FakePart
    Blob = _FakeBlob
    FileData = _FakeFileData
    VideoMetadata = _FakeVideoMetadata


class _FakeUploadedFile:
    def __init__(self):
        self.state = type("State", (), {"name": "ACTIVE"})()
        self.uri = "uploaded://video"
        self.mime_type = "video/mp4"
        self.name = "file-1"


class _FakeFiles:
    def __init__(self):
        self.upload_calls = []

    def upload(self, file):
        self.upload_calls.append(file)
        return _FakeUploadedFile()

    def get(self, name):
        return _FakeUploadedFile()


class _FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"text": self.text})()


class _FakeClient:
    def __init__(self, text):
        self.models = _FakeModels(text)
        self.files = _FakeFiles()


if __name__ == "__main__":
    unittest.main()
