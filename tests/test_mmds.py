from __future__ import annotations

import atexit
import contextlib
import json
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
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
from mmds.optimizers.rewriter.agent import build_rewrite_prompt, rewrite  # noqa: E402
from udfs.test_ops import add_bucket, annotate, keep_large, summarize_group  # noqa: E402


class QueryRoundTripTests(unittest.TestCase):
    def test_parse_render_parse_round_trip_for_structured_prompts(self) -> None:
        query = """
from mmds import Input, Map, Reduce, Record, ForEach

docs = Input("docs.jsonl")
mapped = Map(
    docs,
    ["Summarize ", Record["title"], " from ", Record["video"]],
    schema={"summary": "string"},
)
output = Reduce(
    mapped,
    "_all",
    ["Summaries:\\n", ForEach(["- ", Record["summary"], "\\n"])],
    schema={"report": "string"},
)
"""
        program = load_query(query)
        rendered = render_query(program)
        reparsed = load_query(rendered)

        self.assertEqual(program.input_paths(), ("docs.jsonl",))
        self.assertEqual(rendered, render_query(reparsed))
        self.assertIn("Record", rendered)
        self.assertIn("ForEach", rendered)

    def test_program_from_runtime_plan_renders_normalized_python(self) -> None:
        plan = Filter(Map(Input("data/docs.jsonl"), annotate), ["Keep label ", Record["label"]])
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
        input_path = _write_jsonl_rows(rows)
        plan = Reduce(
            Unnest(Filter(Map(Input(input_path), add_bucket), keep_large), "tags", keep_empty=True),
            ["bucket"],
            summarize_group,
        )

        result = execute(plan)
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
        input_path = _write_jsonl_rows(rows)
        map_plan = Map(
            Input(input_path),
            ["Summarize ", Record["title"], " from ", Record["video"]],
            schema={"summary": "string"},
        )
        reduce_plan = Reduce(
            map_plan,
            "_all",
            ["Bullets:\n", ForEach(["- ", Record["summary"], "\n"])],
            schema={"report": "string"},
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

        result = execute(reduce_plan, prompt_executor=executor)
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
        input_path = _write_json_rows(rows)
        plan = Unnest(Input(input_path), "tags", keep_empty=True)
        result = execute(plan)

        self.assertEqual(
            result,
            [
                {"value": 1, "tags": "solo"},
                {"value": 2, "tags": None},
                {"value": 3, "tags": None},
            ],
        )

    def test_execute_query_program_resolves_relative_input_paths_from_query_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "clips.jsonl"
            data_path.write_text('{"tags": ["a", "b"]}\n{"tags": []}\n', encoding="utf-8")
            query_path = root / "query.py"
            query_path.write_text(
                'from mmds import Input, Unnest\n\n'
                'docs = Input("clips.jsonl")\n'
                'output = Unnest(docs, "tags", keep_empty=True)\n',
                encoding="utf-8",
            )

            program = load_query(query_path)
            result = execute(program)

        self.assertEqual(result, [{"tags": "a"}, {"tags": "b"}, {"tags": None}])


class ValidationTests(unittest.TestCase):
    def test_lambda_specs_are_rejected(self) -> None:
        with self.assertRaises(MMDSValidationError):
            Map(Input("docs.jsonl"), lambda row: row)

    def test_parser_rejects_unsupported_python(self) -> None:
        query = """
from mmds import Input, Map

docs = Input("docs.jsonl")
for row in []:
    docs = Map(docs, "noop", schema={"summary": "string"})
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_parser_rejects_non_udf_callable_specs(self) -> None:
        query = """
from mmds import Input, Map
from math import ceil

docs = Input("docs.jsonl")
output = Map(docs, ceil)
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_reduce_requires_foreach_for_record_access(self) -> None:
        query = """
from mmds import Input, Reduce, Record

docs = Input("docs.jsonl")
output = Reduce(
    docs,
    "_all",
    ["Summary for ", Record["title"]],
    schema={"report": "string"},
)
"""
        with self.assertRaises(MMDSValidationError):
            load_query(query)

    def test_foreach_is_rejected_outside_reduce(self) -> None:
        with self.assertRaises(MMDSValidationError):
            Map(
                Input("docs.jsonl"),
                [ForEach(["- ", Record["title"]])],
                schema={"summary": "string"},
            )

    def test_input_rejects_non_json_paths(self) -> None:
        with self.assertRaises(TypeError):
            Input("docs.csv")

    def test_legacy_object_schema_normalizes_to_concise_form(self) -> None:
        program = load_query(
            """
from mmds import Input, Map

docs = Input("docs.jsonl")
output = Map(
    docs,
    "annotate",
    schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
)
"""
        )

        rendered = render_query(program)
        self.assertIn('schema={"summary": "string"}', rendered)

    def test_prompt_spec_normalizes_legacy_object_schema(self) -> None:
        spec = PromptSpec(
            parts=("annotate",),
            output_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        )

        self.assertEqual(spec.output_schema, {"summary": "string"})


class OptimizerTests(unittest.TestCase):
    def test_rule_optimizer_preserves_results(self) -> None:
        input_path = _write_json_rows([{"value": 2}, {"value": 4}])
        plan = Map(Input(input_path), annotate)

        original = execute(plan)
        optimized = execute(optimize(plan))

        self.assertEqual(original, optimized)

    def test_llm_optimizer_rewrites_and_validates_inputs(self) -> None:
        query = """
from mmds import Input, Map, Filter, Record
from udfs.test_ops import annotate

docs = Input("docs.jsonl")
tagged = Map(docs, annotate)
output = Filter(tagged, ["keep ", Record["label"]])
"""
        client = StaticLLMClient(
            """
```python
from mmds import Input, Map, Filter, Record
from udfs.test_ops import annotate

docs = Input("docs.jsonl")
mapped = Map(docs, annotate)
output = Filter(mapped, ["keep ", Record["label"]])
```
"""
        )

        rewritten = rewrite(query, client, objective="latency")
        self.assertIn('output = Filter(mapped, ["keep ", Record["label"]])', rewritten)
        self.assertIn("Record, and ForEach", build_rewrite_prompt(query, objective="latency"))
        self.assertIn("Preserve the same Input(...) file paths.", build_rewrite_prompt(query))

    def test_llm_optimizer_rejects_changed_inputs(self) -> None:
        query = """
from mmds import Input, Map
from udfs.test_ops import annotate

docs = Input("docs.jsonl")
output = Map(docs, annotate)
"""
        client = StaticLLMClient(
            """
from mmds import Input, Map
from udfs.test_ops import annotate

other = Input("other.jsonl")
output = Map(other, annotate)
"""
        )
        with self.assertRaises(MMDSValidationError):
            rewrite(query, client)

    def test_llm_optimizer_logs_rewrite_prompt(self) -> None:
        query = """
from mmds import Input, Map
from udfs.test_ops import annotate

docs = Input("docs.jsonl")
output = Map(docs, annotate)
"""
        client = StaticLLMClient(query)
        with self.assertLogs("mmds.optimizers.rewriter.agent", level="DEBUG") as captured:
            rewrite(query, client)
        self.assertTrue(any("Sending rewrite prompt to LLM" in line for line in captured.output))
        self.assertTrue(any('docs = Input("docs.jsonl")' in line for line in captured.output))


class GeminiExecutorTests(unittest.TestCase):
    def test_gemini_executor_builds_video_uri_parts_and_schema(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Summarize ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(parts=("Summarize ", {"type": "video", "uri": "https://youtube.com/watch?v=demo"}), output_schema=prompt.output_schema)

        result = executor.execute("map", prompt, resolved, payload={}, context={})
        self.assertEqual(result, {"summary": "done"})
        self.assertEqual(fake_client.models.calls[0]["config"]["response_mime_type"], "application/json")
        self.assertEqual(
            fake_client.models.calls[0]["config"]["response_json_schema"],
            {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        )
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[0].text, "Summarize ")
        self.assertEqual(content.parts[1].file_data.file_uri, "https://youtube.com/watch?v=demo")

    def test_gemini_executor_uploads_local_video_files(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(parts=("Watch ", {"type": "Video", "path": "/tmp/demo.mp4"}), output_schema=prompt.output_schema)

        executor.execute("map", prompt, resolved, payload={}, context={})
        self.assertEqual(fake_client.files.upload_calls, ["/tmp/demo.mp4"])
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[1].file_data.file_uri, "uploaded://video")

    def test_gemini_executor_uploads_shared_local_video_once_under_concurrency(self) -> None:
        # The upload is slow, so without the upload lock every thread would
        # miss the cache and upload the same file.
        fake_client = _FakeClient('{"summary": "done"}', upload_delay_seconds=0.05)
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(parts=("Watch ", Record["video"]), output_schema={"summary": "string"})
        resolved = ResolvedPrompt(
            parts=("Watch ", {"type": "VideoView", "path": "/tmp/half.mkv", "start": 0, "end": 300}),
            output_schema=prompt.output_schema,
        )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: executor.execute("map", prompt, resolved, payload={}, context={}),
                    range(8),
                )
            )

        self.assertEqual(results, [{"summary": "done"}] * 8)
        self.assertEqual(fake_client.files.upload_calls, ["/tmp/half.mkv"])
        self.assertEqual(len(fake_client.models.calls), 8)

    def test_gemini_executor_reports_usage_to_sink(self) -> None:
        usage = _FakeUsageMetadata(
            prompt_token_count=1200,
            candidates_token_count=7,
            thoughts_token_count=30,
            cached_content_token_count=None,
        )
        fake_client = _FakeClient('{"summary": "done"}', usage=usage)
        records = []
        executor = GeminiPromptExecutor(
            model="test-model",
            client=fake_client,
            types_module=_FakeTypes,
            poll_interval_seconds=0.0,
            usage_sink=records.append,
        )
        prompt = PromptSpec(parts=("Summarize this",), output_schema={"summary": "string"})
        resolved = ResolvedPrompt(parts=("Summarize this",), output_schema=prompt.output_schema)

        executor.execute("map", prompt, resolved, payload={}, context={})

        self.assertEqual(
            records,
            [
                {
                    "model": "test-model",
                    "op_type": "map",
                    "usage": {
                        "prompt_token_count": 1200,
                        "candidates_token_count": 7,
                        "thoughts_token_count": 30,
                    },
                }
            ],
        )

    def test_gemini_executor_reports_usage_before_rejecting_invalid_json(self) -> None:
        # The call is billed even though its output is unusable.
        fake_client = _FakeClient("not json", usage={"prompt_token_count": 50})
        records = []
        executor = GeminiPromptExecutor(
            client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0, usage_sink=records.append
        )
        prompt = PromptSpec(parts=("Is it?",), output_schema=None)
        resolved = ResolvedPrompt(parts=("Is it?",), output_schema=None)

        with self.assertRaisesRegex(MMDSValidationError, "invalid JSON"):
            executor.execute("filter", prompt, resolved, payload={}, context={})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["op_type"], "filter")
        self.assertEqual(records[0]["usage"], {"prompt_token_count": 50})

    def test_gemini_executor_reports_usage_before_rejecting_empty_response(self) -> None:
        fake_client = _FakeClient("", usage={"prompt_token_count": 50})
        records = []
        executor = GeminiPromptExecutor(
            client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0, usage_sink=records.append
        )
        prompt = PromptSpec(parts=("Is it?",), output_schema=None)
        resolved = ResolvedPrompt(parts=("Is it?",), output_schema=None)

        with self.assertRaisesRegex(MMDSValidationError, "empty response"):
            executor.execute("filter", prompt, resolved, payload={}, context={})

        self.assertEqual(len(records), 1)

    def test_gemini_executor_reports_missing_usage_as_none(self) -> None:
        fake_client = _FakeClient("true")
        records = []
        executor = GeminiPromptExecutor(
            client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0, usage_sink=records.append
        )
        prompt = PromptSpec(parts=("Is it?",), output_schema=None)
        resolved = ResolvedPrompt(parts=("Is it?",), output_schema=None)

        self.assertIs(executor.execute("filter", prompt, resolved, payload={}, context={}), True)
        self.assertIsNone(records[0]["usage"])

    def test_gemini_executor_rejects_unknown_usage_type(self) -> None:
        fake_client = _FakeClient("true", usage=12)
        executor = GeminiPromptExecutor(
            client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0, usage_sink=lambda record: None
        )
        prompt = PromptSpec(parts=("Is it?",), output_schema=None)
        resolved = ResolvedPrompt(parts=("Is it?",), output_schema=None)

        with self.assertRaisesRegex(MMDSValidationError, "usage metadata type"):
            executor.execute("filter", prompt, resolved, payload={}, context={})

    def test_gemini_executor_without_sink_ignores_usage(self) -> None:
        fake_client = _FakeClient("true", usage=12)
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(parts=("Is it?",), output_schema=None)
        resolved = ResolvedPrompt(parts=("Is it?",), output_schema=None)

        self.assertIs(executor.execute("filter", prompt, resolved, payload={}, context={}), True)

    def test_gemini_executor_accepts_source_video_uri(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(
            parts=("Watch ", {"type": "Video", "source": "https://youtube.com/watch?v=demo"}),
            output_schema=prompt.output_schema,
        )

        executor.execute("map", prompt, resolved, payload={}, context={})
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[1].file_data.file_uri, "https://youtube.com/watch?v=demo")

    def test_gemini_executor_translates_videoview_metadata(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(
            parts=(
                "Watch ",
                {
                    "type": "VideoView",
                    "source": "https://youtube.com/watch?v=demo",
                    "start": 10,
                    "end": 20.5,
                    "fps": 2,
                },
            ),
            output_schema=prompt.output_schema,
        )

        executor.execute("map", prompt, resolved, payload={}, context={})
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[1].file_data.file_uri, "https://youtube.com/watch?v=demo")
        self.assertEqual(
            content.parts[1].video_metadata.kwargs,
            {"start_offset": "10s", "end_offset": "20.5s", "fps": 2.0},
        )

    def test_gemini_executor_uploads_local_videoview_sources(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(
            parts=(
                "Watch ",
                {"type": "VideoView", "source": "file:///tmp/demo.mp4", "start": 3, "end": 9},
            ),
            output_schema=prompt.output_schema,
        )

        executor.execute("map", prompt, resolved, payload={}, context={})
        self.assertEqual(fake_client.files.upload_calls, ["/tmp/demo.mp4"])
        content = fake_client.models.calls[0]["contents"]
        self.assertEqual(content.parts[1].file_data.file_uri, "uploaded://video")
        self.assertEqual(
            content.parts[1].video_metadata.kwargs,
            {"start_offset": "3s", "end_offset": "9s"},
        )

    def test_gemini_executor_rejects_conflicting_videoview_offsets(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(
            parts=(
                "Watch ",
                {
                    "type": "VideoView",
                    "source": "https://youtube.com/watch?v=demo",
                    "start": 10,
                    "start_offset": "10s",
                },
            ),
            output_schema=prompt.output_schema,
        )

        with self.assertRaisesRegex(MMDSValidationError, "cannot include both 'start' and 'start_offset'"):
            executor.execute("map", prompt, resolved, payload={}, context={})

    def test_gemini_executor_logs_built_parts(self) -> None:
        fake_client = _FakeClient('{"summary": "done"}')
        executor = GeminiPromptExecutor(client=fake_client, types_module=_FakeTypes, poll_interval_seconds=0.0)
        prompt = PromptSpec(
            parts=("Watch ", Record["video"]),
            output_schema={"summary": "string"},
        )
        resolved = ResolvedPrompt(
            parts=("Watch ", {"type": "Video", "uri": "https://youtube.com/watch?v=demo"}),
            output_schema=prompt.output_schema,
        )

        with self.assertLogs("mmds.execution.llm.gemini", level="DEBUG") as captured:
            executor.execute("map", prompt, resolved, payload={}, context={})

        self.assertTrue(any("Sending Gemini prompt for map" in line for line in captured.output))
        self.assertTrue(any("Part 1: _FakePart(text='Watch '" in line for line in captured.output))
        self.assertTrue(
            any(
                "Part 2: _FakePart(text=None, inline_data=None, file_data=_FakeFileData(file_uri='https://youtube.com/watch?v=demo'" in line
                for line in captured.output
            )
        )


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

    def __repr__(self):
        return (
            "_FakePart("
            f"text={self.text!r}, "
            f"inline_data={self.inline_data!r}, "
            f"file_data={self.file_data!r}, "
            f"video_metadata={self.video_metadata!r})"
        )


class _FakeBlob:
    def __init__(self, data, mime_type):
        self.data = data
        self.mime_type = mime_type

    def __repr__(self):
        size = len(self.data) if isinstance(self.data, (bytes, bytearray)) else "unknown"
        return f"_FakeBlob(mime_type={self.mime_type!r}, bytes={size!r})"


class _FakeFileData:
    def __init__(self, file_uri, mime_type=None):
        self.file_uri = file_uri
        self.mime_type = mime_type

    def __repr__(self):
        return f"_FakeFileData(file_uri={self.file_uri!r}, mime_type={self.mime_type!r})"


class _FakeVideoMetadata:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return f"_FakeVideoMetadata({self.kwargs!r})"


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
    def __init__(self, upload_delay_seconds=0.0):
        self.upload_calls = []
        self.upload_delay_seconds = upload_delay_seconds

    def upload(self, file):
        self.upload_calls.append(file)
        time.sleep(self.upload_delay_seconds)
        return _FakeUploadedFile()

    def get(self, name):
        return _FakeUploadedFile()


class _FakeUsageMetadata:
    """Mimics the pydantic usage object returned by google-genai."""

    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self, *, mode, exclude_none):
        assert mode == "json" and exclude_none
        return {key: value for key, value in self.fields.items() if value is not None}


class _FakeModels:
    def __init__(self, text, usage=None):
        self.text = text
        self.usage = usage
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"text": self.text, "usage_metadata": self.usage})()


class _FakeClient:
    def __init__(self, text, usage=None, upload_delay_seconds=0.0):
        self.models = _FakeModels(text, usage)
        self.files = _FakeFiles(upload_delay_seconds)


def _remove_temp_file(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def _temp_rows_file(suffix: str):
    """Open a temp file that is deleted when the test process exits.

    The file must outlive this helper (tests pass its path to Input), so it is
    created with delete=False and removed at interpreter exit instead.
    """
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    atexit.register(_remove_temp_file, handle.name)
    return handle


def _write_json_rows(rows: list[dict]) -> str:
    handle = _temp_rows_file(".json")
    try:
        json.dump(rows, handle)
        return handle.name
    finally:
        handle.close()


def _write_jsonl_rows(rows: list[dict]) -> str:
    handle = _temp_rows_file(".jsonl")
    try:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
        return handle.name
    finally:
        handle.close()


if __name__ == "__main__":
    unittest.main()
