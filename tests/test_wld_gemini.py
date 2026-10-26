from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.query_types import wld_gemini as wg  # noqa: E402
from scripts.query_types.wld import Pair  # noqa: E402
from tests.wld_fixtures import FakeGemini  # noqa: E402


def _pair(film: str, animal: str = "bear") -> Pair:
    return Pair(film, animal, f"/videos/{film}.mp4", 60.0, 25.0, 1500, "absent", False, None)


class _ApiError(Exception):
    """Like google.genai's errors: an HTTP status ``code`` and a message."""

    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(f"{code} {message}")
        self.code = code


class IsTransientTests(unittest.TestCase):
    def test_server_errors_and_rate_limits_are_retried(self) -> None:
        for error in (_ApiError(500), _ApiError(503, "UNAVAILABLE"), _ApiError(429, "RESOURCE_EXHAUSTED requests")):
            with self.subTest(error=str(error)):
                self.assertTrue(wg.is_transient(error))

    def test_full_storage_quota_and_other_errors_are_not(self) -> None:
        for error in (
            _ApiError(429, "Quota exceeded for metric: generativelanguage.googleapis.com/file_storage_bytes"),
            _ApiError(400, "INVALID_ARGUMENT"),
            ValueError("expected a boolean"),
        ):
            with self.subTest(error=str(error)):
                self.assertFalse(wg.is_transient(error))


class RunPairsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "results"
        self.scratch = Path(self._tmp.name) / "scratch"
        self.pairs = [_pair("A"), _pair("B")]
        self.waited: list[float] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, task, **kwargs) -> None:
        wg.run_pairs(
            self.pairs, self.out, task, scratch=self.scratch,
            executor_factory=FakeGemini.factory(lambda payload: {}),
            waits=(1, 2, 4), sleep=self.waited.append, **kwargs,
        )

    @staticmethod
    def _calls(executor, n: int) -> None:
        for _ in range(n):
            executor.execute("map", None, None, {}, {})

    def test_each_pair_gets_its_result_usage_and_a_fresh_executor(self) -> None:
        executors = []

        def task(pair, executor, scratch):
            executors.append(executor)
            self._calls(executor, 2)
            return {"answer": pair.film_id == "A"}

        self._run(task)
        result = json.loads((self.out / "A__bear.json").read_text())
        self.assertEqual((result["answer"], result["attempts"], len(result["usage"])), (True, 1, 2))
        self.assertEqual(result["model"], wg.MODEL)
        self.assertIsNot(executors[0], executors[1])
        self.assertTrue((self.out / "_run_info.jsonl").exists())

    def test_finished_pairs_are_not_run_again(self) -> None:
        ran = []
        self._run(lambda pair, executor, scratch: ran.append(pair.film_id) or {})
        self._run(lambda pair, executor, scratch: ran.append(pair.film_id) or {})
        self.assertEqual(ran, ["A", "B"])

    def test_temporary_error_is_retried_after_a_wait_and_billed_tokens_are_kept(self) -> None:
        tries = {"A": 0}

        def task(pair, executor, scratch):
            self._calls(executor, 1)
            if pair.film_id == "A" and tries["A"] < 2:
                tries["A"] += 1
                raise _ApiError(503, "UNAVAILABLE")
            return {"ok": True}

        self._run(task)
        self.assertEqual(self.waited, [1, 2])
        result = json.loads((self.out / "A__bear.json").read_text())
        self.assertEqual((result["attempts"], len(result["usage"])), (3, 1))  # only the successful try's usage
        failed = [json.loads(line) for line in (self.out / "_failed_usage.jsonl").read_text().splitlines()]
        self.assertEqual([f["attempt"] for f in failed], [1, 2])
        self.assertTrue(all("503" in f["error"] for f in failed))

    def test_gives_up_after_the_last_wait(self) -> None:
        def task(pair, executor, scratch):
            raise _ApiError(500, "INTERNAL")

        with self.assertRaises(_ApiError):
            self._run(task)
        self.assertEqual(self.waited, [1, 2, 4])
        self.assertFalse((self.out / "A__bear.json").exists())

    def test_other_errors_stop_at_once(self) -> None:
        for error in (_ApiError(429, "file_storage_bytes limit exceeded"), ValueError("bad JSON")):
            with self.subTest(error=str(error)):
                self.waited.clear()

                def task(pair, executor, scratch, error=error):
                    raise error

                with self.assertRaises(type(error)):
                    self._run(task)
                self.assertEqual(self.waited, [])

    def test_retry_restarts_the_pair_so_time_excludes_the_wait(self) -> None:
        starts = []

        def task(pair, executor, scratch):
            starts.append(pair.film_id)
            if len(starts) == 1:
                raise _ApiError(503)
            return {}

        self._run(task)
        self.assertEqual(starts, ["A", "A", "B"])


class AskRowTests(unittest.TestCase):
    def test_runs_a_one_row_plan_and_returns_its_row(self) -> None:
        from mmds import Input, Map, Record

        def plan(path):
            return Map(Input(path), ["Say yes about ", Record["animal"]], schema={"ok": "boolean"})

        executor = FakeGemini(lambda record: None, lambda payload: {"ok": payload["animal"] == "bear"})
        with tempfile.TemporaryDirectory() as tmp:
            row = wg.ask_row(plan, {"pair_id": "A|bear", "animal": "bear"}, Path(tmp) / "row.jsonl", executor)
        self.assertTrue(row["ok"])

    def test_more_than_one_row_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(wg, "execute", return_value=[{}, {}]):
            with self.assertRaisesRegex(ValueError, "expected one answer, got 2"):
                wg.ask_row(lambda path: None, {"pair_id": "A|bear"}, Path(tmp) / "row.jsonl", executor=None)


if __name__ == "__main__":
    unittest.main()
