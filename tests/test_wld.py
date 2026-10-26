from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mmds.optimizers.rewriter import directives  # noqa: E402
from scripts.query_types import wld  # noqa: E402
from scripts.query_types import wld_localization, wld_presence  # noqa: E402
from tests.wld_fixtures import has_ffmpeg, make_video  # noqa: E402


def _annotation(categories: list[str], frames: list[list[int]], film_id: str = "film") -> dict:
    """Build a WLD annotation where ``frames[i]`` lists the class ids boxed in frame ``i``."""
    return {
        "vid": film_id,
        "frame_num": len(frames),
        "categories": categories,
        "bboxes": [[[0, 0, 10, 10, class_id] for class_id in ids] for ids in frames],
    }


class ParseLabelsTests(unittest.TestCase):
    def test_class_ids_count_from_one(self) -> None:
        # In the WLD leopard film, categories are listed alphabetically and id 1
        # is the first name. Reading ids from 0 would relabel leopard as lion.
        labels = wld.parse_labels(_annotation(["leopard", "lion"], [[1], [2], []]))
        self.assertEqual(wld.class_frames(labels, ["leopard"]), [0])
        self.assertEqual(wld.class_frames(labels, ["lion"]), [1])

    def test_out_of_range_ids_raise(self) -> None:
        for bad in (0, 3, -1, True, 1.0):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "outside 1..2"):
                wld.parse_labels(_annotation(["leopard", "lion"], [[bad]]))

    def test_malformed_box_raises(self) -> None:
        annotation = _annotation(["dog"], [[1]])
        annotation["bboxes"][0][0] = [0, 0, 10, 10]
        with self.assertRaisesRegex(ValueError, "expected"):
            wld.parse_labels(annotation)

    def test_frame_count_must_match_the_boxed_frames(self) -> None:
        annotation = _annotation(["dog"], [[1], []])
        annotation["frame_num"] = 3
        with self.assertRaisesRegex(ValueError, "frame_num is 3"):
            wld.parse_labels(annotation)

    def test_bear_covers_every_kind_of_bear_but_not_koala(self) -> None:
        labels = wld.parse_labels(_annotation(["polar bear", "koala", "bear", "brown bear"], [[1], [2], [3], [4], [2, 3]]))
        self.assertEqual(wld.class_frames(labels, wld.QUERY_CLASSES["bear"]), [0, 2, 3, 4])

    def test_label_removal_removes_only_the_wrong_category(self) -> None:
        labels = wld.parse_labels(_annotation(["koala", "bear"], [[1, 2], [2], [1]], film_id="aN3jEIAXvbk"))
        fixed = wld.fix_labels(labels)
        self.assertEqual(wld.class_frames(fixed, wld.QUERY_CLASSES["bear"]), [])
        self.assertEqual(wld.class_frames(fixed, ["koala"]), [0, 2])

    def test_label_removal_for_a_missing_category_fails_loudly(self) -> None:
        labels = wld.parse_labels(_annotation(["koala"], [[1]], film_id="aN3jEIAXvbk"))
        with self.assertRaisesRegex(ValueError, "not one of its categories"):
            wld.fix_labels(labels)

    def test_films_without_fixes_are_unchanged(self) -> None:
        labels = wld.parse_labels(_annotation(["bear"], [[1]], film_id="other"))
        self.assertEqual(wld.fix_labels(labels), labels)

    def test_label_addition_adds_a_new_category_to_its_frames_only(self) -> None:
        labels = wld.parse_labels(_annotation(["tiger"], [[1], [1], [], [1]], film_id="f"))
        with mock.patch.dict(wld.LABEL_ADDITIONS, {"f": (("leopard", 1, 2),)}):
            fixed = wld.fix_labels(labels)
        self.assertEqual(fixed.categories, ("tiger", "leopard"))
        self.assertEqual(wld.class_frames(fixed, ["leopard"]), [1, 2])
        self.assertEqual(wld.class_frames(fixed, ["tiger"]), [0, 1, 3])

    def test_label_addition_reuses_an_existing_category(self) -> None:
        labels = wld.parse_labels(_annotation(["leopard", "tiger"], [[1], [2], [2]], film_id="f"))
        with mock.patch.dict(wld.LABEL_ADDITIONS, {"f": (("leopard", 2, 2),)}):
            fixed = wld.fix_labels(labels)
        self.assertEqual(fixed.categories, ("leopard", "tiger"))
        self.assertEqual(wld.class_frames(fixed, ["leopard"]), [0, 2])

    def test_label_addition_fails_loudly_outside_the_film_or_on_labeled_frames(self) -> None:
        labels = wld.parse_labels(_annotation(["leopard"], [[], [1], []], film_id="f"))
        for bad in ((-1, 0), (2, 3), (2, 1)):
            with self.subTest(frames=bad), mock.patch.dict(wld.LABEL_ADDITIONS, {"f": (("leopard", *bad),)}):
                with self.assertRaisesRegex(ValueError, "outside 0-2"):
                    wld.fix_labels(labels)
        with mock.patch.dict(wld.LABEL_ADDITIONS, {"f": (("leopard", 0, 1),)}):
            with self.assertRaisesRegex(ValueError, "already labeled in frames 0-1"):
                wld.fix_labels(labels)

    def test_the_unlabeled_leopard_becomes_present_for_its_shot(self) -> None:
        # UsnHCvC92mM has no leopard category; frames 8597-8705 hold only a tiger.
        frames = [[1] if 8597 <= f <= 8705 else [] for f in range(9000)]
        labels = wld.fix_labels(wld.parse_labels(_annotation(["tiger"], frames, film_id="UsnHCvC92mM")))
        leopard = wld.film_truth(labels, 25.0)["leopard"]
        self.assertEqual(leopard["presence"], "present")
        self.assertEqual(len(leopard["labeled"]), 1)
        self.assertAlmostEqual(leopard["labeled"][0][0], 343.88)
        self.assertAlmostEqual(leopard["labeled"][0][1], 348.24)
        self.assertEqual(wld.film_truth(labels, 25.0)["tiger"]["labeled"], leopard["labeled"])

    def test_query_classes_are_not_found_in_unrelated_films(self) -> None:
        labels = wld.parse_labels(_annotation(["dolphin"], [[1]]))
        self.assertEqual(wld.class_frames(labels, wld.QUERY_CLASSES["bear"]), [])


class IntervalTruthTests(unittest.TestCase):
    def test_frames_become_seconds_with_the_film_frame_rate(self) -> None:
        self.assertEqual(wld.frames_to_intervals([0, 1, 2], fps=25, gap=0), [(0.0, 0.12)])
        self.assertEqual(wld.frames_to_intervals([50], fps=25, gap=0), [(2.0, 2.04)])

    def test_gaps_up_to_one_second_are_merged(self) -> None:
        # Frame 0 ends at 0.04 s; frame 26 starts at 1.04 s: a gap of exactly 1 s.
        self.assertEqual(len(wld.frames_to_intervals([0, 26], fps=25)), 1)
        self.assertEqual(len(wld.frames_to_intervals([0, 27], fps=25)), 2)

    def test_presence_needs_one_second_of_visible_frames(self) -> None:
        self.assertEqual(wld.presence(list(range(25)), fps=25), "present")
        self.assertEqual(wld.presence(list(range(24)), fps=25), "brief")
        self.assertEqual(wld.presence([], fps=25), "absent")

    def test_presence_counts_visible_frames_not_merged_gaps(self) -> None:
        # Two single frames 0.8 s apart merge into one interval, but only
        # 0.08 s is visible: brief, not present.
        self.assertEqual(wld.presence([0, 20], fps=25), "brief")

    def test_short_stretches_are_ignored_time(self) -> None:
        long_run = list(range(0, 50))  # 2 s
        short_run = list(range(200, 210))  # 0.4 s, far from the long run
        labeled, ignored = wld.localization_truth(long_run + short_run, fps=25)
        self.assertEqual(labeled, [(0.0, 2.0)])
        self.assertEqual(ignored, [(8.0, 8.4)])

    def test_invalid_fps_raises(self) -> None:
        with self.assertRaises(ValueError):
            wld.frames_to_intervals([0], fps=0)

    def test_film_truth_covers_every_query(self) -> None:
        frames = [[]] * 25 + [[1]] * 30
        labels = wld.parse_labels(_annotation(["black bear"], frames))
        truth = wld.film_truth(labels, fps=25)
        self.assertEqual(set(truth), set(wld.QUERY_CLASSES))
        self.assertEqual(truth["bear"]["presence"], "present")
        self.assertAlmostEqual(truth["bear"]["first_seconds"], 1.0)
        self.assertAlmostEqual(truth["bear"]["visible_seconds"], 1.2)
        self.assertEqual(truth["dog"]["presence"], "absent")
        self.assertIsNone(truth["dog"]["first_seconds"])
        self.assertFalse(truth["dog"]["presence_any_frame"])


class ParseSrtTests(unittest.TestCase):
    def test_wld_style_file_with_byte_order_mark(self) -> None:
        text = (
            "﻿1\n00:00:08,730 --> 00:00:19,580\n"
            "It was here, at this moment, she had understood\n\n"
            "2\n00:00:19,590 --> 00:00:24,620\nsolve the puzzle.\n"
        )
        self.assertEqual(
            wld.parse_srt(text),
            [
                {"start": 8.73, "end": 19.58, "text": "It was here, at this moment, she had understood"},
                {"start": 19.59, "end": 24.62, "text": "solve the puzzle."},
            ],
        )

    def test_crlf_multi_line_text_missing_numbers_hours_and_dots(self) -> None:
        text = "01:02:03.500 --> 01:02:04,000\r\nfirst line\r\nsecond line\r\n\r\n\r\n"
        cue = wld.parse_srt(text)[0]
        self.assertAlmostEqual(cue["start"], 3723.5)
        self.assertEqual(cue["text"], "first line second line")

    def test_cue_without_text_is_skipped(self) -> None:
        text = "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nhello\n"
        self.assertEqual([cue["text"] for cue in wld.parse_srt(text)], ["hello"])

    def test_empty_file(self) -> None:
        self.assertEqual(wld.parse_srt("﻿\n\n"), [])

    def test_bad_cues_raise(self) -> None:
        for text in ("1\nnot a time\nhello\n", "1\n00:00:05,000 --> 00:00:04,000\nhello\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                wld.parse_srt(text)

    def test_every_wld_subtitle_file_parses_after_known_fixes(self) -> None:
        subtitles = ROOT / "data" / "wld" / "subtitles"
        if not subtitles.exists():
            self.skipTest("WLD subtitles are not downloaded")
        for path in sorted(subtitles.glob("*.srt")):
            with self.subTest(film=path.stem):
                cues = wld.load_subtitles(path.stem, ROOT / "data" / "wld")
                self.assertGreater(len(cues), 10)
                self.assertTrue(all(cue["end"] >= cue["start"] for cue in cues))
                starts = [cue["start"] for cue in cues]
                self.assertEqual(starts, sorted(starts))

    def test_known_typo_is_fixed_to_the_neighbouring_time(self) -> None:
        text = (
            "247\n00:18:51,929 --> 00:18:54,599\nbirds or mammals\n\n"
            "248\n018:54.600 --> 00:19:00.269\nmarsupials.\n"
        )
        cues = wld.parse_srt(wld.fix_subtitles("aN3jEIAXvbk", text))
        self.assertAlmostEqual(cues[1]["start"], 18 * 60 + 54.6)

    def test_fix_that_no_longer_matches_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "found it 0 times"):
            wld.fix_subtitles("aN3jEIAXvbk", "1\n00:00:01,000 --> 00:00:02,000\nhello\n")
        self.assertEqual(wld.fix_subtitles("other", "unchanged"), "unchanged")


class PairVideoTests(unittest.TestCase):
    PAIR = wld.Pair("QkJYSeYX8oM", "bear", "data/wld/videos_360p/QkJYSeYX8oM.mp4", 100.0, 25.0, 2500, "present", True, 1.0)

    def test_whole_film_and_window_come_from_youtube(self) -> None:
        self.assertEqual(self.PAIR.video(), {"type": "video", "uri": "https://www.youtube.com/watch?v=QkJYSeYX8oM"})
        self.assertEqual(
            self.PAIR.video(540, 570),
            {"type": "video", "uri": "https://www.youtube.com/watch?v=QkJYSeYX8oM", "start": 540.0, "end": 570.0},
        )

    def test_half_a_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.PAIR.video(start=5.0)


class CheckFilmTests(unittest.TestCase):
    PROBE = {"frames": 100, "fps": 25.0, "width": 640, "height": 360, "duration": 4.0}

    def test_frame_counts_within_two_are_accepted(self) -> None:
        for label_frames in (98, 100, 102):
            with self.subTest(label_frames=label_frames):
                entry = wld.check_film("a", self.PROBE, label_frames)
                self.assertEqual(entry["status"], "ok")
                self.assertEqual(entry["frame_diff"], 100 - label_frames)

    def test_larger_frame_count_differences_are_rejected_with_a_reason(self) -> None:
        for label_frames in (97, 103):
            with self.subTest(label_frames=label_frames):
                entry = wld.check_film("a", self.PROBE, label_frames)
                self.assertEqual(entry["status"], "frame_mismatch")
                self.assertIn(f"labels have {label_frames}", entry["reason"])


class RealAnnotationTests(unittest.TestCase):
    def test_every_wld_annotation_parses_and_names_its_classes(self) -> None:
        annotations = ROOT / "data" / "wld" / "annotations"
        if not annotations.exists():
            self.skipTest("WLD annotations are not downloaded")
        bear_films = []
        for path in sorted(annotations.glob("*.json")):
            labels = wld.load_labels(path.stem, ROOT / "data" / "wld")
            if wld.class_frames(labels, wld.QUERY_CLASSES["bear"]):
                bear_films.append(path.stem)
        # Six of the fifteen films have bear labels; one (aN3jEIAXvbk) is a
        # labeling error removed by LABEL_REMOVALS, leaving five.
        self.assertEqual(len(bear_films), 5)
        self.assertNotIn("aN3jEIAXvbk", bear_films)
        # The leopard WLD missed is added for exactly its shot.
        usnh = wld.load_labels("UsnHCvC92mM", ROOT / "data" / "wld")
        self.assertEqual(wld.class_frames(usnh, ["leopard"]), list(range(8597, 8706)))


def _write_film_files(root: Path, film_id: str, frames: list[list[int]], height: int = 24) -> None:
    """An annotation (one 'bear' category) and the filelist entry for one film."""
    (root / "annotations").mkdir(exist_ok=True)
    annotation = {**_annotation(["bear"], frames, film_id), "height": height}
    (root / "annotations" / f"{film_id}.json").write_text(json.dumps(annotation))
    with (root / "filelist.txt").open("a") as handle:
        handle.write(film_id + "\n")


@unittest.skipUnless(has_ffmpeg(), "needs ffmpeg")
class PrepareTests(unittest.TestCase):
    """Three films: one ready, one whose frame count is off by more than 2, one never downloaded."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "videos").mkdir()
        # 2 s at 5 fps = 10 frames; the bear is boxed in the last 6 (1.2 s).
        _write_film_files(self.root, "ok", [[]] * 4 + [[1]] * 6)
        make_video(self.root / "videos" / "ok.mp4", seconds=2.0)
        _write_film_files(self.root, "off", [[]] * 20)
        make_video(self.root / "videos" / "off.mp4", seconds=2.0)
        _write_film_files(self.root, "gone", [[]] * 10)
        (self.root / "downloads.json").write_text(json.dumps(
            {"yt_dlp_version": "2026.10.01", "films": [{"film_id": "ok", "status": "downloaded", "formats": "137+140"}]}
        ))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_films_are_checked_and_only_ready_films_get_truth(self) -> None:
        films = {film["film_id"]: film for film in wld.prepare(self.root, height=24)}
        self.assertEqual(films["ok"]["status"], "ok")
        self.assertEqual((films["ok"]["frames"], films["ok"]["frame_diff"], films["ok"]["height"]), (10, 0, 24))
        self.assertEqual(films["ok"]["download"], {"film_id": "ok", "status": "downloaded", "formats": "137+140",
                                                   "yt_dlp_version": "2026.10.01"})
        self.assertEqual(films["off"]["status"], "frame_mismatch")
        self.assertIn("labels have 20", films["off"]["reason"])
        self.assertEqual(films["gone"]["status"], "missing")
        truth = json.loads((self.root / "truth.json").read_text())
        self.assertEqual(set(truth), {"ok"})
        self.assertEqual(truth["ok"]["bear"]["presence"], "present")
        self.assertAlmostEqual(truth["ok"]["bear"]["first_seconds"], 0.8)

    def test_second_prepare_reuses_the_encoded_film(self) -> None:
        wld.prepare(self.root, height=24)
        again = {film["film_id"]: film for film in wld.prepare(self.root, height=24)}
        self.assertIsNone(again["ok"]["encode_seconds"])


class DownloadTests(unittest.TestCase):
    """yt-dlp stubbed out: film X downloads, Y is gone from YouTube, Z is already on disk."""

    def test_download_records_what_happened(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for film_id, height in (("X", 480), ("Y", 720), ("Z", 360)):
                _write_film_files(root, film_id, [[]], height=height)
            (root / "videos").mkdir()
            (root / "videos" / "Z.mp4").write_bytes(b"")
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if command[-1] == "--version":
                    return subprocess.CompletedProcess(command, 0, "2026.10.01\n", "")
                if command[-1].endswith("v=X"):
                    (root / "videos" / "X.mp4").write_bytes(b"video")
                    return subprocess.CompletedProcess(command, 0, "137+140-2\n", "")
                return subprocess.CompletedProcess(command, 1, "", "ERROR: [youtube] Y: Video unavailable\n")

            records = {r["film_id"]: r for r in wld.download(root, yt_dlp=("yt-dlp",), run=run)}
            saved = json.loads((root / "downloads.json").read_text())

        self.assertEqual(records["X"], {"film_id": "X", "status": "downloaded", "formats": "137+140-2"})
        self.assertEqual(records["Y"]["status"], "unavailable")
        self.assertIn("Video unavailable", records["Y"]["reason"])
        self.assertEqual(records["Z"]["status"], "present")
        self.assertEqual(saved["yt_dlp_version"], "2026.10.01")
        # Each film is fetched at its annotation's resolution; Z is never fetched.
        fetched = [c for c in commands if c[-1] != "--version"]
        self.assertEqual(len(fetched), 2)
        self.assertIn("bv*[height<=480][vcodec^=avc1]", fetched[0][fetched[0].index("-f") + 1])

    def test_command_asks_for_h264_english_audio_and_mp4(self) -> None:
        command = wld.download_command("abc", 720, Path("videos"), yt_dlp=("yt-dlp",))
        formats = command[command.index("-f") + 1]
        self.assertTrue(formats.startswith("bv*[height<=720][vcodec^=avc1]+ba[language^=en]"))
        self.assertEqual(command[command.index("--merge-output-format") + 1], "mp4")
        self.assertEqual(command[-1], "https://www.youtube.com/watch?v=abc")


class CliTests(unittest.TestCase):
    def test_download_and_prepare_commands(self) -> None:
        with mock.patch.object(wld, "download", return_value=[]) as download, \
                mock.patch.object(wld, "prepare", return_value=[]) as prepare:
            wld.main(["download"])
            wld.main(["prepare"])
        download.assert_called_once()
        prepare.assert_called_once()
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            wld.main(["unpack"])


class ReadmeTests(unittest.TestCase):
    """WLD_README's table of plans and the rewrite directives behind them."""

    README = ROOT / "scripts" / "query_types" / "WLD_README.md"
    SECTION = "## Plans and the rewrite directives behind them"

    def _rows(self) -> list[list[str]]:
        section = self.README.read_text(encoding="utf-8").split(self.SECTION, 1)[1].split("\n## ", 1)[0]
        table = section.split("| Plan | Directive |", 1)[1].splitlines()[2:]  # skip the header's rest and |---|
        return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in table if line.startswith("|")]

    def test_every_named_directive_exists(self) -> None:
        named = {name for row in self._rows() for name in re.findall(r"`(\w+)`", row[1])}
        self.assertTrue(named)
        self.assertLessEqual(named, set(directives.__all__))

    def test_every_plan_of_both_experiments_is_listed(self) -> None:
        listed = {name for row in self._rows() for name in re.findall(r"`(\w+)`", row[0])}
        plans = {*wld_presence.GEMINI_PLANS, "yoloe_full", "yoloe_early", *wld_localization.PLANS}
        self.assertEqual(plans - listed, set())


if __name__ == "__main__":
    unittest.main()
