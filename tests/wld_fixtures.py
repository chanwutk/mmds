"""A tiny fake WLD dataset, and fake Gemini and YOLOE, for tests of the WLD experiments.

Two 300-second films at 5 fps, ``A`` and ``B``. A bear is on screen in film
A from 130 s to 170 s, and the narration mentions it at 125-135 s. The fake
YOLOE sees the bear from 130 s to 200 s, so it overshoots by 30 s. A dog
appears in film B for half a second, which is a "brief" appearance. Every
other (film, animal) pair is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mmds import execute as real_execute
from scripts.query_types.common import write_json_atomic
from scripts.query_types.wld import QUERY_CLASSES, Pair
from scripts.query_types.wld_gemini import MODEL
from scripts.query_types.wld_yoloe import VARIANTS, scan_dir, scan_units

FILMS = ("A", "B")
DURATION = 300.0
FPS = 5.0
BEAR_ON_SCREEN = (130.0, 170.0)
YOLOE_SEES_BEAR = (130.0, 200.0)
SUBTITLES = {
    "A": "1\n00:00:05,000 --> 00:00:09,000\nThe valley wakes up.\n\n"
         "2\n00:02:05,000 --> 00:02:15,000\nA bear walks out of the forest.\n",
    "B": "﻿1\n00:00:05,000 --> 00:00:09,000\nThe river is quiet today.\n",
}


def has_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_video(path: Path, seconds: float = DURATION, fps: float = FPS) -> None:
    """A tiny test-pattern video with a silent audio track."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size=32x24:rate={fps:g}:duration={seconds:g}",
         "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", f"{seconds:g}", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def make_fake_wld(root: Path) -> None:
    """Write films, films.json, truth.json and subtitles under ``root``.

    Like the real layout, each film is under ``videos_360p/`` (the prepared
    copy) and ``videos/`` (the original download).
    """
    films = []
    (root / "videos_360p").mkdir()
    (root / "videos").mkdir()
    for film in FILMS:
        path = root / "videos_360p" / f"{film}.mp4"
        make_video(path)
        shutil.copyfile(path, root / "videos" / f"{film}.mp4")
        films.append({"film_id": film, "status": "ok", "path": str(path), "duration": DURATION, "fps": FPS,
                      "frames": int(DURATION * FPS)})
    absent = {"presence": "absent", "presence_any_frame": False, "visible_seconds": 0.0, "first_seconds": None,
              "labeled": [], "ignored": []}
    truth = {film: {animal: dict(absent) for animal in QUERY_CLASSES} for film in FILMS}
    truth["A"]["bear"] = {"presence": "present", "presence_any_frame": True, "visible_seconds": 40.0,
                          "first_seconds": BEAR_ON_SCREEN[0], "labeled": [list(BEAR_ON_SCREEN)], "ignored": []}
    truth["B"]["dog"] = {"presence": "brief", "presence_any_frame": True, "visible_seconds": 0.5,
                         "first_seconds": 50.0, "labeled": [], "ignored": [[50.0, 50.5]]}
    (root / "films.json").write_text(json.dumps(films))
    (root / "truth.json").write_text(json.dumps(truth))
    (root / "subtitles").mkdir()
    for film, text in SUBTITLES.items():
        (root / "subtitles" / f"{film}.srt").write_text(text, encoding="utf-8")


def bear_boxes(start: float, end: float, stride: int, conf: float = 0.8) -> list[dict]:
    """Fake YOLOE boxes for the bear on the frames sampled from ``[start, end)``."""
    frames = range(int(round(start * FPS)), int(round(end * FPS)), stride)
    seen_from, seen_to = YOLOE_SEES_BEAR
    return [
        {"frame_idx": f, "bbox": [0, 0, 1, 1], "confidence": conf}
        for f in frames
        if seen_from <= f / FPS < seen_to
    ]


def fake_yoloe(boxes: Callable[[str, str, float, float, int], list[dict]] | None = None):
    """A stand-in for ``mmds.execute`` that answers Detect plans with fake boxes.

    ``boxes(film, animal, start, end, stride)`` gives one animal's boxes;
    by default only the bear in film A. Detect sees only the names it was
    prompted with, and ``stop_after_n=1`` keeps just the first box at the
    plan's confidence. Every other plan runs for real.
    """
    boxes = boxes or (lambda film, animal, start, end, stride: bear_boxes(start, end, stride)
                      if (film, animal) == ("A", "bear") else [])

    def execute(plan, executor=None):
        if plan.kind != "detect":
            return real_execute(plan, executor)
        rows = [json.loads(line) for line in Path(plan.source.input_path).read_text().splitlines()]
        out = []
        for row in rows:
            film = row["pair_id"].split("|")[0]
            video = row["video"]
            start, end = video.get("start", 0.0), video.get("end", DURATION)
            detections = []
            for animal in plan.spec.classes:
                found = [b for b in boxes(film, animal, start, end, plan.spec.frame_stride)
                         if plan.spec.conf is None or b["confidence"] >= plan.spec.conf]
                if plan.spec.stop_after_n == 1:
                    found = found[:1]
                if found:
                    detections.append({"type": animal, "bboxes": found})
            out.append({**row, "detections": detections})
        return out

    return execute


class FakeGemini:
    """A Gemini executor whose answer comes from ``answer(payload)``.

    Every call reports 50 text input tokens of the experiments' model to the
    usage sink, so tests can check what each plan pays in tokens and dollars.
    """

    def __init__(self, sink, answer: Callable[[dict], Any]):
        self.sink = sink
        self.answer = answer

    @classmethod
    def factory(cls, answer: Callable[[dict], Any]):
        return lambda sink: cls(sink, answer)

    def execute(self, op_type, prompt, resolved_prompt, payload, context):
        usage = {"prompt_token_count": 50, "prompt_tokens_details": [{"modality": "TEXT", "token_count": 50}]}
        self.sink({"model": MODEL, "model_version": "fake-1", "op_type": op_type, "usage": usage})
        return self.answer(payload)


def write_scans(
    out_root: Path,
    variant_name: str,
    pairs: list[Pair],
    *,
    seconds: float = 30.0,
    stride: int = 1,
    boxes: Callable[[str, str], list[dict]] | None = None,
) -> None:
    """Scan files as ``wld_yoloe.run_scan`` writes them; by default only film A's bear is seen."""
    boxes = boxes or (lambda film, animal: bear_boxes(0.0, DURATION, stride) if (film, animal) == ("A", "bear") else [])
    result_dir = scan_dir(variant_name, out_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, covered in scan_units(VARIANTS[variant_name], pairs):
        write_json_atomic(result_dir / f"{name}.json", {
            "variant": variant_name, "model": VARIANTS[variant_name].model, "video": covered[0].path,
            "stride": stride, "sampled_frames": int(DURATION * FPS) // stride, "seconds": seconds,
            "device": "cpu", "gpu": None,
            "frame_max_conf": {
                p.animal: {str(b["frame_idx"]): b["confidence"] for b in boxes(p.film_id, p.animal)} for p in covered
            },
        })
