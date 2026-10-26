# WLD experiments (camera-ready)

Two experiments on the WildLife Documentary dataset (WLD; Chen et al., CVPR 2017;
[github.com/hellock/WLD](https://github.com/hellock/WLD)): narrated wildlife films where every
animal in every frame is boxed. Each question is about one **pair**, a (film, animal): 5 films still
on YouTube × 9 animals = 45 pairs.

| Experiment | Question | Script |
|---|---|---|
| 1. Presence | "Is a {animal} visible anywhere in this film?" (yes/no) | `wld_presence.py` |
| 2. Localization | "When is a {animal} on screen?" (time intervals) | `wld_localization.py` |

## Videos

The 5 films, 1 h 54 min in total. Gemini reads them from YouTube. YOLOE reads copies downloaded at
the resolution the boxes were labeled at (`large` variants) or a 360p re-encode of those (`small`
variants). Times after each animal are its labeled seconds on screen.

| Film | YouTube title | Length | Labeled at | Query animals visible |
|---|---|---|---|---|
| `UsnHCvC92mM` | Indonesia's Tiger Sanctuary Is A Squalid Hell Hole | 26:37 | 720p | tiger 402 s, lion 22, elephant 22, bear 19, deer 15, monkey 13, leopard 4 |
| `aN3jEIAXvbk` | Outback Wildlife Rescue Episode 3 | 23:42 | 480p | none of the 9 (WLD labels koala 298 s, python 239, kite 214, cock 15, and four briefer animals) |
| `3pSkqwX2FlU` | Lakadema The Leopard 2OF4 HD 720p | 9:16 | 720p | leopard 407 s, lion 49, monkey 41, elephant 5 |
| `QkJYSeYX8oM` | Amur tiger in third millennium. English edition | 38:05 | 480p | tiger 778 s, deer 98, boar 48, bear 17, dog 7 |
| `NyvKQVIJdes` | Swimming With Dolphins, The Dark Secrets Surrounding Dolphins | 16:41 | 1080p | none of the 9 (WLD labels dolphin 205 s) |

The 9 query animals are bear (WLD's bear, polar bear, brown bear and black bear together), elephant,
dog, deer, tiger, leopard, monkey, lion and boar: 16 present pairs and 29 absent ones, after the
label fixes below. `aN3jEIAXvbk` and `NyvKQVIJdes` have animals, but none of these 9, so their 18
pairs test false positives, such as calling a koala a bear.

## Code

| Module | What it holds |
|---|---|
| `wld.py` | Dataset: download, prepare, labels and their fixes, subtitles, pairs |
| `wld_yoloe.py` | YOLOE: the four scan variants, scans, the detection rule, hit windows, early-stop timing check |
| `wld_gemini.py` | The per-pair Gemini loop both experiments use (fresh executor, usage, retries, resume) |
| `common.py` | Shared helpers: video cutting, intervals and scores, dollar costs, summaries, provenance |
| `run_yoloe_gpu.sh` | All YOLOE work, one process at a time on one GPU |

## Plans and the rewrite directives behind them

Plan names are built from these parts:

| Part | Means |
|---|---|
| `mllm` | Gemini, the multimodal LLM |
| `yoloe` | the YOLOE detector |
| `transcript` | the film's narration (its subtitles) |
| `full`, `video` | the whole film |
| `early` | stops at the first "yes" |

So `transcript_yoloe_mllm` is narration first, then YOLOE, then Gemini.

The experiments build each plan directly in Python, one pair at a time, so they can time and price
every stage. Most plans have the shape a rewrite directive in `src/mmds/optimizers/rewriter/directives/`
produces from the naive query (see DESIGN.md, "Implemented Video Rewrite Directives"):

| Plan | Directive | How the experiment differs |
|---|---|---|
| `mllm_full` (Exp 1), `mllm_video` (Exp 2) | – | Not a rewrite: the naive query, Gemini on the whole film, which every directive starts from |
| `transcript_only` | `ModalitySubstitution` | – |
| `transcript_mllm` | `PerViewTemporalPushdown` | Narration windows are padded by ±10 s, and windows over ~2 min are split into clips |
| `yoloe_mllm` (Exp 2) | `DetectedFrameWindowBeforeMap` | Gemini is asked about each window separately and its times are moved back to film time; the directive asks once about all windows of a row |
| `yoloe_mllm` (Exp 1) | `DetectGateBeforeMap`, then `DetectedFrameWindowBeforeMap` | A pair with no hit is answered "no" without Gemini (the gate); otherwise windows are asked most confident first, stopping at the first "yes" |
| `transcript_yoloe_mllm` | `PerViewTemporalPushdown`, then `DetectedFrameWindowBeforeMap` | The two rewrites chained; whether the rewrite engine applies the second to the first's output is not tested here |
| `mllm_early`, `yoloe_full`, `yoloe_early`, `yoloe_video`, `transcript_yoloe` | none yet | `mllm_early` asks Gemini about 2-minute windows in order. The YOLOE plans replace the video model with `Detect`: over the whole film, or inside the narration windows for `transcript_yoloe`. `yoloe_early` stops with `Detect(stop_after_n=1)` |
| `transcript_windows` | – | Not a rewrite: the narration windows scored as an answer, to show the first stage of the narration chains |

## Setup

- Clone WLD's `annotations/`, `subtitles/` and `filelist.txt` into `data/wld/` (git-ignored).
- `ffmpeg`/`ffprobe` and `uv` on the PATH (yt-dlp runs as `uvx yt-dlp@latest`).
- YOLOE weights `yoloe-11s-seg.pt` and `yoloe-11l-seg.pt` and the text encoder `mobileclip_blt.ts` in
  the repo root (ultralytics downloads missing ones).
- Gemini: `GEMINI_API_KEY` in the environment (e.g. `set -a; source .env; set +a`). Keep the key off
  shared machines; Gemini steps run where the key is.

Every command runs from the repo root with `PYTHONPATH=src:.`. Every run step skips work already done,
so a stopped step resumes when run again. Times are wall-clock per repeat, as measured in repeats 1–3;
tokens are Gemini input tokens, nearly all of them video.

## Steps

| # | Where | Command | Output | Time / cost |
|---|---|---|---|---|
| 1 | any | `python -m scripts.query_types.wld download` | `data/wld/videos/`, `downloads.json` | minutes |
| 2 | any | `python -m scripts.query_types.wld prepare` | `data/wld/videos_360p/`, `films.json`, `truth.json` | minutes |
| 3 | GPU | `scripts/query_types/run_yoloe_gpu.sh GPU` | `data/query_types/wld/yoloe/{variant}/`, `yoloe/validation.json` | ~3 h on one TITAN Xp |
| 4 | key | Experiment 1, for `r` in 1 2 3: `python -m scripts.query_types.wld_presence run --plan P --repeat r` for `P` in `mllm_full mllm_early yoloe_mllm` | `presence/run{r}/{P}/` | ~50 min, ~10.8M input tokens per repeat |
| 5a | key | Experiment 2, for each `r`: `python -m scripts.query_types.wld_localization run --repeat r --stage S` for `S` in `candidates transcript_only mllm_video` | `localization/run{r}/{S}/` | ~15 min, ~6.2M input tokens |
| 5b | GPU | `CUDA_VISIBLE_DEVICES=GPU python -m scripts.query_types.wld_localization run --repeat r --stage yoloe_windows` (step 3 runs it for the repeats it is given) | `localization/run{r}/yoloe_windows/` | ~15 min |
| 5c | key | `... wld_localization run --repeat r --stage S` for `S` in `transcript_mllm yoloe_mllm transcript_yoloe_mllm` | `localization/run{r}/{S}/` | ~50 min, ~3.5M input tokens |
| 6 | any | `python -m scripts.query_types.wld_presence score` and `... wld_localization score` | tables printed; `presence/summary.json`, `localization/summary.json` | seconds |

On one machine with both a GPU and the key, `wld_localization run --repeat r` runs steps 5a–5c in order.
Move `data/query_types/wld/` between machines with `rsync` when steps run on different ones.

Step 3 needs PyTorch with CUDA (the scans refuse to run on a CPU). It runs one process at a time on
the GPU you name, and nothing else of ours should run on the machine meanwhile: six scans side by
side doubled every scan's time. To keep it running after logging out, with `PYTHON` naming the
environment's interpreter:

```bash
PYTHON=.venv/bin/python nohup scripts/query_types/run_yoloe_gpu.sh 0 1 > logs/run_yoloe_gpu.log 2>&1 &
```

The YOLOE work ran on one TITAN Xp of a shared server, inside the dev container from the
`setup/freddie-gpu` branch (`docker exec -d <container> bash -lc '<the line above>'`).

Outputs (all under `data/query_types/wld/`; files starting with `_` are bookkeeping):

```
yoloe/{small,small_vocab,large,large_vocab}/   one scan per pair (one-name) or per film (vocabulary)
yoloe/validation.json                            real vs estimated early-stop time, five pairs
presence/run{r}/{plan}/{film}__{animal}.json     answer, time, tokens, every call's usage
localization/run{r}/{stage}/{film}__{animal}.json
presence/summary.json, localization/summary.json the scored tables; the only results kept in git
*/_run_info.jsonl                                git commit, versions, weights checksums, machine, time
*/_failed_usage.jsonl                            tokens billed by attempts that failed
```

## Frozen settings and why

| Setting | Value | Where | Why |
|---|---|---|---|
| Gemini model | `gemini-3.1-flash-lite` | `wld_gemini.MODEL` | Same as the paper |
| Prompts | presence, video localization, narration-only, narration candidates | `wld_presence.PROMPT`, `wld_localization.*_PROMPT` | Frozen before running |
| Present | visible ≥ 1 s in total; under 1 s is "brief" and not scored | `wld.MIN_PRESENT_SECONDS` | Sub-second appearances measure frame sampling, not the plans |
| Labeled intervals | frames merged across gaps ≤ 1 s; stretches < 1 s are ignored time | `wld.MERGE_GAP_SECONDS` | Same 1 s rule |
| YOLOE rule | confidence ≥ 0.25 in one sampled frame, sampled at 5 fps; sweep 0.1–0.6 × 1–3 frames | `wld_yoloe.MAIN_RULE`, `SWEEP_*` | YOLOE's defaults, no tuning |
| YOLOE filter | `large`: yoloe-11l, original video, one name | `wld_yoloe.FILTER_VARIANT` | The only variant that found all 16 present pairs |
| Window padding | ±10 s | `wld_yoloe.PAD_SECONDS` | The paper's lecture setting |
| Early-stop chunks | 2 min | `wld_presence.CHUNK_SECONDS` | Fine enough to stop early, few enough calls |
| Clips for Gemini (Exp 2) | standalone, ≤ ~2 min, CRF 28, 64k audio, sent inline | `wld_localization.CLIP_*` | Fits Gemini's 20 MB inline limit |
| Interval metric | tIoU ≥ 0.3, one-to-one | `wld_localization.TIOU_THRESHOLD` | The paper's lecture metric |
| Repeats | 3 for Gemini plans (mean ± sd); YOLOE once (deterministic) | — | Gemini answers vary run to run |
| YOLOE timing | one process at a time on one GPU; each timing records the load average | `run_yoloe_gpu.sh` | Processes side by side compete for the CPUs: six at once doubled every scan's time |
| Prices | Gemini 3.1 Flash-Lite: $0.25 per 1M input tokens ($0.025 if cached), $1.50 per 1M output; GPU: $0.526/h (AWS g4dn.xlarge, one T4, on demand), both as of 2026-10-25 | `common.GEMINI_PRICES`, `common.GPU_DOLLARS_PER_HOUR` | Published list prices. The T4 rate stands in for the TITAN Xp, which no cloud rents, so each plan also reports the GPU price at which it breaks even |

**How Gemini sees video.** Whole films and Experiment 1's windows: the film's public YouTube URL,
with start/end offsets for windows; nothing is uploaded. Experiment 2's windows: standalone clips
sent inline. Two things forced this:
- Uploading through the Files API filled the project's 20 GB storage quota, and deleting the files
  did not free it (a known Google issue).
- On YouTube start/end windows, Gemini's localization times were inconsistent (window time, film
  time, or neither) from run to run. Clips start at 0, so clip time is unambiguous. Yes/no answers
  (Experiment 1) do not depend on times.

**Data fixes** (each fails loudly if the file changes). The two label fixes were checked by eye where a
model disagreed with WLD's labels; nothing else was relabeled, so other label errors may remain.
- `wld.LABEL_REMOVALS`: the "bear" boxes in `aN3jEIAXvbk` are opening landscape shots.
- `wld.LABEL_ADDITIONS`: a leopard walks through the cage shot at 343.88–348.24 s (frames 8597–8705)
  in `UsnHCvC92mM`, which WLD labels only as "tiger".
- `wld.SUBTITLE_FIXES`: one timestamp typo in `aN3jEIAXvbk`'s subtitles.
- `wld.MAX_FRAME_DIFF` = 2: today's YouTube copies are 1–2 frames off the 2017 labels (≤ 0.08 s).

## Rewrites and the conditions they need

What the two experiments show about when each rewrite pays off. The numbers are means over 3 repeats;
`presence/summary.json` and `localization/summary.json` have every number.

| Rewrite | Used as | Holds when | Evidence |
|---|---|---|---|
| A detector replaces the video model | substitute | The detector does not confuse the queried animal with similar ones | Fails here: every YOLOE variant confuses similar species (tiger and leopard, for example), with precision 0.35–0.41, so presence F1 is 0.48–0.57, against 0.99 for Gemini |
| The detector picks the windows the video model watches | filter | The detector finds every animal that is there (recall ≈ 1); it saves money only when the GPU is cheap enough to pay for a full scan | Holds for presence: F1 1.000, 26% of the video, 74% fewer tokens. Cheaper than Gemini on the whole film below $0.58 per GPU-hour |
| The detector stops at its first hit | exact | Always exact for the detector; it saves time when the animal is present | Small model: 576 s instead of 3,482 s, with the same answers |
| The video model stops at its first "yes" (2-min windows) | approximation | The animal is as visible in a 2-minute window as in the whole film | F1 0.948 against 0.989: it missed a 4 s leopard in all 3 runs, which whole-film Gemini saw in 2 of 3; 34% fewer tokens |
| The narration localizes the animal | substitute | The narration describes what is on screen when it is on screen | Fails: seconds F1 0.32, because narration says when an animal is discussed, not when it is visible |
| The narration picks the windows the video model or detector sees | filter | The narration mentions the animal while it is on screen. That holds for a film's subject (tiger, leopard: 80–91% of on-screen time inside the windows), not for animals that appear in passing (7 of 16 present pairs below 20%), so those need a fallback | The windows cover 11% of the films and contain 78% of labeled time |
| Narration windows, then the detector, then the video model | filter chain | Both filters hold | Each stage raises precision (0.23, then 0.61, then 0.77) while recall goes from 0.78 to 0.48. Best seconds F1, 0.595, with 8% of the video sent; $0.30 against $1.43 for Gemini on the whole film |

## Caveats

- 10 of WLD's 15 films are no longer on YouTube; results cover the 5 that are.
- YOLOE times are measured on a TITAN Xp, one scan at a time and one frame per `predict` call. They
  include decoding (but not converting) the frames between samples. The server is shared with other
  users; each timing's `load_average` shows how busy it was.
- `Detect` gives YOLOE one frame per call, which kept the GPU only about 67% busy. Batching frames
  would likely shorten every YOLOE time. It is left to a follow-up, which should be re-timed the same
  way (one process at a time).
- Each question is timed and priced as its own query. A vocabulary scan answers all 9 animals of a
  film in one pass, but each pair is still charged that whole pass (`wld_yoloe.load_scans`). Asked
  together, the 45 questions would share 5 passes: 727 s (about $0.11) for `large_vocab`, not
  6,546 s.
- Gemini times include Google fetching the video from YouTube, but no upload, unlike the paper's
  Table 2 runs, which uploaded files.
- In Experiment 2, about 13% of the intervals Gemini returns for clips lie past the clip's end and
  are dropped as invalid; the raw answers are kept in each result's `answers`.
