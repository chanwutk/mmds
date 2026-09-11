# SoccerNet 25-game experiment dataset

This pipeline freezes the first 25 games in SoccerNet's official Action
Spotting training registry, downloads both 224p and 720p halves plus
`Labels-v2.json`, validates every artifact, audits sampled speech/language, and
transcribes only games manually confirmed to contain English commentary.

The commands are deliberately gated. Omitting `--execute` prints a preview and
does not write data, contact SoccerNet, load Whisper, or download model weights.

## 1. Review the exact sample

```bash
PYTHONPATH=. ./.venv/bin/python -m scripts.soccernet.download
```

The selection strategy is `first_in_official_registry`, not a randomized
sample. On the first executed download, the exact ids are frozen in
`data/soccernet/selection.json`; later runs refuse to silently change them.

## 2. Download and validate after code approval

`SOCCERNET_PASSWORD` may be exported by the shell or stored in the repository
root `.env`. The password is never written to selection or manifest files.

```bash
PYTHONPATH=. ./.venv/bin/python -m scripts.soccernet.download --execute
```

Downloads first land below `data/soccernet/.partial` and are validated before
an atomic move into their final game directory. Existing valid files resume;
existing invalid destination files cause a hard failure instead of being
silently overwritten. By default, the downloader refuses to start another file
once free space falls below a 20 GiB reserve.

The final `manifest.json` includes label counts, shown/not-shown goal counts,
file sizes, duration, video/audio codecs, audio stream counts, and aggregate
224p-versus-720p size/duration/audio comparisons.

## 3. Audit likely commentary and language

The default audit uses the 224p copy because processing duplicate 720p audio
adds cost without useful evidence. It samples three 30-second windows from
each half and runs multilingual Whisper `small`:

```bash
PYTHONPATH=. ./.venv/bin/python -m scripts.soccernet.audit_audio --execute
```

Outputs:

- `audio_audit.json`: automatic sample-level language, text, and speech scores.
- `audio_review.csv`: a manual review worksheet.

Whisper can detect speech and language, but it cannot reliably prove that a
speaker is match commentary rather than an announcement or interview. For each
game, manually fill these columns in `audio_review.csv`:

- `manual_has_commentary`: `yes` or `no`
- `manual_language`: `en` for English
- `approved_for_transcription`: `yes` only when both checks pass

Rerunning the audit refuses to overwrite this worksheet unless
`--force-review-template` is supplied explicitly.

## 4. Transcribe approved games

After manual review:

```bash
PYTHONPATH=. ./.venv/bin/python -m scripts.soccernet.transcribe --execute
```

Full transcription reads the 224p copies and defaults to Whisper `turbo` with
English fixed as the language. It writes timestamped segment JSON and plain
text below `data/soccernet/transcripts/<game-id>/`. Completed halves resume
without another model call. The command refuses to transcribe a row unless the
manual worksheet explicitly confirms both commentary and English.

## Verification

These tools use lazy imports and hermetic unit tests, so the main suite does
not contact SoccerNet or load a Whisper model:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```

## Three-game goal-pushdown experiment

The active v4 paper experiment is frozen to these English-commentary games:

- Everton 3–1 Chelsea
- Chelsea 2–0 Aston Villa
- Chelsea 2–2 West Brom

It compares three modes: full-video naive localization, transcript-only final
localization, and high-recall transcript filtering followed by audiovisual
localization of physically materialized candidate clips. The candidate radius
is ±30 seconds. Naive and transcript-to-video use the exact same video prompt,
output schema, model, and timestamp contract.

Preparation is offline. It produces six half-level input rows, a separate
evaluator-only ground-truth file, and rendered plans for review:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment prepare
```

The prediction input contains videos and transcripts but no labels or goal
timestamps. The controlled directory is
`goal_pushdown_3_games_v4_materialized`; completed v3 and v2 directories are
preserved unchanged. The predictor and evaluator configurations are separate,
and prediction stages do not open the ground-truth or evaluation files.

Candidate ranges are re-encoded serially as zero-origin MP4/H.264/AAC clips.
The optimized Gemini input is an ordinary standalone `Video`, not a
`VideoView`. Each clip's bytes, duration, start time, streams, SHA-256, size, and
probe metadata are validated before inference. A checkpoint safely resumes
interrupted encoding.

Review the generated v4 plans under:

```text
data/soccernet/experiments/goal_pushdown_3_games_v4_materialized/plans/
```

Model and encoding stages require `--execute` and write durable usage,
media-upload, provenance, and completion records. Run each command only after
validating the preceding artifact:

```bash
# 1. Full-video baseline.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment \
  run-naive --execute

# 2. Transcript-only final predictions.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment \
  run-transcript-only --execute

# 3. High-recall transcript candidates; inspect candidates.jsonl afterward.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment \
  run-candidates --execute

# 4. Physically materialize the candidate windows and validate every clip.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment \
  materialize-clips --execute

# 5. Run the shared video localizer over standalone candidate clips.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment \
  run-transcript-video --execute

# 6. Deterministically evaluate all three immutable prediction artifacts.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment \
  evaluate

# 7. Accuracy, pruning, latency, upload, token, and cost comparison.
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment compare
```

The transcript-only method is preplanned by `prepare`. It asks the LLM for
final goals—not high-recall candidates—and never supplies a video. Every
returned timestamp must equal the start of an existing transcript segment.
Invalid timestamps remain false positives and are never repaired using ground
truth. This isolates the marginal accuracy and cost contribution of the
short-video VLM localizer.

The optional generated-signal 1B method belongs to the historical v3
experiment. It is intentionally absent from the controlled v4 runner and paper
comparison.

End-to-end latency is emitted only for clean cold-start stages. If execution is
resumed after a provider failure or partial encoding, cached responses or reused
clips and failed encoding attempts make the successful retry's wall time incomplete. The comparison sets
combined latency and its reduction to `null`, records the invalidating stages,
and keeps observed stage-time and API-time diagnostics. Accuracy, tokens,
uploads, and estimated cost remain available.

Check stage completion without executing anything:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m scripts.soccernet.goal_experiment status
```

All VLM calls use the stable `gemini-3.1-flash-lite` model and 224p videos so
the physical plan is the only experimental variable. Predictions are matched
one-to-one against visible SoccerNet goals at ±5, ±10, ±30, and ±60 seconds,
with ±30 seconds as the primary metric.
