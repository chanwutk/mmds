# Cross-Modal Lecture Experiment Runbook

This experiment compares three physical plans over three MIT 8.03SC lectures
and three queries (nine lecture-query pairs):

1. naive full-lecture audiovisual verification;
2. transcript-only final interval retrieval;
3. transcript candidate selection, physical materialization of padded/merged
   candidate windows as standalone clips, and audiovisual verification of only
   those clips.

The frozen defaults are Gemini `gemini-3.1-flash-lite`, Whisper `small`, English
transcription, `fps=1.0`, 30 seconds of candidate context on each side, and a
primary temporal-IoU threshold of 0.3. `catalog` prints the complete configuration.

No download, Whisper invocation, or Gemini invocation occurs unless the relevant
command includes `--execute`.

## 1. Inspect the frozen catalog

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment catalog

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment status
```

The catalog contains source URLs and query text, but no answers or expected
lecture-query matches.

## 2. Preview, then download the three lectures

The first command is a no-op preview. Run the second only after approving the
catalog and downloader.

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment download

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment download --execute
```

Downloads go to `data/lectures/videos/` through `.part` files. Each file must
contain both video and audio streams according to `ffprobe`; its byte size,
SHA-256 hash, media metadata, and source URL are then frozen in
`data/lectures/manifest.json`. An interrupted Range download or a crash after
validation can be resumed without accepting unvalidated bytes.

Before transcription, manually play every video and confirm that:

- it is the intended lecture;
- native audio is present and understandable;
- the lecturer speaks English;
- the duration is plausible and playback reaches the end.

You can inspect stream metadata without modifying anything:

```bash
ffprobe -v error -show_streams -show_format \
  data/lectures/videos/MIT8_03SCF16_lec03_300k.mp4
```

Repeat the check for lectures 09 and 11, then inspect status again.

## 3. Preview, then transcribe once with Whisper small

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment transcribe

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment transcribe --execute
```

Whisper output is materialized once per lecture under
`data/lectures/transcripts/`. Every segment receives a stable consecutive
`segment_id`; later LLM stages may select only those IDs. The transcript is
content-bound to the video SHA-256 and Whisper model, so changed media or model
metadata fails closed instead of silently reusing an old transcript.

The raw model result is atomically saved first as `*.whisper.raw.json`, including
its source hash, model, requested language, and elapsed time. If normalization
or a later write fails, rerunning the same command reuses that raw checkpoint
without invoking Whisper again. `status` reports this recoverable state as
`raw_checkpoint`.

Whisper may emit slightly overlapping adjacent segments. The workflow preserves
those original boundaries when segment starts and ends both remain
nondecreasing. It still rejects reversed, backtracking, non-finite, and
zero-length boundaries. Candidate padding and union-merging make ordered overlap
safe for the downstream plans.

Whisper timestamps are quantized and the final segment can end a few
milliseconds beyond the container duration. The versioned normalization
contract permits only the final segment to overrun by at most 0.1 seconds,
clamps its normalized end to the exact `ffprobe` duration, and records the raw
end, overrun, and reason. Internal overruns or larger final overruns remain
errors. If the contract changes, matching raw checkpoints rebuild normalized
artifacts without loading or invoking Whisper.

After transcription, verify `language` is `en`, read representative segments
from the beginning/middle/end, and compare several timestamps to the video.

## 4. Create and complete evaluator-only ground truth

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment init-review
```

Edit:

`data/lectures/experiments/cross_modal_lecture_3x3_v1/ground_truth_review.json`

For all nine pairs, watch the video and record every minimal actual event
interval as `start_seconds`/`end_seconds`. An empty `events` list is a reviewed
negative—not an unreviewed pair. Set `review_status` to `complete`, fill
`reviewer`, and retain useful annotation notes.

Only after all nine decisions have been checked:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment freeze-review
```

This creates immutable `ground_truth.json`. Prediction inputs are built
separately and recursively checked for evaluator-only keys.

## 5. Prepare and inspect all plans

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment prepare
```

Review `config.json`, `input.jsonl`, and every normalized plan under the
experiment's `plans/` directory. The configuration freezes source, transcript,
input, prompt, schema, and label hashes. The naive and optimized verifiers use
the same audiovisual prompt and output schema. The primary optimized plan uses
standalone zero-origin clips. The older offset-view plan remains only to audit
the initial timestamp-coordinate failure and is excluded from the primary
comparison.

Calling a run command without `--execute` remains a no-op:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-naive
```

## 6. Execute one approved stage at a time

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-naive --execute

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-transcript-only --execute

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-candidates --execute
```

At this point, stop and inspect:

- `runs/transcript_candidates/candidates.jsonl`;
- `runs/transcript_candidates/candidate_metrics.json`;
- coverage of every positive event;
- total candidate duration and selectivity;
- invalid model-selected segment ranges.

After approving the candidate output:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment materialize-clips

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment materialize-clips --execute
```

The first command is a no-op preview. The executed command accurately re-encodes
every merged candidate range as a standalone MP4 under
`runs/materialized_clips/clips/`. It freezes each source range, source hash,
output hash, byte size, duration, stream metadata, encoding contract, and local
elapsed time. Interrupted work resumes only from content-validated checkpoint
entries.

Before any new Gemini call, inspect `runs/materialized_clips/manifest.json`,
`runs/materialized_clips/input.jsonl`, and play all clips. Confirm that:

- each clip is the intended lecture/query window;
- playback starts at clip time zero and contains both video and audio;
- the requested event remains visible or audible in every positive candidate;
- no source lecture offset appears in `materialized_video_context`;
- `status` reports `materialized_clips: complete`.

After approving the clips:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-optimized-materialized --execute

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment compare
```

Do not rerun `run-optimized`: that command addresses the preserved offset-view
diagnostic under `runs/optimized`. The corrected primary result is written to
`runs/optimized_materialized` without changing the diagnostic predictions.

Use `status` between any two steps. Provider calls are serialized for the
primary latency benchmark. Successful responses are cached only for recovery;
if a run uses cache hits or records failed provider attempts, its cold-start
latency is explicitly marked invalid while accuracy and durable usage records
remain inspectable.

Each completed stage ends with `completion.json`, containing SHA-256 hashes of
its predictions/candidates or materialization manifest/input, metrics, and
timing artifact. Materialized clip bytes are independently hashed inside their
manifest. Altered completed artifacts are rejected by downstream stages and
reported as `invalid_completed_artifacts` by `status`.

## 7. Interpret the result

All three methods use the exact same ground truth, temporal-IoU thresholds
(0.1, 0.3, 0.5), primary threshold (0.3), and prediction deduplication threshold
(0.8). Invalid intervals cannot match and remain false positives. Evaluation
reports precision, recall, F1, and exact lecture-query-pair accuracy, including
reviewed negative pairs.

The comparison keeps query-time latency separate from one-time Whisper
materialization. Optimized query time includes transcript candidate generation,
local standalone-clip materialization, and clip verification. It also reports
materialization bytes/time, token/cost reductions, and candidate-duration
reduction. `fps=1.0` is explicit for both naive and optimized verification;
audio remains present in every materialized clip.

## 8. V2: binary verification followed by conditional episode localization

V2 is additive: it does not rewrite the completed v1 plans, predictions,
evaluation, or comparison. It reuses the content-addressed high-recall
transcript candidates and standalone clips, while including their original
cold-start costs in optimized totals.

First freeze and inspect the v2 contracts locally:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment prepare-v2
```

Review `v2_config.json` and all four rendered plans under `plans/v2/`. The naive
and optimized plans must share the same binary prompt/schema and the same
complete-episode prompt/schema. Only the supplied video extent may differ.

Preview both binary stages, then execute only after approval:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-naive-v2-binary
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-optimized-v2-binary

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-naive-v2-binary --execute
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-optimized-v2-binary --execute
```

Stop and inspect each `runs/v2/*_binary/decisions.json` and
`binary_evaluation.json`. Binary evaluation is pair-level: multiple candidate
clips are combined with OR, and a pair with no candidate predicts absence.
`localization_input.jsonl` retains every decision so that the rendered
localization plan performs an explicit `Filter(event_present)`.

After approving the decisions, preview and execute the conditional localizers:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-naive-v2-localization
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-optimized-v2-localization

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-naive-v2-localization --execute
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-optimized-v2-localization --execute
```

Only positive decisions invoke the VLM localizer. The localizer must return
complete contiguous experiment episodes rather than isolated evidence moments.
Finally:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment compare-v2
```

`v2_comparison.json` reports binary F1 and temporal tIoU separately, exposes
per-stage API usage, and sums binary plus conditional-localization cost. Use
`status` between every stage; v2 state appears in its own nested section.

## 9. V3: transcript-owned timestamps with audiovisual boundary refinement

V3 is additive and does not rerun the naive method. `prepare-v3` binds the
existing completed v1 naive artifacts, freezes two new plans, and records that
this three-lecture run is a post-hoc method-development diagnostic rather than
held-out paper evidence.

First freeze and inspect the local contracts:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment prepare-v3
```

Review `v3_config.json` and both plans under `plans/v3/`. The proposal plan must
return complete episodes as existing Whisper segment IDs. The refinement plan
must contain `episode_refinements` with `maxItems: 1`, receive the aligned
transcript excerpt, and contain no minute/second output fields.

Preview, then run only the transcript proposal stage:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v3-transcript-proposals

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v3-transcript-proposals --execute
```

Stop and inspect `runs/v3/transcript_episode_proposals/proposals.jsonl` and
`proposal_metrics.json`. Check that each positive physical experiment is one
complete proposal rather than several sub-actions. A proposal may span several
minutes. Its 30-second padding is verification context only and is not its final
predicted interval.

After approving the proposals, preview and materialize their clips:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment materialize-v3-proposal-clips

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment materialize-v3-proposal-clips --execute
```

Inspect and play every clip under
`runs/v3/materialized_proposal_clips/clips/`. Also inspect
`refinement_input.jsonl`: every proposed boundary must appear in
`transcript_context_segment_ids`, and the aligned excerpt must cover the full
padded clip.

After approving those inputs, preview and execute audiovisual refinement:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v3-refinement

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v3-refinement --execute
```

The VLM may reject a proposal with an empty list or return exactly one refined
range using allowed transcript segment IDs. It cannot return free-form video
timestamps. Finally run:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment compare-v3
```

`v3_comparison.json` compares the refined temporal intervals with the frozen v1
naive baseline using the same tIoU evaluator. It includes proposal tokens/cost,
local clip materialization, and refinement tokens/cost. Before treating this as
paper evidence, freeze the method on separate development lectures and evaluate
it once on held-out lectures.

## 10. V4: mandatory-predicate evidence grounding

V4 preserves the completed v3 diagnostic and reuses only its transcript
proposals and standalone clips. It does not reuse the v3 refinement predictions.
The v1 naive result remains frozen and is not rerun.

First freeze and inspect the v4 contracts:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment prepare-v4
```

Review `v4_config.json`, `v4_query_input.jsonl`, and both plans under
`plans/v4/`. The query-condition plan must receive only query text. The grounding
plan must accept only per-condition segment-ID evidence and contain no
minute/second output fields or model-generated final interval.

Preview, then compile each distinct query once:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v4-query-conditions

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v4-query-conditions --execute
```

Stop and inspect `runs/v4/query_conditions/conditions.jsonl`. Every explicit
object, action, modality, measurement, temporal relation, and causal relation in
the query must remain mandatory. Reject a decomposition that weakens the query
to topical similarity. Also inspect `grounding_input.jsonl`; it must contain all
eight v3 proposal clips and attach the same conditions to every row sharing a
query.

After approving those decompositions, preview and execute predicate grounding:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v4-predicate-grounding

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v4-predicate-grounding --execute
```

The VLM returns evidence only for required conditions directly established by
the clip. Missing any required condition rejects the proposal. When all are
present, deterministic code creates the smallest interval spanning their
Whisper segment evidence. Unknown, duplicate, invented, or out-of-context IDs
remain invalid false-positive predictions.

Finally run:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment compare-v4
```

`v4_comparison.json` includes the original v3 proposal/materialization cost,
three query-compilation calls, and predicate-grounding calls. It explicitly
excludes the unused v3 refinement cost. This remains a post-hoc diagnostic; use
different lectures to freeze the method before a held-out paper evaluation.

## 11. V5: mandatory gates with anchor-only timestamps

V5 preserves all completed v1-v4 artifacts. It reuses the frozen v3 transcript
proposals and clips, but prevents contextual conditions from widening the final
timestamp. Every compiled condition is either a `gate` or an `anchor`: both are
mandatory for acceptance, while only anchor evidence determines the returned
interval.

First freeze the v5 contracts locally. This command makes no model calls:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment prepare-v5
```

Review `v5_config.json`, `v5_query_input.jsonl`, and both rendered plans under
`plans/v5/`. Confirm that the query compiler receives only query text, its
schema requires role values `gate` or `anchor`, and the grounding schema permits
only condition IDs plus Whisper segment IDs—never free-form timestamps.

Preview, then compile each distinct query once:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v5-query-conditions

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v5-query-conditions --execute
```

Stop and inspect `runs/v5/query_conditions/conditions.jsonl` before any VLM
grounding call. For each query, verify that:

- every explicit object, action, modality, measurement, temporal relation, and
  causal relation is retained;
- no unstated simultaneity, causality, ordering, or actor identity was added;
- conditions are nonredundant and at least one condition is an `anchor`;
- an `anchor` describes the target occurrence whose boundaries should be
  returned, while a `gate` is only a required qualification;
- the same frozen conditions are attached to every proposal row for that query
  in `grounding_input.jsonl`.

After approving the roles, preview and execute role-aware predicate grounding:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v5-predicate-grounding

PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment run-v5-predicate-grounding --execute
```

The proposal is rejected unless evidence exists for every gate and anchor. If
all conditions are satisfied, deterministic code returns the smallest span
covering anchor evidence only; gate evidence cannot widen it. Finally run:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lectures.experiment compare-v5
```

`v5_comparison.json` includes the original v3 proposal/materialization cost,
the three v5 query-compilation calls, and v5 grounding calls. It excludes v3
and v4 grounding. Because the role distinction was designed after inspecting
the pilot's v4 output, v5 remains a post-hoc diagnostic that needs separate
development lectures and a once-only held-out evaluation.
