# Lecture paper-evaluation profiles

This package is isolated from the earlier `scripts/lectures/` v1-v5 pilot. It
supports two immutable profiles over the same three methods:

1. full-video audiovisual localization (`naive`);
2. timestamped-transcript-only localization (`transcript_only`);
3. high-recall transcript filtering, deterministic clip materialization, then
   the same audiovisual localizer (`transcript_video`).

Every external or model-backed command previews by default. Add `--execute`
only after reviewing `catalog` and `status`.

## Verified three-pair profile

`verified-3pair` is the final human-verified positive-pair localization
experiment. It contains exactly three matched lecture-query rows and does not
form a Cartesian product:

- Lecture 3 / tone-induced glass shattering: `[4427,4432]`;
- Lecture 9 / fire-induced resonant sound: `[1643,1654]`;
- Lecture 20 / successful large soap bubble: `[435,441]` and `[460,466]`.

The soap query excludes the partial attempts at `[444,446]` and `[479,480]`.
All three input pairs are positive, so this profile measures conditional event
localization and execution efficiency, not negative rejection or pair-presence
specificity. Use the profile explicitly on every command:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m scripts.lecture_paper_eval.experiment \
  --profile verified-3pair catalog
PYTHONPATH=src:. ./.venv/bin/python -m scripts.lecture_paper_eval.experiment \
  --profile verified-3pair status
PYTHONPATH=src:. ./.venv/bin/python -m scripts.lecture_paper_eval.experiment \
  --profile verified-3pair download
```

Its default root is `data/lecture_verified_eval/`; it never modifies the
completed four-by-four artifacts.

## Frozen four-by-four profile

Annotation version 2 contains seven human events. Lecture 15 has four Chladni
formation intervals on the full-video clock: `[4195,4225]`, `[4289,4300]`,
`[4338,4360]`, and `[4375,4395]`. The last three correct a missing leading
one-hour component in annotation version 1; the first was separately confirmed
by the human annotator.

```bash
PYTHONPATH=src:. ./.venv/bin/python -m scripts.lecture_paper_eval.experiment catalog
PYTHONPATH=src:. ./.venv/bin/python -m scripts.lecture_paper_eval.experiment status
PYTHONPATH=src:. ./.venv/bin/python -m scripts.lecture_paper_eval.experiment download
```

The default profile remains `frozen-4x4`, with default root
`data/lecture_paper_eval/`, so all earlier commands remain compatible.

## Immutable stage order

The stage order is identical for both profiles:

```text
download --execute
transcribe --execute
prepare
run-naive --execute
run-transcript-only --execute
run-candidates --execute
materialize-clips --execute
run-transcript-video --execute
evaluate
compare
```

Fresh preparations write annotation version 2 directly. A preparation created
with annotation version 1 before the timestamp clarification must instead run
this one-time local command before evaluation:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m scripts.lecture_paper_eval.experiment amend-ground-truth
```

The amendment does not overwrite `ground_truth.json` or either prediction
configuration/completion. It writes versioned evaluator sidecars plus a
content-addressed audit record containing the original and replacement hashes
and a snapshot of every prediction artifact that existed at amendment time.
It is refused after evaluation starts. Prediction stages continue to validate
only the unchanged label-free prediction prefix.

Prediction stages validate only the label-free prediction prefix. The evaluator
owns and opens the active ground-truth revision; model plans never receive it or
any derived field. One-time Whisper time is reported separately from query-time
latency.
