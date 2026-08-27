# Getting Started with MMDS

This guide takes you from a fresh clone to **running your first query**. It assumes no
prior knowledge of the project — just basic command-line comfort.

For *what* MMDS is, see [README.md](README.md). For *how it's built*, see
[DESIGN.md](DESIGN.md). This page is purely "set it up and run something."

---

## 0. Prerequisites

**Supported platforms:** Linux (x86_64 / aarch64), Apple-Silicon (M-series) macOS, or Windows.

> ⚠️ **Intel (x86_64) macOS is not supported.** A transitive dependency (`torch`, pulled in
> by `ultralytics` for the `Detect` operator) ships no Intel-mac wheel, so `uv sync` will
> fail. Use a Linux machine (e.g. a lab/EECS server) or an Apple-Silicon Mac instead.

You do **not** need to install Python yourself — the project pins Python 3.12 and `uv`
will fetch it for you. You do **not** need a GPU for the first example (it uses a hosted
model); a GPU only helps the local `Detect` operator.

---

## 1. Get the code

```bash
git clone git@github.com:chanwutk/mmds.git
cd mmds
```

(If you're reading this inside an already-cloned repo, just `cd` into it.)

---

## 2. Install `uv`

[`uv`](https://docs.astral.sh/uv/) is the package/environment manager this project uses.

```bash
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your shell (or follow the installer's hint) so `uv` is on your `PATH`, then verify:

```bash
uv --version
```

---

## 3. Build the environment

```bash
uv sync
```

This reads `uv.lock`, downloads Python 3.12 if needed, creates a `.venv/`, and installs
all dependencies. **The first run is large** — it pulls `torch`, `opencv`, and
`ultralytics` (hundreds of MB), so give it a few minutes on a good connection. Subsequent
runs are instant.

After it finishes, `.venv/bin/python` exists and `import mmds` works.

---

## 4. Smoke test (no API key, no network)

Confirm the install is healthy by running the unit suite. The tests are **hermetic** —
they mock out OpenCV, `yt-dlp`, and the object detector, so they need no API key, no
network, and no model downloads:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```

You should see a run ending in `OK`. If you do, your environment is correctly set up.
(This is also the project's primary verification command — re-run it after any change.)

---

## 5. Get a free Gemini API key

The first real example calls Google's Gemini model, which needs an API key. Get a free one:

1. Go to **[Google AI Studio → API keys](https://aistudio.google.com/apikey)**.
2. Sign in and click **Create API key** (the free tier is enough to start).
3. Export it in your shell:

```bash
export GEMINI_API_KEY="paste-your-key-here"
```

> The SDK also accepts `GOOGLE_API_KEY`. If both are set, `GOOGLE_API_KEY` wins.
> To persist the key across sessions, add the `export` line to your `~/.bashrc` /
> `~/.zshrc` (don't commit it to git).

---

## 6. Run your first query

```bash
./run examples/wildlife_species.py
```

`./run` executes the `output` expression in that file through the Gemini executor and prints
the result rows as JSON. The example
([`examples/wildlife_species.py`](examples/wildlife_species.py)):

1. loads video clips from `data/animals.jsonl`,
2. asks Gemini *"what animals do you see in this clip?"* for each, and
3. `Unnest`s the answer into one row per detected species.

Each row's video is a **public YouTube URL**, which Gemini ingests on its servers — so this
example needs network access but does **not** download anything locally. Expect output like:

```json
[
  { "video": { "type": "VideoView", "source": "https://www.youtube.com/watch?v=…", "start": 0, "end": 19 },
    "title": "2021 Swan Valley Wildlife Trail Camera Compilation",
    "animal_types": "deer" },
  …
]
```

(Exact species depend on the model.) 🎉 That's your first MMDS query.

Want to try the aggregation example next? `./run examples/wildlife_species_count.py`
groups those rows by species and asks the model to count the clips per species.

---

## 7. Optional local datasets

The first examples use public URLs, so no media checkout is needed. The two
I24V clips used by the cross-camera example are also bundled. Full-corridor
I24V, UCA, and campus workflows reference optional local files through small
committed manifests; those additional MP4 and PNG assets are excluded from Git.

Follow [`data/README.md`](data/README.md) and its dataset-specific guides to copy
or symlink authorized files into the expected paths. The example drivers perform
a preflight check and print every missing local-media path with that README
reference before starting model inference or Gemini calls.

---

## 8. Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| `uv sync` errors about `torch` having no wheel for your platform | You're on **Intel macOS**, which is unsupported. Use Linux or an Apple-Silicon Mac. |
| `uv: command not found` | `uv` isn't on your `PATH` yet — restart your shell or re-run the installer's PATH hint (step 2). |
| `ModuleNotFoundError: No module named 'cv2'` (or `mmds`) | The environment isn't built/active. Re-run `uv sync` (step 3) and use `./.venv/bin/python`. |
| The query raises a missing-API-key / authentication error | `GEMINI_API_KEY` isn't set in the current shell (step 5). |
| `Required local media is missing` | Install the optional dataset files listed in [`data/README.md`](data/README.md); the cross-camera I24V clips, manifests, and ground truth are committed. |
| `Gemini returned an empty response` / invalid JSON | Usually a transient model issue or a video Gemini couldn't access (must be a public URL). Re-run; try a different clip. |
| First example is slow | Gemini is fetching and analyzing the video server-side. Give it time; later runs of the same clip are faster. |

---

## 9. Next steps

You're set up. To start contributing:

- **[DESIGN.md](DESIGN.md)** — the authoritative architecture document. Read this first:
  it explains the operators, the plan model, parsing/rendering, execution, and the
  invariants you must not break.
- **[AGENTS.md](AGENTS.md)** — the required workflow and architectural constraints (e.g.
  keep DESIGN.md in sync with code, add tests for every behavior change).
- **[`examples/`](examples/)** — more runnable queries, including a `Map → Filter` video
  pipeline and local YOLOE object detection (`Detect`, no API key needed).
- **Test-driven loop** — re-run the suite from step 4 after every change; expand
  [`tests/`](tests/) for any new behavior.

Welcome aboard!
