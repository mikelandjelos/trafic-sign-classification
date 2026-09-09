# GTSRB Representation Comparison

See @docs/PROJECT_TASKS.md for full scope, task list, and known gotchas.

## Hard rules

### Working cadence

- **One subtask at a time.** Finish it, verify it, present it for review, and STOP.
  Do not start the next subtask until it has been explicitly reviewed and approved.
- Ask before installing anything.
- Commit when asked. Commit messages are a brief, clear one-liner.

### Documentation

- Implementation related docs can't ever lag behind the implementation - they must always be up to date;
- The tasks @docs/PROJECT_TASKS.md must be checked/updated as we go; they can change over time (this is just a draft), but it must always be up-to-date with the current work;

### Report material

The report is as important as the implementation. Nothing gets reconstructed on Day 5.

- Every task that produces a decision, a number, or a figure gets a note in
  `docs/report-material/`, written **as the work happens**, and indexed in `00-INDEX.md`.
- Unresolved decisions go in the open-questions table in `00-INDEX.md`, not in chat.
- Surprises and mistakes get written down even when inconvenient — a failed prediction is
  worth more to the discussion section than one that held.
- `docs/proposition/` = the submitted proposal; `docs/report/` = the final Serbian writeup;
  `docs/report-material/` = the working notes that feed it.

### Environment

- Python 3.11.11 via pyenv, Poetry-managed. Run everything through `poetry run ...`
  (`poetry run python -m gtsrb.data`, `poetry run pytest`).
- `.python-version` lists 3.11.11 AND 3.13.0 on purpose — the `poetry` shim lives in 3.13.
- `poetry run ruff check src/ tests/` before committing.

### Project specific related

- Train/val splits MUST be track-disjoint. Never split by image.
- **The track key is `(class_id, track_id)`.** The raw `TTTTT` filename prefix restarts
  inside every class directory — 75 raw prefixes vs 1307 real tracks.
- LinearSVC only. Never SVC(kernel='rbf') -- O(n^2) on 39k samples.
- Dense SIFT for BoVW. Detector-based SIFT returns zero keypoints on small crops.
- CPU-only. No CUDA available.
- Always `from gtsrb import config; config.set_seeds()` at the top of a script. Paths come
  from `config`, never hardcoded.
- Degradations MUST draw from `config.rng_for(degradation, level, index)`. A single global
  stream makes the degraded test set depend on execution order, so the five methods would
  be compared on different pixels.
- All results append to `results.csv` in tidy format:
  `run_id, method, preproc, degradation, level, metric, value`, where `run_id` is
  `<timestamp>-<commit>` and keys into `results/platform_<run_id>.json`.
- Timings are relative costs in one recorded environment, never deployment latency.
