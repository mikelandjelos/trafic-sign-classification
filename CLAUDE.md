# GTSRB Representation Comparison

See @docs/PROJECT_TASKS.md for full scope, task list, and known gotchas.

## Hard rules

### Working cadence

- **One subtask at a time.** Finish it, verify it, present it for review, and STOP.
  Do not start the next subtask until it has been explicitly reviewed and approved.
- Ask before installing anything.
- Commit when asked. Commit messages are a brief, clear one-liner.

### Always build a demo

**Every significant task ships a demo before it counts as verified.** A task is significant
if it adds a stage to the pipeline: a preprocessing config, a degradation, a representation,
a classifier stage, the evaluation grid.

- Lives in `scripts/demo/<topic>_mechanics.py`, writes to `figures/demo/<topic>/`, and is
  listed in that task's `docs/report-material/` note.
- It must be **built from the real module**, never a re-derivation. A demo that reimplements
  the thing it is checking verifies nothing.
- Deterministic and re-runnable from a clean checkout. Figures are gitignored; the script is
  the artifact.

**Why this is a hard rule: tests pass on things that are visibly wrong.**

- The motion-blur kernel's extent depended on its angle (4.00 px at 0° vs 2.83 px at 45°).
  Every numeric test passed. *Plotting the kernels* is what exposed it — and it had already
  been written into the docs with a confident, wrong justification.
- PCA at k=2 reconstructs every sign, triangles included, toward a speed-limit disc, and
  spends real variance on background brickwork. Neither is visible in explained-variance
  numbers; both belong in the report.

So the demo must be able to **fail**: plot the quantity that would expose a bug, not the one
that flatters the implementation. When a demo reports a diagnostic that could be misread
(e.g. "small residual = robust"), put the correcting quantity in the same figure.

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
- `docs/proposition/` = the project proposal; `docs/report/` = the final Serbian writeup;
  `docs/report-material/` = the working notes that feed it.
- **NEVER modify `docs/proposition/` unless explicitly asked to.** It is the author's
  document. If implementation diverges from it, record the divergence in
  `docs/report-material/` and raise it — do not edit the proposal to match the code.
  This applies to `predlog_projekta.tex`, `references.bib`, and the build artifacts.

### Never tailor the project to the predictions

`predictions.md` is locked and dated. It is a **record of what was expected**, not a target.

- **Never choose a design, parameter, or configuration because it protects a prediction.**
  Every choice must be defensible on grounds that would still hold if the prediction were
  the opposite. If the only argument for a setting is "otherwise the predicted mechanism
  won't show", that is a reason to *measure both*, not to pick one.
- Diverging from a prediction is a fine outcome — the requirement is that we can say
  clearly **what happened and why the assumption was imprecise**. Task 9.6 is that table,
  and it is more valuable when rows say "wrong, because …".
- A mechanism that is measured (e.g. "PC1 is 52 % brightness") is evidence. Do not upgrade
  it into confirmation of a prediction it merely rhymes with; record the counter-reading
  too, *before* the result is in.

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
