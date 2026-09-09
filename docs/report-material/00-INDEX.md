# Report material — index

Notes written **as the work happens**, to be consumed when writing the final report
(`docs/report/`, Serbian). Nothing here is reconstructed after the fact: every decision,
number, and figure gets its rationale recorded at the time it is made.

Distinct from:

- `docs/PROJECT_TASKS.md` — the task tracker (what to do, what is done)
- `docs/proposition/` — the already-submitted proposal (`predlog_projekta.tex`)
- `docs/report/` — the final Serbian writeup (does not exist yet)

## Conventions

- One file per report section, numbered in the order they will likely appear.
- Working language English; the report is translated at the end. Terms that have an
  established Serbian rendering are noted inline as `English (srpski)`.
- Every claim that will appear in the report cites either a results.csv row, a figure
  file, or a source in `docs/proposition/references.bib`.
- **Surprises get written down even when they are inconvenient.** A prediction that
  failed is more valuable to the discussion section than one that held.

## Notes

| Note | Feeds report section | Status |
|---|---|---|
| [01-environment-and-setup.md](01-environment-and-setup.md) | Methodology → reproducibility; Appendix | complete (task 0.1) |
| [02-dataset-structure.md](02-dataset-structure.md) | Data; Methodology → train/val protocol | complete (task 0.2) |
| [03-reproducibility.md](03-reproducibility.md) | Methodology → protocol; Limitations | complete (task 0.3) |
| [04-splitting-protocol.md](04-splitting-protocol.md) | Methodology → experimental protocol | complete (task 1.2) |

<!-- Add a row per note as it is created. Keep status honest: in progress / complete / stale. -->

## Open questions to resolve before the report

Running list; each gets closed with a decision and a rationale, not dropped.

| # | Question | Raised at | Status |
|---|---|---|---|
| Q2 | **Opportunity, not a problem.** The claim that a random split inflates validation accuracy is currently argued rather than measured. Training one method twice — track-disjoint split vs. random per-image split — turns it into a number. ~15 min once task 4.3 exists. | task 1.2 | **approved** — scheduled as task **4.5**; see [04](04-splitting-protocol.md) |
| Q1 | GTSRB crops carry only a ~10 % margin (min 5 px) around the sign, so task 3.4's "±40 % jitter, re-crop from source, no padding" is not literally achievable — an outward 40 % expansion leaves the image for essentially every sample. Need a policy: clamp to image bounds, restrict to inward/positional jitter, or allow padding at high levels and report it as a limitation. | setup, before task 0.1 | **open** — decide at task 3.4 with measured numbers on how many samples clip at each level |
