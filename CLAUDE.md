# GTSRB Representation Comparison

See @docs/PROJECT_TASKS.md for full scope, task list, and known gotchas.

## Hard rules

### Documentation

- Implementation related docs can't ever lag behind the implementation - they must always be up to date;
- The tasks @docs/PROJECT_TASKS.md must be checked/updated as we go; they can change over time (this is just a draft), but it must always be up-to-date with the current work;

### Project specific related

- Train/val splits MUST be track-disjoint. Never split by image.
- LinearSVC only. Never SVC(kernel='rbf') -- O(n^2) on 39k samples.
- Dense SIFT for BoVW. Detector-based SIFT returns zero keypoints on small crops.
- CPU-only. No CUDA available.
- All results append to results.csv in tidy format.
