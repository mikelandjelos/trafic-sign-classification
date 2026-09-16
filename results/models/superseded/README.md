# Superseded model artifacts

Models trained under the **per-method preprocessing** protocol, which was withdrawn on
2026-09-16 when preprocessing was fixed at `raw_gray` for the whole comparison
(`PROJECT_TASKS.md` §1, `00-INDEX.md` Q7).

Kept rather than deleted because they are the artifacts behind numbers that still appear in
the report as the *measured preprocessing effect* (note 08 addendum, task 9.7). They are not
part of the comparison and `scripts/evaluate_grid.py` must never load them.

| file | method | preproc | val macro-F1 | superseded by |
|---|---|---|---|---|
| `pca_svm_clahe_gray.joblib` | `pca_svm` | `clahe_gray` | 0.7977 | `pca_svm_raw_gray.joblib` (0.7713) |
| `cnn_feat_svm_clahe_gray.joblib` | `cnn_feat_svm` | `clahe_gray` | 0.9797 | `cnn_feat_svm_raw_gray.joblib` (0.9774) |

**Why this directory exists at all.** The 8.1 runner refuses to start when two artifacts
match one method, rather than picking by sort order — silently evaluating the comparison on
a superseded model is exactly the Q5 class of failure it was written to prevent. Moving these
here resolves the ambiguity without destroying anything.

Regenerable at any time:

    poetry run python scripts/train_pca.py --preproc clahe_gray
    poetry run python scripts/train_cnn_features.py --preproc clahe_gray
