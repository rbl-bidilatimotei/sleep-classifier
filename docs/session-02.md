# Session 2: preparation analysis

Date: 2026-10-29. Scope: development participant 1066528.

## Observed results

There are 952 labeled epochs under the assumed epoch-start convention.
Heart rate has 77 empty epochs. Motion has 124 empty epochs.
Exact duplicate removal removes 11,078 heart-rate rows and zero motion rows.
Conflicting readings cause an explicit error. Raw files remain unchanged.
The step history contains no records within the 24 hours before PSG.
These findings describe one participant, not the full cohort.

## Coverage definition

The report contains unique readings, occupied one-second bins, and the largest observation gap per epoch.
The gap calculation includes both epoch boundaries. Empty epochs receive a 30-second gap.
Bin occupancy describes timestamp density, not signal quality or a validity decision.
Heart rate and motion have different sampling frequencies. Their occupancy fractions are not directly comparable.
No interpolation, rejection threshold, or imputation is applied.

## Label boundary evidence

The original source revision is `c357a9a26bdb26c3ca0c25fe3df374fdefe92564`.

- [Epoch definition](https://github.com/ojwalch/sleep_classifiers/blob/c357a9a26bdb26c3ca0c25fe3df374fdefe92564/source/preprocessing/epoch.py): duration is 30 seconds.
- [PSG loading](https://github.com/ojwalch/sleep_classifiers/blob/c357a9a26bdb26c3ca0c25fe3df374fdefe92564/source/preprocessing/psg/psg_service.py): label times are reconstructed from the first timestamp at 30-second increments.
- [Observation alignment](https://github.com/ojwalch/sleep_classifiers/blob/c357a9a26bdb26c3ca0c25fe3df374fdefe92564/source/preprocessing/raw_data_processor.py): timestamps are floored into buckets anchored at the first PSG timestamp.

This code supports epoch-start indexing as the original classifier's convention.
It does not prove the label export's timestamp semantics.
Our analysis explicitly assumes [t, t + 30 seconds). An observation at the end belongs to the next epoch.
Confirm export semantics before treating this assumption as established ground truth.

## Reproduce

```bash
uv sync --locked --group notebook
uv run --locked python -m sleep_classifier.analysis --config configs/explore.yaml
uv run --locked --group notebook python scripts/execute_notebook.py --notebook notebooks/02_coverage_and_steps.ipynb
```

Tables, figures, and provenance are saved under `artifacts/audits/1066528-coverage/`.
The executed notebook is saved under `artifacts/notebooks/`.
The analysis verifies source checksums before reading the participant files.
Your IPython dependency and lockfile edits are included in this work chunk.

## Next milestone

Extend the verified audit across the cohort for integrity and eligibility checks.
Select test participants before using development coverage distributions to choose thresholds.
Keep participant 1066528 in development. Preserve missing step coverage explicitly.
The current notebook is the first preparation analysis, not a completed cohort audit.
