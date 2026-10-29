# Session 3: cohort readiness

Date: 2026-10-29. This milestone prepares the cohort for the first model.

## Integrity and eligibility

The complete PhysioNet v1.0.0 archive was downloaded. All 124 participant files passed publisher checksum verification.
The inventory checks 31 participants for numeric data, label timing, step counts, conflicting readings, and sensor overlap with scored epochs.
The report records eligibility and file hashes. It does not expose stage distributions or test coverage metrics.
Two participants fail the current conflict rule: 3509524 and 759667.
Their heart-rate files contain different values at identical timestamps. Raw files remain unchanged.
They are excluded from the first baseline until a separate conflict-resolution policy is agreed.
This exclusion narrows the population covered by our eventual results.

The initial strict label check was revised before selecting the split.
The original code recognizes -1 as unscored and 4 as legacy N4.
These are valid source codes. They do not invalidate a participant.
Unscored epochs are excluded from coverage summaries. N4 remains a recognized scored stage.
The four-class preparation proposal combines N3 and N4 as deep sleep.

Source: [original stage definitions](https://github.com/ojwalch/sleep_classifiers/blob/c357a9a26bdb26c3ca0c25fe3df374fdefe92564/source/sleep_stage.py).
The epoch-start convention remains the explicit assumption documented in session 2.

## Frozen participant split

`configs/splits/baseline.json` records seed 42, participant IDs, and the inventory digest.
There are 23 training, three validation, and three test participants.
Participant 1066528 is in training. Its earlier inspection cannot contaminate the final test set.
The split command reuses identical files and refuses to overwrite a different split.
Changing exclusions later requires a separately versioned protocol and split, not an overwritten baseline.

Test participants received integrity and eligibility checks only.
Detailed coverage reports read the 26 development participants after the split.
The development report verifies their file hashes against the inventory.
An automated test rejects any attempted load of test-participant observations during development analysis.

## Candidate coverage rules

Each rule requires observations from both sensors. Gaps include both epoch boundaries.
Motion occupancy measures occupied one-second bins within a 30-second epoch.

| Rule | Minimum HR readings | Maximum HR gap | Motion occupied bins | Maximum motion gap | Retained development epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| observed | 1 | 30 s | at least one observation | 30 s | 23015 / 23425 (98.25%) |
| moderate | 3 | 15 s | 24 / 30 | 5 s | 22708 / 23425 (96.94%) |
| strict | 4 | 10 s | 27 / 30 | 2 s | 22300 / 23425 (95.20%) |

Select `moderate` as the starting baseline rule in `configs/coverage.yaml`.
It limits long gaps while retaining most development epochs. It is not a validated optimum.
No model performance was used to select it. Thresholds must remain fixed before final test evaluation.
The report measures candidate eligibility; feature generation and rejection are the next implementation milestone.

For sequence preparation, retain original time indices and break sequences at rejected or unscored intervals.
Reset recurrent state across these breaks. Do not concatenate separated epochs as adjacent observations.
Report retained coverage alongside classification metrics, including participant-level coverage.

## Step history

Two development participants have no recorded step-bin support in the prior 24-hour interval.
Other participants also have incomplete support. Never substitute zero activity for missing records.
The plot measures recorded interval support, not validated continuous step-sensor availability.
The final bin has an unknown endpoint and contributes no inferred support.
The first model omits steps. A later comparison must include historical steps and coverage indicators.

## Reproduce

```bash
uv run --locked python -m sleep_classifier.download_cohort --archive data/sleep-accel-1.0.0.zip --destination data/raw/sleep-accel/1.0.0
uv run --locked python -m sleep_classifier.cohort inventory --config configs/cohort.yaml
uv run --locked python -m sleep_classifier.cohort split --config configs/cohort.yaml
uv run --locked python -m sleep_classifier.development --config configs/coverage.yaml
uv run --locked --group notebook python scripts/execute_notebook.py --notebook notebooks/03_cohort_readiness.ipynb
```

The archive is about 550 MB. Extracted data requires about 2.2 GB, plus the retained archive.
Generated reports remain under `artifacts/cohort/readiness/`. Raw data and reports are excluded from Git.
Notebook 03 reads the shared report implementation. Its source remains output-free.

Validation: 47 tests passed. Notebook 03 executed in a fresh local kernel, and its generated plot was inspected.
No training, accuracy result, GPU check, or Colab validation is included in this milestone.
