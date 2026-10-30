# Session 4: causal feature preparation

Date: 2026-10-30. Scope: training and validation data preparation.

## Dataset summary

The source contains 31 participants. Two have unresolved conflicting heart-rate readings.
The baseline contains 23 training, three validation, and three test participants.
NREM is a parent category, not a separate target alongside light and deep sleep.
The four targets are wake (0), light (1), deep (2), and REM (3).
Source N1 and N2 map to light. N3 and legacy N4 map to deep.
Source -1 remains unscored and receives target -1 with eligibility false.

Raw sensor streams are not synchronized rows. Preparation summarizes each sensor within an assumed [t, t + 30 seconds) epoch.
The prediction becomes available at the epoch end. This is delayed classification rather than next-epoch forecasting.
The source timestamp convention remains an explicit assumption.

## Prepared representation

The six model inputs are heart-rate mean, standard deviation, minimum, maximum, acceleration-magnitude mean, and acceleration-magnitude standard deviation.
Heart rate uses bpm. Acceleration magnitude uses g and includes gravity.
Standard deviations use population normalization (ddof=0).
Motion magnitude removes axis orientation, but it can also discard useful axis information.
This is a simple baseline representation, not a claim of optimal features.

Each participant CSV also contains timestamps, targets, sensor coverage, eligibility, and segment IDs.
Only `FEATURE_COLUMNS` defines model inputs. Metadata and targets must not become model features.
Raw steps remain available, but this baseline omits them.

Exact duplicates are removed from derived observations. Conflicting observations cause an error.
Empty epoch summaries remain NaN. No imputation, interpolation, or scaling is performed.
An eligible epoch must satisfy the frozen moderate coverage rule and have a scored target.

Rows retain their original timestamps. Rejected and unscored rows receive segment ID -1.
A retained row after an invalid row or missing epoch starts a new segment.
Segment IDs are participant-local. Sequence construction must use both participant and segment ID.
Reset recurrent state across those boundaries.

## Results and limits

| Partition | Participants | All label rows | Eligible scored epochs |
| --- | ---: | ---: | ---: |
| Training | 23 | 20913 | 19928 |
| Validation | 3 | 2914 | 2780 |

The retained total is 22,708, matching the earlier development coverage report.
The all-label denominator includes unscored rows; the earlier 96.94% denominator includes scored epochs only.
Test observations remain unopened. Preparation verifies development source hashes against the inventory and saved split.
The generated provenance records the rule, split, source hashes, and explicit feature allowlist.

No model is trained. Scaling will be fitted on eligible training rows and reused on validation and later test rows.
This pipeline produces batch epoch summaries. A streaming sensor buffer is a later implementation task.
Tests verify that later valid observations cannot change earlier features.
They do not establish clinical validity or generalization to new devices.

## Reproduce

```bash
uv run --locked python -m sleep_classifier.prepare --config configs/prepare.yaml
uv run --locked --group notebook python scripts/execute_notebook.py --notebook notebooks/04_causal_features.ipynb
```

Run the session 3 inventory and split commands first on a fresh checkout.
Output is saved under `data/processed/baseline/` and excluded from Git.
Notebook 04 imports the shared implementation and displays one training participant.

Validation covers epoch boundaries, duplicate conflicts, missing observations, target mapping, segment breaks, and test-participant isolation.
The next experiment is a majority-class reference and a simple non-sequential classifier, followed by a small GRU.
Use the same six features and participant split to compare these models.

## Repository reconciliation and style update

PR #3 was merged into main on 2026-10-30. PR #4 remained open and retained the earlier cohort history.
Current main was merged into the feature branch without rewriting its history.
Conflicts used the merged cohort implementation and retained the feature-specific additions and user edits.

Feature construction now uses `EpochFeatureBuilder`. Preparation uses `FeaturePreparationJob`.
Their smaller methods preserve the existing public wrapper functions and output schema.
Source notebooks remain `.ipynb` files. The report runner exports executed HTML directly from memory.
Executed `.ipynb` files are no longer written by the runner.
Training execution requires separate, explicit authorization from Timo.
