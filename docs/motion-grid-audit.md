# Audit fixed-time-lag motion changes before model integration

Status: development-only audit. No motion-grid model training or feature promotion is included.

## Why this milestone

The [twenty-epoch HR reference](experiments/hr-history-20-2026-11-19/README.md) does not improve the primary score.
Retain the ten-epoch full-HR working recipe and stop the local HR-window tweaks.
The earlier [causal audit](causal-feature-audit.md) found axis-change information that magnitude summaries discard, alongside differing sampling cadences.
A change per adjacent observation depends on the observation spacing, so its amplitude needs context before becoming a model feature.

Audit one declared fixed-time-lag definition and its support before selecting a later training ablation.
Use the working full-HR reference and the completed twenty-epoch model as related development evidence.
Their agreement is not independent replication.

## Sampling definition

For each completed interval [start, start + 30 seconds):

1. Place grid ticks at start + 1 through start + 29 seconds
2. Select the latest raw observation at or before each tick
3. Require that observation to belong to the same epoch
4. Reject a reading older than 0.2 seconds, allowing 1e-8 seconds for rounding
5. Compare adjacent grid ticks only when both readings are supported and their timestamps increase
6. Average the Euclidean norms of those axis-vector differences

This yields 29 scheduled points and 28 potential pairs per epoch.
The first grid point is inside the interval, and the final point precedes its exclusive end.
No interpolation, future observation, cross-epoch pair, or carry from a previous epoch enters the diagnostic.
The same exact-duplicate/conflict policy used in preparation applies to raw observations.

The general implementation permits a positive grid step below fifteen seconds and a reading-age limit smaller than that step.
This audit fixes one second and 0.2 seconds; it does not sweep values against classification scores.
Actual selected sample lags can differ from exactly one second. Record their median, minimum, and maximum.

## Values, support, and limits

The diagnostic is an axis-vector change in source acceleration units. It is not position displacement, jerk, or a time derivative.
One supported pair is sufficient to report an exploratory mean; low support remains visible rather than becoming model eligibility.
No supported pairs means a missing mean. Observed zero change remains zero.
Keep every existing target, including unsupported cases.

Record supported grid point/pair counts and fractions, maximum reading age, and actual sample lags.
Report support sensitivity at fractions 0.5, 0.8, and 0.95 without selecting a training threshold.
Show coverage by participant and stage. Missingness can differ across them and must not disappear from the analysis.

A fixed declared lag reduces one cadence ambiguity on constructed trajectories. It does not correct sensor filtering or guarantee cadence independence.
One-second endpoints can miss faster activity returning to the same vector between ticks.
Orientation changes also affect axis vectors; a larger value is not a calibrated measure of movement severity.
The notebook demonstrates both added direction information and the aliasing limitation with synthetic observations.

## Data boundary and immutable evidence

Use only the same 26 development participants and 22,708 matched eligible targets.
Read the three reserved identifiers as metadata solely to enforce exclusion; never open their observations.
VerifiedMotionSource checks the publisher checksums and development allowlist before each raw motion read.
Raw files, prepared features, model weights, scalers, predictions, and participant assignments remain unchanged.

The reference is artifacts/transformer-window-local-hr-grouped-validation.
The comparison is artifacts/hr-history-20-grouped-validation under the explicit window_history policy.
Saved scores, prediction probabilities, source hashes, and target alignment are validated before the raw audit.
There is no model-training CLI option in this audit.

Write a new hashed snapshot under artifacts/motion-grid-audit. Existing snapshots block reruns.
Snapshot contents include rows, summary, portable dashboard, static figures, configuration hash, implementation hashes, and output hashes.
The new math module is included in the implementation record.
Cached reporting verifies the snapshot without rereading raw sensor observations.
A cached snapshot is historical evidence; it does not revalidate every source file on each view.

## Inspect the findings

Open [notebook 35](../notebooks/35_motion_grid_audit.ipynb).
Read each What/Why note. Keep RUN_SETUP false in the current environment.
Keep RUN_AUDIT false when the cached snapshot exists.
The remaining cells load verified rows and expose support, cadence, stage availability, and reference-error contrasts.

Per-participant stage medians reduce the temptation to treat neighboring epochs as independent people.
Reference-error summaries compare missed-to-Light Wake/Deep/REM targets with correct targets of the same participant and true stage.
True-Light distributions provide a separate contrast; they are not interchangeable with correctly detected Deep or REM.
These label-conditioned views are post-hoc diagnostics, never model inputs or feature-performance estimates.
The older persistent-error flag retains its Deep/REM-to-Light definition and is not relabeled as a Wake diagnostic.

The portable dashboard adds optional grid-change, support, age, and actual-lag views.
Actual time gaps, segment resets, and missing values remain visible. It uses embedded data and no network requests.
Legacy audit snapshots retain their original views when optional catalogs are absent.
Offline rendering tests and static plot inspection do not establish native-browser layout verification.

## Run one new audit

Plan without raw reads or model construction:

```bash
uv run --locked --group baseline --group notebook python -m sleep_classifier.causal_feature_audit \
  --config configs/motion-grid-audit.yaml --plan
```

Create a snapshot only when its output directory does not exist:

```bash
uv run --locked --group baseline --group notebook python -u -m sleep_classifier.causal_feature_audit \
  --config configs/motion-grid-audit.yaml --run
```

The notebook uses the validated experiment's config path and logs its identity before launching that command.
It rejects an existing snapshot before the CLI starts. Preserve measured evidence rather than clearing directories.
Fresh clones require the development features, completed grouped artifacts, and checksum-verified raw motion files.

Export the cached report:

```bash
uv run --locked --group baseline --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/35_motion_grid_audit.ipynb
```

Export skips setup, raw audit, and recursive export. It does not train or create a raw snapshot automatically.
Keep source notebooks output-free and executed HTML under ignored artifacts/notebooks.

## Decision after review

First check support, reading ages, actual lags, and residual cadence differences.
Then inspect within-person stage and error contrasts, including overlap with true Light.
Visual separation alone cannot establish model value or an information ceiling.
If evidence supports a candidate, define one later feature ablation with training-only transforms, explicit availability, unchanged targets, and the same folds.
Keep all training operator-controlled. The final test remains untouched.
