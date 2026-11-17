# Window-local heart-rate deviation ablation

Status: prepared for local training by Timo. No candidate model scores exist yet.

## Question

The [causal audit](experiments/causal-feature-audit-2026-11-17/README.md) found persistent Deep/REM-to-Light errors across four related recipes.
Past-relative heart-rate patterns differ between some persistent and reference-correct targets.
These descriptive differences do not prove separation from true Light or a classification benefit.
Test one feature family against the original unsmoothed Adam/P3 reference.

Add two deterministic channels inside each supplied model window:

1. Current completed epoch mean heart rate minus a strictly past mean
2. A binary flag indicating sufficient past history in that same window

The six original features remain the public raw-window input.
The transform expands them to eight channels before patch formation and projection.
It summarizes information already present in the window. It adds no new sensor observations.

## Preserve the fifteen-minute information boundary

Use at most ten previous eligible epochs and require at least four.
Exclude the current epoch from the past mean.
For every input position, use only earlier positions inside the supplied window.
The sequence dataset already resets at participant and gap-induced segment boundaries.
The transform never accesses a participant table, preceding window, future token, label, or padded observation.

Do not precompute these channels across a complete session and then slice thirty-epoch windows.
That approach could expose ten additional epochs through features attached to the oldest token.
It would extend the effective history from thirty to forty epochs and confound the feature comparison.
The new transform restarts its available history at each window boundary.

For a sequence of heart-rate means 60, 61, 62, 63, 66:

| Position | Earlier epochs available | Past mean | Raw deviation | Availability |
| --- | ---: | ---: | ---: | ---: |
| First | 0 | Unavailable | Unavailable | 0 |
| Second | 1 | Insufficient | Unavailable | 0 |
| Third | 2 | Insufficient | Unavailable | 0 |
| Fourth | 3 | Insufficient | Unavailable | 0 |
| Fifth | 4 | 61.5 | 4.5 BPM | 1 |

The first four positions in every window have insufficient window-local history.
At the target position, availability matches the audit definition because the thirty-epoch context contains the required ten past epochs.
Keep startup targets. Do not remove unavailable cases from training, checkpoint selection, or reporting.
This is a heart-rate trend representation, not physiological HRV from beat-to-beat intervals.

## Scaling and checkpoints

Fit the original six-feature StandardScaler on training participants as before.
Fit deviation normalization on one available target deviation per training epoch.
Do not count every overlapping input-token occurrence as a separate normalization sample.
Validation participants do not contribute to either fit.

The model receives normalized original heart rate.
Subtracting its past normalized mean cancels the original scaler mean.
Multiply that difference by the saved original heart-rate scale to recover a BPM deviation.
Then apply the saved deviation mean and scale.
A constant training deviation uses scale one.
Unavailable standardized deviations become zero after normalization; their availability flag remains zero.
Available positions have flag one. The binary flag is not standardized.

The fitted state records its training-only origin, available target count, original heart-rate scale, and deviation mean/scale.
Checkpoint metadata records the transform definition and effective channel names.
Selected and intermediate checkpoints restore the same transform and normalization without fitting on inference observations.
The input API and stored raw scaler remain six-dimensional.

## Fixed comparison

| Setting | Reused reference | Candidate |
| --- | --- | --- |
| Input representation | Six original channels | Six original channels plus deviation and availability |
| Raw input API | 30 epochs × 6 features | Same |
| Effective model channels | 6 | 8 |
| Parameters | 2,756 | 2,852 |
| Patches / width / heads / layers / feedforward | 3 epochs / 16 / 2 / 1 / 32 | Same |
| Dropout / optimizer / rate | 0.2 / original Adam / 0.001 | Same |
| Loss / sampling | Unsmoothed cross-entropy / natural | Same |
| Budget / patience / batch size | 40 / 12 / 64 | Same |
| Clipping / scheduler / OPAL | Norm 1.0 / none / none | Same |
| Training seed / fold split seed | 42 / 42 | Same |
| Checkpoint selection / diagnostics | Validation macro-F1 / enabled | Same |

The two extra channels add 96 input-projection parameters.
Initialization layout also changes. This is a feature-recipe comparison, not a parameter-matched proof of feature value.
Raw eligibility, source feature tables, and the three final-test participants remain unchanged.
Reuse all five original reference folds. Train five fresh candidate fits, capped at 200 epochs.

## Run locally

Open [notebook 29](../notebooks/29_window_local_heart_rate.ipynb).
Read each What/Why note before its code cell.
Keep setup disabled in the current environment.
Prepare the candidate once. Keep preparation disabled after its manifest exists.
Inspect the fold plan, transform definitions, training-only normalization, and five no-gradient checks.
Enable RUN_TRAINING when ready. The notebook reuses the reference and never retrains it automatically.

The candidate model configuration is `configs/transformer-window-local-hr.yaml`.
The grouped configuration is `configs/transformer-window-local-hr-grouped.yaml`.
Results use `artifacts/transformer-window-local-hr-grouped-validation`.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-window-local-hr-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-window-local-hr-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-window-local-hr-grouped-validation/fold-01/gru.yaml --check
```

The checks fit preprocessing on training participants and run forward passes without optimization.
Start training only through the gated cell or this explicit command:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-window-local-hr-grouped.yaml --train
```

Existing result directories block accidental reruns. Automatic resume is unsupported.
Fresh clones require development features and the completed reference artifacts.
Keep MPS to match the measured reference.

## Decision criteria

Primary: paired mean fold macro-F1 and each original fold difference.
Inspect participant differences and Wake/Light/Deep/REM precision, recall, F1, support, and confusion matrices.
Keep hard-label NLL, availability groups, short-history behavior, and transitions as secondary evidence.
Do not select a model from a single stage, fold, or lower NLL alone.
Use the explicit `window_features` paired policy to reject changes in other controls or the effective history.

A useful seed-42 gain requires replication before replacing the reference.
The audit is post-hoc development evidence, and repeated use of these folds is adaptive exploration.
No final-test or performance improvement claim is supported before training.

Export the report after the local run:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/29_window_local_heart_rate.ipynb
```

Report execution skips setup, preparation, participant training, and recursive export.
Keep source notebooks output-free. Store executed HTML and large model artifacts locally under ignored artifacts.

## Preparation evidence

The [preparation snapshot](experiments/window-local-hr-preparation-2026-11-17/README.md) records five MPS forward checks and synthetic CPU/MPS causal checks.
All 791 tests pass. The notebook exports from a fresh kernel without training.
All 130 copied feature tables match the reference byte-for-byte. Candidate model scores remain pending.
