# Repeat the fixed-lag motion candidate on MPS

Status: prepared for five local seed-42 fits. No new model has been trained.

## Question and reference

The [CPU motion candidate](experiments/fixed-lag-motion-2026-11-20/README.md) improved mean fold macro-F1 by 0.00715 against an MPS reference.
Only folds 02 and 05 improved, and pooled selected-checkpoint NLL worsened.
That comparison changed both the feature recipe and the execution backend.
It cannot establish how much the motion feature contributed.

Repeat the same motion candidate on MPS. Reuse the completed full-HR ten-epoch MPS reference.
Keep the CPU run intact as separate exploratory evidence. Do not overwrite or retrain either existing study.
The primary comparison is new MPS motion versus existing MPS no-motion, with no device exception.
No result is available until all five new fits complete.

## Fixed recipe

The candidate YAML differs from the original motion YAML only in device selection and output path.
Use explicit `device: mps`, rather than `auto`. A machine without MPS must stop instead of falling back to CPU.
The original candidate actually trained on CPU. The reused reference actually trained on MPS.
The new configured and resolved backend must both be MPS.

Retain training seed 42, split seed 42, the same five participant folds, and all 22,708 eligible development targets.
Retain the same verified motion snapshot, one-second lag, 0.2-second maximum observation age, and within-epoch backward selection.
Retain seven raw inputs, nine internal channels, mean imputation fitted on training targets, and training-only scaling.
Retain both HR channels, ten strictly past HR epochs, minimum history four, and thirty raw context epochs.
Retain P3 patches, width 16, two attention heads, one layer, feedforward width 32, and dropout 0.2.
Retain forty epochs, patience twelve, batch size 64, learning rate 0.001, gradient clip 1.0, and the original weighted cross-entropy.
Retain zero-decay Adam, no scheduler or smoothing, the same diagnostics, and selection by validation macro-F1.
Both motion candidates have 2,900 parameters. The no-motion reference has 2,852; that necessary input-projection difference remains explicit.

No source dataset, prepared CPU fold, fitted state, prediction, checkpoint, or final-test observation changes.
The new study uses artifacts/fixed-lag-motion-mps-grouped-validation.
The previous CPU study remains artifacts/fixed-lag-motion-grouped-validation.
The reused reference remains artifacts/transformer-window-local-hr-grouped-validation.

## Local execution

Open [notebook 37](../notebooks/37_fixed_lag_motion_mps.ipynb). Restart the kernel before its first run.
Every code cell explains what it does and why. Source setup, preparation, checks, and training switches default to false.
The five folds are already prepared in this checkout. Keep RUN_SETUP and RUN_PREPARATION false here.
Enable RUN_CHECK for optional no-gradient checks. Enable RUN_TRAINING when ready for the five new fits.
The training cell derives the command path from the validated experiment and checks its prepared device against the measured reference.
Existing result directories block accidental retraining.

A fresh checkout requires the development features, completed MPS reference artifacts, and the verified motion snapshot with its config.
The optional setup cells clone the PR branch and install locked uv groups.
Run preparation once in an MPS-capable environment:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/fixed-lag-motion-mps-grouped.yaml --prepare
```

Run training only when ready:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/fixed-lag-motion-mps-grouped.yaml --train
```

No-gradient validation and cached reporting do not optimize a model.
Report-only notebook execution does not require an available MPS device when all execution switches remain false.
The actual preparation, preflight, and training commands require MPS.
Keep the three reserved final-test participants untouched.

## Evaluation and limits

Use the existing `motion_features` paired policy with `allow_device_mismatch=False`.
Both paired scores and temporal slices reject a device mismatch. Do not carry the notebook-36 exception into this repeat.
Verify the same folds, target identities, feature source hashes, fitted HR state, and model parameter counts before interpreting results.
Judge mean paired fold macro-F1 first. Show all folds and participants, pooled accuracy, stage precision/recall/F1, NLL, and error destinations.
Check whether Deep improves consistently, whether REM→Light errors decrease, and whether Wake recall or startup targets worsen.
Keep the original checkpoint rule instead of selecting a favorable epoch from the completed curves.

Matching the backend removes the known device difference from this feature comparison.
It does not make one seed independent replication or guarantee identical numerical behavior across runtime versions.
The original extra 48 projection weights also remain part of the feature-recipe intervention.
Review this complete matched result before changing architecture, loss, regularization, or seed count.
The CPU scores can remain beside it for descriptive context; they are not a second device-matched feature comparison.
