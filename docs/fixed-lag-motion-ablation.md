# Test one fixed-lag motion feature

Status: five local candidate fits are complete. Reporting includes an explicit CPU/MPS limitation.

## What the audit showed

The [completed audit](experiments/motion-grid-audit-2026-11-19/README.md) retained 22,708 targets from 26 development participants.
All targets have a supported estimate; 22,685 meet the descriptive 95% point-and-pair support check.
This validates feasibility, not prediction benefit.

The reference assigns 2,203 of 4,860 REM targets to Light, versus 1,509 of 3,119 Deep targets.
Those rates are 45.3% and 48.4%. REM has the larger count, while Deep has the larger rate.
The persistent-error chart counts mistakes shared by both related HR recipes, rather than every reference error.
Its counts are 1,792 REM targets and 1,336 Deep targets.
These models share data and architecture; agreement does not provide independent replication.

Wake motion-magnitude variation has a broad distribution across participant medians.
Each plotted value is one participant's median within a true stage, not one epoch or a confidence interval.
The box spans the middle half of those medians. Whiskers use Matplotlib's usual 1.5-IQR convention; outliers are hidden.
Wake can include both quiet and active periods. Sensor characteristics and participant behavior can also contribute to the spread.
This plot does not identify their separate causes.
Light, Deep, and REM motion distributions overlap substantially. A motion feature could help some decisions while leaving quiet errors unresolved.

## The intervention

Retain the full-HR ten-epoch seed-42 recipe.
Append only `motion_grid_change_mean` to the six existing prepared columns.
The unchanged HR transform then adds its deviation and availability channels: seven external aggregate inputs become nine internal channels.

Inside each completed 30-second epoch, place ticks at seconds 1 through 29.
At each tick, use the latest same-epoch raw observation at or before it, no more than 0.2 seconds old.
Average axis-vector differences over supported neighboring tick pairs.
Use the existing tested `FixedLagMotionFeatures` implementation. Never interpolate, read a future sample, or join across epochs.
The feature uses source acceleration units. It is not displacement, jerk, or a calibrated movement score.
One-second endpoints can miss faster activity that returns to the same vector between ticks.

Read values from the checksum-verified immutable audit snapshot, rather than processing raw motion again for each fold.
Project only participant identity, epoch time, and the sensor value before joining.
Never import audit target labels, predictions, stage summaries, or error flags into training inputs.
Match every existing eligible target exactly. Reject absent identities and reserved-test participants; do not silently remove rows.

## Missingness and scaling

Keep genuine observed zeros. An epoch without supported pairs has a missing value.
Fit its imputation mean on observed training targets only, separately for each fold.
Use that saved mean for unavailable training, validation, and later inference values.
Fit StandardScaler on training rows only. Mean-imputed motion values consequently map to approximately zero.
The current cohort requires no imputation; the policy still defines behavior for missing future inputs.
Missing motion and observed mean motion are intentionally indistinguishable in this first one-channel ablation.
Add neither an availability flag nor a support cutoff in this comparison.
The original six feature scales and fitted HR transform remain unchanged.
Record source hashes, missing/observed counts, imputation mean, feature order, and normalization in metrics and checkpoints.

## Fixed controls and necessary capacity change

Reuse the existing full-HR reference results. Run only five new candidate fits.
Keep the same participant folds, split seed 42, training seed 42, and 22,708 validation targets across all folds.
Keep context 30, P3 patches, width 16, two attention heads, one layer, feedforward width 32, and dropout 0.2.
Keep forty epochs, patience twelve, batch size 64, learning rate 0.001, gradient clip 1.0, and the original weighted cross-entropy.
Keep zero-decay Adam, no scheduler, no smoothing, no agent, and the existing diagnostics and macro-F1 checkpoint rule.
Require the same resolved device as the measured reference.

Adding one channel necessarily adds 3 × 16 = 48 input-projection weights: 2,852 becomes 2,900 parameters.
All other architecture settings remain fixed. Initialization and optimization trajectories can change with the new projection shape.
This is a feature-recipe comparison, not a capacity-matched proof of a physiological mechanism.

## Run locally

Open [notebook 36](../notebooks/36_fixed_lag_motion.ipynb). Every code cell states what it does and why.
The folds are already prepared in this checkout. Keep RUN_SETUP and RUN_PREPARATION false here.
Run the no-gradient checks. Enable RUN_TRAINING only when ready for the five new fits.
Existing results stop accidental retraining; retain their evidence.

For another machine, restore development features, the completed reference, and the verified motion audit snapshot with its configuration.
Clone the PR branch and install the locked groups through the notebook's optional setup cells.
The grouped preparation resolves the audit path for each fold and hashes the sensor source files.
Changes to those sources invalidate the prepared study before training starts.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/fixed-lag-motion-grouped.yaml --prepare

uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/fixed-lag-motion-grouped.yaml --train
```

Checkpoint construction restores the model and its semantic feature definition without access to the audit.
Replaying prepared six-column rows also needs the matching audit snapshot to supply the seventh input.
Replay uses stored imputation and scaling; it does not fit anything.
Future streaming callers must compute the same causal aggregate and provide it in the declared feature order.
This milestone does not introduce an inference server or a new raw streaming API.

## Evaluate after training

Use the explicit `motion_features` paired policy to reject unrelated changes.
Judge the mean paired fold macro-F1 difference first. Show all five folds and participant differences.
Also report pooled accuracy, per-stage precision/recall/F1, confusion matrices, NLL, and temporal slices.
Check whether improved Deep or REM recall costs Wake or Light performance, and whether benefits concentrate in a few participants.
Keep the original selected-checkpoint rule; do not select a favorable epoch from the final curves.
Treat one seed as screening evidence. Decide on seed replication after reviewing the complete result.
Keep the three final-test participants untouched. No candidate training runs automatically during HTML export.

## Completed-run device limitation, 2026-11-20

The saved candidate preparation manifest, fold recipes, and completed metrics all record CPU.
The reused reference records MPS. All other base recipe settings match.
The prior notebook checked the current automatic device selection, rather than the device frozen during preparation.
That check could pass while the already-prepared folds used another device.
The comparison rejected this mismatch correctly, but its generic error hid the specific differing key.

The existing checkpoints, metrics, predictions, and prepared inputs remain valid saved evidence.
Do not change their device fields or rerun training to complete reporting.
The paired validator remains strict by default. An explicit `allow_device_mismatch=True` exception is available only for the motion policy.
Notebook 36 enables that report-only exception for this completed run and displays the CPU/MPS fold table.
Paired and temporal JSON reports retain an `exploratory_device_mismatch` marker and interpretation caveat.
Other recipe, population, source, normalization, and capacity checks remain active.
This comparison cannot isolate the motion feature from numerical backend effects.

The training cell now passes the measured reference device to `validate_training_start`.
That guard checks the prepared manifest before any optimization. It does not permit device differences during new controlled training.
Keep RUN_TRAINING, RUN_PREPARATION, and RUN_CHECK false to finish this notebook from saved evidence.
Restart the kernel and run all report cells. The executed HTML already contains the completed report.
No new training or final-test observations are needed.
