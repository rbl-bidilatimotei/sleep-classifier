# Compare two constant learning rates on the working motion recipe

Status: all five local fits are complete. LR0.0003 loses primary macro-F1 within forty epochs.

## Question

The [three-seed motion comparison](experiments/motion-mps-seeds-2026-11-20/README.md) supports motion as the working feature recipe.
The primary score improves at every seed, but stage trade-offs, partial-history weaknesses, and late probability overfitting remain.

Test learning rate 0.0003 against the completed 0.001 seed-42 motion reference on MPS.
The hypothesis is that smaller optimizer steps may improve the selected checkpoint's probability quality or stage trade-off within the same budget.
This is not a guaranteed overfitting cure or an asymptotic convergence comparison.
The older lower-rate experiment used an unpatched Transformer without the current feature recipe, so it does not answer this question.

## Fixed controls

Change only the constant learning rate and fresh output paths.
Reuse artifacts/fixed-lag-motion-mps-grouped-validation as the measured LR0.001 reference.
Write the new candidate to artifacts/motion-learning-rate-grouped-validation.
Keep both completed motion backends and all three seed pairs intact.

Retain seed 42, split seed 42, the same five participant folds, 26 development participants, and 22,708 eligible targets.
Retain explicit MPS, the exact motion source and definition, training-only imputation/scaling, both HR channels, and HR history ten.
Retain seven raw inputs, nine effective channels, thirty raw epochs, P3 patches, width sixteen, and 2,900 parameters.
Retain two attention heads, one layer, feedforward width 32, dropout 0.2, batch size 64, and gradient clip 1.0.
Retain zero-decay Adam, weighted cross-entropy, no smoothing, no scheduler, and no agent.
Retain forty epochs, patience twelve, identical diagnostics, and checkpoint selection by validation macro-F1.
Do not extend the budget or change checkpoint selection during the run.
Keep the three final-test participants untouched.

The compared models have identical capacity and feature schemas. No channel or projection weight is added in this comparison.
Each fold's saved motion provenance, support counts, imputation state, HR state, and model constructor must match exactly.
Motion-source hashes must also stay consistent across folds.

## Reused-reference validation and report policy

The reused reference has seven raw inputs and cannot use the legacy six-input self-comparison.
Notebook 39 verifies it through the completed no-motion-to-motion paired comparison before new work starts.
The new primary comparison then uses the existing learning_rate policy with fixed motion inputs on both sides.
The report records fixed_motion_inputs provenance. It does not present this as another feature-addition experiment.
Strict device matching remains active; a device exception is unavailable for the learning-rate policy.
The original no-motion learning-rate path remains supported.

## Initial local execution (completed)

Open [notebook 39](../notebooks/39_motion_learning_rate.ipynb). Restart the kernel before its first run.
Each code cell explains what it does and why. Source setup, preparation, checks, and training switches default to false.
The five folds are already prepared in this checkout. Keep RUN_SETUP and RUN_PREPARATION false here.
Enable RUN_CHECK for optional no-gradient checks. Enable RUN_TRAINING when ready for the five new local fits.
The training cell binds its command to the validated experiment and checks the prepared MPS device against the measured reference.
Existing results block accidental retraining.

A fresh checkout needs development features, the verified motion snapshot, the completed motion reference, and its original feature-comparison baseline artifacts.
Use the optional setup cells to clone the working branch and install locked uv groups.
Prepare once on an MPS-capable machine:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/motion-learning-rate-grouped.yaml --prepare
```

Train only when ready:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/motion-learning-rate-grouped.yaml --train
```

For report-only execution, keep all execution switches false. No candidate conclusion appears before all five folds finish.
HTML export skips setup, preparation, training, and recursive export. Reporting does not optimize a model.

## Evaluate after training

Use mean paired fold macro-F1 as the primary measure. Show every fold and participant, including regressions.
Inspect selected pooled NLL, per-stage precision/recall/F1, error destinations, and temporal slices as secondary evidence.
Check whether Deep's trade-off or probability quality improves without recurring costs to Wake, Light, or startup targets.
Distinguish lower training loss, better validation probabilities, and better predicted labels; they are different outcomes.
Keep the original macro-F1 checkpoint rule rather than selecting a lower-NLL epoch after inspection.

Record selected and final epochs. If the lower rate reaches the forty-epoch cap, report the fixed-budget limitation.
That result alone cannot establish which rate would converge better with additional compute.
A promising result would need further seed replication before becoming the default optimizer setting.
Timo runs all new training locally. This preparation performs no participant-data optimization.

## Measured findings

The [completed comparison](experiments/motion-learning-rate-2026-11-20/README.md) loses mean fold macro-F1 by 0.00767.
Light gains recall and a little F1, while Wake, Deep, and REM F1 decline. Selected pooled NLL improves by 0.00209.
Three candidate folds reach the cap and select epoch 40 or 39. Keep LR0.001 as the working rate.
Recommend a separate eighty-epoch LR0.0003 budget comparison with patience twelve unchanged. No new run is prepared here.
For completed reporting, keep all execution switches false and restart the kernel.
