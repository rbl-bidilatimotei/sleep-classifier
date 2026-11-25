# Frozen-model replay with sensor-only eligibility and resets

Status: full five-fold replay verified on 2026-11-25. No model was trained or changed.
Read the [measured findings](experiments/sensor-only-replay-2026-11-25/README.md).
Start with [notebook 45](../notebooks/45_sensor_only_frozen_replay.ipynb).

## Question

Does removing label-dependent eligibility and resets change predictions from the frozen working models?
The [preparation audit](label-free-sensor-audit.md) found matching numerical features but 566 changed shared contexts.
This replay measures the prediction consequences without changing model capacity, weights, fitted preprocessing, or backend.
It is an input-policy experiment. It does not train another model or tune a recipe.

## Fixed inputs

Use the five seed-42 checkpoints from the working fixed-lag motion study.
Each checkpoint predicts only its original held-out development participants.
The passed chronological audit verifies checkpoint identities, original probabilities, and legacy-reset behavior.
The passed sensor preparation snapshot supplies seven raw channels, coverage decisions, and sensor-only segments.
The full-study configuration is [sensor-only-replay.yaml](../configs/sensor-only-replay.yaml).
Its paths resolve relative to the YAML file. MPS stays explicit and must match the measured reference backend.

The source guard verifies the frozen fold plan before any model restoration.
It rejects changed source hashes, different development membership, unexpected prepared files, and reserved-test observations.
Candidate rows must contain exactly the development cohort before the held-out fold projection.
No raw sensors or raw labels are reopened by this replay.

Original research checkpoints remain the weight source. Their digests must match the passed chronological audit and registered fold metadata.
The evaluation-bundle loader retains its exact-runtime requirement. This workflow does not change that contract or publish another bundle.

## Prediction and scoring order

For each fold:

1. Verify its checkpoint and restore the saved model in evaluation mode.
2. Project participant, time, sensor segment, sensor eligibility, and seven raw channels.
3. Select sensor-eligible rows without consulting targets or reference eligibility.
4. Replace unavailable fixed-lag motion with that fold's saved training mean.
5. Apply the saved scaler and window-local heart-rate transformation through the streaming predictor.
6. Predict every candidate epoch in chronological order with gradients disabled.
7. Open the labeled research control after the candidate pass completes.
8. Reset the stream and reproduce every saved legacy-control prediction.
9. Join completed candidate predictions to scored control targets by participant and exact epoch time.
10. Compare the model state and fitted metadata before publication.

Candidate prediction files contain no target column. Reference predictions and targets remain in a separate file.
The stream retains at most thirty epochs. Sensor gaps and sensor-derived segment changes reset its history.
Covered unscored epochs stay in candidate history. Sleep-label availability never drives candidate selection or resets.
The saved imputation mean, scaler, heart-rate transform, constructor settings, weights, and buffers remain unchanged.
No fit, backward pass, or optimizer step is called.

## Controls and metrics

The legacy-reset stream must reproduce all saved probabilities within absolute tolerance 2e-6, with zero class disagreements.
History lengths and reset reasons must also match. Invalid probability vectors cannot pass the numerical control.
Every fold needs unchanged weights and fitted metadata, evaluation mode, and no gradients.
Those conditions define a passed run. Candidate accuracy has no acceptance threshold.

The main comparison scores both policies on exactly the same shared target identities.
Report accuracy, fixed-four-class macro-F1, per-stage precision/recall/F1, confusion matrices, and unsmoothed validation NLL.
NLL averages the negative log probability assigned to the true stage. Lower values indicate better true-class probability assignments.
The metric clips probabilities below 1e-12 for numerical safety. It contains no class weighting or label smoothing.

The report also records:

- All candidate predictions, including epochs without a scored reference target
- Every reference-only target excluded by the conservative session-tail boundary
- Reference metrics on its original complete eligible population, labeled separately from paired metrics
- Fold and participant paired scores, class disagreements, probability differences, and history-length changes

Do not score extra candidate epochs as Light or assign them invented targets.
Do not compare full-reference accuracy with candidate accuracy on a smaller intersection.
A single-fold preflight cannot establish the five-fold policy effect.

## Run locally

Install the locked environment:

```bash
uv sync --locked --inexact --group gru --group notebook --group dev
```

Inspect the verified source plan without restoring model weights:

```bash
uv run --locked --inexact --group gru python -m sleep_classifier.sensor_only_replay \
  --config configs/sensor-only-replay.yaml
```

Open notebook 45. Keep RUN_SETUP false when the local environment is ready.
Set RUN_REPLAY true and execute the cells in order.
The notebook validates readiness before launching this command:

```bash
uv run --locked --inexact --group gru --group notebook python -u -m sleep_classifier.sensor_only_replay \
  --config configs/sensor-only-replay.yaml --run
```

Progress logs show fold, policy, and completed epoch counts. The source experiment and its checkpoints are never overwritten.
The output directory is `artifacts/sensor-only-frozen-replay`.
Publication stages predictions, report, and manifest, then uses an exclusive atomic rename.
A failed control publishes no complete snapshot. Existing output identifiers cannot be replaced.
Notebook reruns inspect and verify an existing snapshot instead of rerunning inference.
Choose a fresh configuration and output path for an intentional repeat.

The manifest identifies output and implementation bytes. The report pins the sensor snapshot, chronological evidence, checkpoint hashes, and frozen folds.
Cached inspection verifies hashes, current source identities, fold completeness, control parity, integrity flags, and pooled scope totals.
It does not run another inference pass. A new clone needs the existing research and preparation artifacts.

Export the report-only HTML:

```bash
uv run --locked --inexact --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/45_sensor_only_frozen_replay.ipynb
```

The report appears at `artifacts/notebooks/45_sensor_only_frozen_replay.executed.html`.
Setup, replay, and recursive export cells are skipped. Source notebook flags default to false and outputs stay empty.
Tables, shared-scale confusion matrices, and paired participant plots appear after the complete replay passes.

## Verification and limits

The [fold-01 preflight](experiments/sensor-only-replay-preflight-2026-11-25/README.md) first verified execution on MPS.
The subsequent [full replay](experiments/sensor-only-replay-2026-11-25/README.md) passed all five controls exactly.
The default full-study destination is now occupied by the verified immutable snapshot.
Keep RUN_REPLAY false to inspect it. Notebook reruns verify and reuse the saved outputs.
Synthetic tests cover label-independent projection, saved imputation, causal history, control rejection, publication, cached evidence, and notebook gates.
Training and optimizer tests are excluded from this milestone's targeted checks.

This replay starts from verified cached sensor summaries. It does not implement live packet arrival, clock synchronization, or a service endpoint.
The declared origin and conservative tail policy remain those from notebook 44.
Keep the final test reserved. Review the full five-fold results before selecting an inference policy or planning another model experiment.

Validation completed with 425 targeted non-training tests, Ruff, and whitespace checks.
Notebook 45 passed fresh-kernel report-only execution. Synthetic completed-report checks also rendered both chart types and the participant table.

The completed replay changes 177 class decisions within 566 changed histories, with 29 additional correct decisions overall.
Paired macro-F1 rises by 0.003528, while REM F1 falls slightly. These changes are heterogeneous across folds and participants.
Next, audit a bounded raw-event adapter with explicit completion and session-end semantics before running another model sweep.

The [bounded raw-event adapter](raw-event-adapter.md) is now prepared in notebook 46, with a bounded prefix preflight.
Full development feature and fixed-prediction parity remain opt-in before live-source or service work.
