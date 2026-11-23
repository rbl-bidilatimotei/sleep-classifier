# Chronological inference parity

## Why this experiment comes next

Notebook 40's longer training budget partly closes the lower-rate gap. It does not beat the working recipe on mean fold macro-F1.
Pause small optimizer and seed sweeps. Retain the LR0.001 motion checkpoints and test the intended inference behavior.
The next question has a clear acceptance condition: can chronological inference reproduce offline validation?

This milestone adds no model capacity and trains no weights. It does not aim to raise accuracy.
Repeated development-fold tuning increases selection exposure. Another small score gain alone would provide limited new evidence.
These results do not establish a performance ceiling. Larger modeling changes require a specific hypothesis and a fixed evaluation plan.

## Fixed scope

Use all five saved seed-42 MPS folds from configs/fixed-lag-motion-mps-grouped.yaml.
Retain both HR channels, fixed-lag motion, context thirty, P3 patches, width sixteen, and dropout 0.2.
Use the checkpoint's stored scaler and HR transform. Do not fit preprocessing again.
Verify the frozen fold manifest, registered metrics, selected weights, and saved prediction hashes.
Keep all three final-test participants reserved. Use no final-test observations.

Replay only causally prepared, eligible validation summaries. The model receives no target labels.
The audit uses labels only to match each input row with its saved validation record.
Unavailable motion values must already use the checkpoint's saved imputation policy through the existing motion adapter.
The streaming predictor rejects missing or non-finite values rather than inventing a replacement policy.

This is a prepared-feature interface. It does not yet accept raw accelerometer or heart-rate events.
Raw sensor aggregation, support thresholds, unavailable epochs, and real acquisition latency need a later end-to-end audit.
A row represents a completed thirty-second epoch. Its saved timestamp retains the existing epoch convention.
Do not interpret its output as a forecast before that epoch has completed.

## Inference contract

StreamingEpochPredictor loads a selected checkpoint with weights_only enabled.
Supply participant_id, segment_id, the epoch timestamp, and the exact named raw feature summaries.
Here, raw means original feature units before scaling. It does not mean raw sensor samples.
Return the four-class probabilities, predicted class, retained history length, and any reset reason.
Class indices remain wake 0, light 1, deep 2, and REM 3.

The predictor retains at most context_epochs rows. It recomputes the retained window with the existing model.
It resets history on participant changes, segment changes, or timestamp gaps.
It rejects duplicate or backward timestamps within the active participant. Call reset() for an explicit new session.
Use one predictor per active stream. This object does not manage concurrent users or a network service.
The existing window-local HR transform stays inside the restored model and uses only the retained window.

## Acceptance conditions

Compare chronological probabilities with the existing offline batch replay on the same backend.
Also compare offline replay with the saved validation probabilities.
Both comparisons require maximum absolute difference at most 0.000002 and zero predicted-class disagreements.
This matches the existing checkpoint replay tolerance; do not relax it after a failure without investigation.
Near-tied probabilities can change argmax while remaining close. The audit reports that discrepancy as a failure.

Every history length must match offline participant/segment boundaries, including startup and context eviction.
Model parameters and buffers must remain unchanged. No gradients may be created.
Synthetic tests also cover future-prefix independence, missing inputs, invalid order, and explicit or automatic resets.
Passing these checks demonstrates implementation parity within this scope, not improved model quality or clinical validity.

Record per-epoch probabilities, reset reasons, history lengths, and approximate processing duration.
Python p50/p95 timings are descriptive. They include local runtime overhead and do not establish a deployment latency guarantee.

## Run locally

Open notebooks/41_chronological_inference.ipynb. Restart the kernel and execute cells in order.
Every code cell explains what it does and why it exists.
All switches default to false. Keep RUN_SETUP false in the existing environment.
Enable RUN_AUDIT when ready. This runs inference only; there is no training switch.

```bash
uv run --locked --group gru --group notebook python -u \
  -m sleep_classifier.chronological_replay --config configs/chronological-inference.yaml --run
```

The output is artifacts/chronological-inference/report.json with one prediction CSV per fold.
An existing verified report is reused by the notebook. The CLI refuses to overwrite an existing snapshot.
Keep partial or failed snapshots. Inspect their evidence before choosing a separate destination for a repeat.
The report reader verifies current source identities and saved prediction hashes before displaying cached evidence.
Report-only HTML execution skips the optional inference cell and shows pending status until the full audit is run.

A bounded MPS preflight checked the first thirty-one fold-01 epochs, including first context eviction.
Its maximum streaming/offline difference was 1.79e-7, with zero predicted-class disagreements.
This is not the full audit. See [preflight evidence](experiments/chronological-inference-preflight-2026-11-23/README.md).

## Next decision

If parity fails, inspect the mismatched epochs before changing architecture or numerical tolerances.
If parity passes, consolidate the model card, inference contract, and measured limitations.
Use a new modeling hypothesis before reopening training. Reserve the final test for an explicitly frozen model-selection decision.
Docker, FastAPI, and CI remain later engineering milestones; this PR adds no service or deployment pipeline.

## Completed audit: 2026-11-23

The [full audit](experiments/chronological-inference-2026-11-23/README.md) passed on all five folds and 22,708 development validation epochs.
The maximum probability difference is 8.94e-7, below 2e-6. There are zero class disagreements.
Offline probabilities match the saved validation probabilities exactly. All histories and reset events match the prepared segmentation.
Weights and buffers remain unchanged; no gradients or training occurred.
Keep RUN_AUDIT false for report inspection. Existing snapshots are verified and reused rather than overwritten.

The next work chunk should consolidate the model card and versioned inference-bundle contract.
Specify the feature semantics, availability, missing-data rules, saved transforms, reset behavior, and model identity before adding a service.
Raw-sensor adapter parity remains a separate later audit. Keep all final-test observations reserved.
