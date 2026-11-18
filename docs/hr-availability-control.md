# Isolate HR deviation with an availability-only control

Status: prepared for local training by Timo. No control scores exist yet.

## Question

The [three-seed replication](experiments/window-local-hr-seeds-2026-11-18/README.md) supports the combined HR feature recipe as the working development reference.
Mean paired seed macro-F1 improves by 0.01834. REM improves at all seeds; Wake F1 declines at all seeds.
That comparison changes two channels and the original projection layout together.
It cannot isolate the value of deviation beyond the availability cue.

The availability flag also identifies positions with enough earlier observations inside the supplied window.
It can describe segment startup and relative window position, even though it contains no new sensor measurements.
Test whether HR-deviation values add value when this cue and the model layout remain fixed.

## Intervention

| Setting | Reused full-HR reference | Availability-only control |
| --- | --- | --- |
| Original six channels | Retain | Retain |
| HR deviation | Compute and standardize | Compute, standardize, then replace with zero |
| History availability | Binary flag after four earlier epochs | Same |
| Raw/effective channels | 6 / 8 | Same |
| Parameters | 2,852 | Same nominal count |
| Parameter ordering and initial draws | Original full-HR layout | Same |
| Normalization fit | Training targets only | Same inputs and fitted state |
| Context/history requirement | Thirty raw epochs / up to ten previous / minimum four | Same |

The control declares `window_features.masked_channels: [heart_rate_delta_past]`.
The deterministic transform applies the mask after scaling and unavailable-history masking, before patch projection.
The flag remains unchanged. The original HR summaries remain available, so the model can still learn past-relative patterns itself.
No observations or eligible targets are removed.

Retain the masked input dimension rather than deleting it.
Deleting it would change projection size, parameter ordering, and initialization draws.
Zeroing the channel keeps those construction details fixed. Its projection weights become inactive, so nominal size does not imply equal functional capacity.
Checkpoint metadata records the mask and restores it for inference and replay.
Legacy full-HR recipes without the optional key retain their existing settings, model state, and random initialization.

## Fixed controls and budget

Reuse the completed seed-42 full-HR grouped result. Do not retrain it.
Train five fresh masked candidates on the same five development folds, capped at 200 total epochs.
Keep seed 42, split seed 42, the 26 development participants, and all three final-test participants fixed.

Retain P3, width 16, two heads, one layer, feedforward width 32, and dropout 0.2.
Retain original Adam 0.001, no scheduler or agent, unsmoothed unweighted cross-entropy, and natural sampling.
Retain forty epochs, patience twelve, batch 64, norm clipping one, and macro-F1 checkpoint selection.
Keep per-epoch evidence, intermediate checkpoints, and MPS unchanged.
The only recipe differences are the explicit mask and output paths.

## Run locally

Open [notebook 31](../notebooks/31_hr_availability_control.ipynb).
Read each What/Why explanation. Inspect the reused reference, fold plan, mask, and five no-gradient checks.
Keep setup disabled in the current environment.
Keep preparation disabled after the candidate manifest exists.
Set RUN_TRAINING to true when ready. Execute the training cell interactively.
The reference is never trained by that cell.

Fresh clones need development feature tables and the completed artifacts/transformer-window-local-hr-grouped-validation reference.
Use MPS to match the measured reference. The source notebook defaults to training disabled.
Existing output directories block accidental re-preparation or retraining. Automatic resume is unsupported.

Prepare only when its output directory does not exist:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/hr-availability-control-grouped.yaml --prepare
```

Run the prepared-fold checks through the notebook. For one fold:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/hr-availability-control-grouped-validation/fold-01/gru.yaml --check
```

Train only through the gated cell or this explicit command:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/hr-availability-control-grouped.yaml --train
```

## Evaluation and interpretation

Use the explicit window_feature_ablation paired policy.
Require identical base recipes, model layout, effective channel names, parameter counts, fitted normalization, device, populations, and coverage.
Only the declared deviation mask may differ.
The existing source, metric-hash, prediction-score, and budget checks remain active.

Report candidate minus full-HR reference scores. Negative differences favor retaining deviation.
Primary: mean paired fold macro-F1, with every fold and participant difference.
Also inspect Wake/Light/Deep/REM precision, recall, F1, support, confusion matrices, selected NLL, and saved loss curves.
Retain history bins 1–4, 5–29, full context, and the stage-transition slice.
History bins are disjoint; transitions overlap them. Do not average slice macro-F1 scores to reconstruct pooled scores.

If zeroing deviation reduces performance, the values help conditional on availability and this training procedure.
A near-neutral result would motivate checking whether the availability cue or construction changes explain much of the earlier gain.
It would not prove equivalence or that past-relative HR information is physiologically irrelevant.
The original HR sequence still contains that information, and this is one training seed on repeatedly used development participants.
Do not replace the working reference from a single fold or NLL alone.
Keep the final test untouched.

Export the executed report:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/31_hr_availability_control.ipynb
```

Export skips setup, preparation, training, and recursive export. Keep source outputs clear and generated HTML under ignored artifacts/notebooks.

## Preparation evidence

The [preparation snapshot](experiments/hr-availability-preparation-2026-11-18/README.md) records five MPS checks and synthetic CPU/MPS initialization checks.
All 876 tests, Ruff, whitespace checks, and fresh-kernel report export pass.
No control training ran. Keep RUN_PREPARATION false in the current checkout; set RUN_TRAINING true when ready.
