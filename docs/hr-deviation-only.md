# Isolate availability with a deviation-only control

Status: completed locally by Timo. See the [measured findings](experiments/hr-deviation-only-2026-11-18/README.md).

## Question

The [replicated full-HR recipe](experiments/window-local-hr-seeds-2026-11-18/README.md) improves mean fold macro-F1 at three seeds.
The [availability-only control](experiments/hr-availability-control-2026-11-18/README.md) loses macro-F1 and accuracy on every fold after deviation is zeroed.
This supports explicit deviation conditional on availability and the seed-42 training procedure.
It does not establish whether the availability flag is useful alongside deviation.
Test that complementary question before adding features or changing the architecture.

## Intervention

| Setting | Reused full-HR reference | Deviation-only control |
| --- | --- | --- |
| Original six features | Retain | Retain |
| Standardized HR deviation | Retain | Retain |
| Explicit history availability | Retain | Set to zero |
| Raw/effective channels | 6 / 8 | Same |
| Nominal parameter count | 2,852 | Same |
| Initial weights and parameter ordering | Full-HR layout | Same |
| Training-only normalization | Full-HR fitting procedure | Same values |

Declare `window_features.masked_channels: [heart_history_available]`.
Apply the mask after scaling and missing-history handling, before patch projection.
Unavailable deviations still become zero. Available deviations retain their original standardized values.
The control loses the explicit flag distinguishing unavailable history from an ordinary standardized zero.
The model still receives valid lengths and positional information, so that flag remains derivable from existing inputs.
This experiment tests its explicit representation, rather than the addition of new sensor information.

Retain the zeroed channel rather than deleting it, preserving initialization draws and parameter layout.
Its projection weights become inactive; matching nominal size does not imply equal functional capacity.
The existing deterministic transform and checkpoint restoration already support this mask.
The paired policy now permits either single-channel ablation and records the actual masked channel.
It rejects both-channel masks, masked references, and different masks across folds.
The earlier availability-only recipe remains valid and distinguishable.

## Fixed recipe and budget

Reuse the completed full-HR seed-42 result. Do not retrain it or use the availability-only result as the reference.
Train five fresh controls across the same participant folds, capped at 200 total epochs.
Keep training seed 42, split seed 42, all 26 development participants, and the three reserved final-test participants.

Keep raw context thirty, previous-history limit ten, and minimum-history requirement four.
Keep P3, width 16, two heads, one layer, feedforward width 32, and dropout 0.2.
Keep original Adam 0.001, no scheduler or agent, unsmoothed unweighted cross-entropy, and natural sampling.
Keep forty epochs, patience twelve, batch 64, norm clipping one, and validation macro-F1 checkpoint selection.
Preserve diagnostics, prediction artifacts, intermediate checkpoints, and the MPS device.
Fit both normalization stages on training participants only, even though the flag is masked.
All eligible targets remain present.

## Run locally

Open [notebook 32](../notebooks/32_hr_deviation_only.ipynb).
Read each What/Why explanation. Inspect the reused reference, exact mask, work budget, and five no-gradient checks.
Keep setup disabled in the current environment.
Keep preparation disabled once the control manifest exists.
Set RUN_TRAINING to true when ready. Run the training cell interactively.
The cell trains only the five new controls.

Fresh clones require development features and the completed artifacts/transformer-window-local-hr-grouped-validation reference.
Use MPS to match that reference. Source notebooks default to training disabled.
Existing output directories block accidental preparation or training reruns. Automatic resume is unsupported.

Prepare only if the output directory does not exist:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/hr-deviation-only-grouped.yaml --prepare
```

Check each prepared fold through the notebook. For one fold:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/hr-deviation-only-grouped-validation/fold-01/gru.yaml --check
```

Train only through the gated cell or this explicit command:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/hr-deviation-only-grouped.yaml --train
```

## Evaluation and decision

Use the explicit window_feature_ablation policy with the availability mask recorded in the report.
Require identical fitted scaling, parameter counts, channel names, base recipe, raw inputs, fold population, budgets, and device.
The reporter rejects source/metric corruption and different ablations across folds.
Existing saved reports remain historical artifacts; a rebuilt report can include additional mask metadata without changing measured scores.

Primary: candidate-minus-full-HR mean fold macro-F1, with all fold and participant differences.
Inspect each stage's precision, recall, F1, support, confusion matrix, selected NLL, and evaluation loss curves.
Check whether REM gains survive while Wake and stage-transition behavior improve or regress.
Use the same history slices: 1–4, 5–29, and full thirty-epoch context.
History bins are disjoint; transitions overlap them. Do not average their macro-F1 values to reconstruct pooled scores.

A clear regression supports retaining the flag conditional on deviation and this training procedure.
A gain would support replication before promoting the simpler representation.
A small change would not prove equivalence or that availability is irrelevant.
Keep the full HR recipe as the working reference until the comparison is reviewed.
Repeated development selection is adaptive exploration; keep the final test untouched.

Export the HTML report without training:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/32_hr_deviation_only.ipynb
```

Export skips setup, preparation, training, and recursive export. Keep source outputs clear and generated HTML under ignored artifacts/notebooks.

## Preparation evidence

The [preparation snapshot](experiments/hr-deviation-only-preparation-2026-11-18/README.md) records five MPS preflights and synthetic CPU/MPS checks.
All 919 tests, Ruff, whitespace checks, and fresh-kernel report export pass.
That preparation snapshot predates Timo’s completed run. Reuse the measured artifacts rather than retraining them.

## Training identity safeguard

See the [blocked-command diagnosis](notebook-training-identity.md) before retrying the reported run.
Notebook 31 targets the completed availability-only experiment; notebook 32 targets the fresh deviation-only experiment.
Commands now derive their config from the validated experiment, log its identity, and check readiness before starting the CLI.
Reload notebook 32, restart its kernel, and run its setup cells from the top before enabling training.

## Measured outcome

Removing availability reduces mean fold macro-F1 by 0.00577, with four declining folds.
REM and Deep recall rise, but precision falls; their F1 scores decline. Light loses 326 correct predictions.
Retain both channels for now. Repeat the same ablation with seeds 43 and 44 before interpreting the smaller, mixed flag effect.
