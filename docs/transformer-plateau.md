# Plateau-scheduled learning-rate diagnostic

Status: completed locally by Timo.
The [measured comparison](experiments/transformer-plateau-2026-11-12/README.md) improves NLL and smoothness but loses macro-F1.
Retain the constant-0.001 recipe as the stage-classification reference.

## Why this experiment?

The [constant-rate comparison](experiments/transformer-learning-rate-2026-11-12/README.md) lowered pooled NLL but lost mean fold macro-F1 and REM/Deep recall.
Constant 0.0003 also fit more slowly within the fixed budget.
Retain the stronger stage-classification reference at 0.001.
Test whether faster initial learning followed by smaller plateau updates improves the trade-off.
This is a hypothesis, not a promised overfitting cure.

## Fixed protocol

Reuse artifacts/transformer-diagnostics: seed 42, original folds 01 and 04, and constant Adam 0.001.
Train only the scheduled candidate. The new budget is two fits and at most eighty total epochs.
Keep model, features, participant splits, device, dropout 0.2, natural sampling, unweighted cross-entropy, batch 64, clipping one, and diagnostics fixed.
Keep forty epochs, stopping patience twelve, and validation macro-F1 checkpoint selection.
The three final-test participants remain unused. OPAL stays disabled.

| Scheduler control | Fixed setting | Meaning |
| --- | --- | --- |
| Kind | ReduceLROnPlateau | Reduce the rate after the monitored score stops improving |
| Initial learning rate | 0.001 | Preserve the stronger reference's initial fitting behavior |
| Monitor / mode | validation_macro_f1 / max | Higher stage-balanced classification score is better |
| Factor | 0.3 | Multiply each current rate by 0.3 on a qualifying reduction |
| Scheduler patience | 3 | Reduce after four consecutive non-improving scheduler evaluations |
| Threshold / mode | 0.0001 / abs | Improvement must exceed the scheduler's best score by this amount |
| Cooldown | 0 | Resume plateau counting without a cooldown interval |
| Minimum rate | 0.00003 | Stop reducing below this floor |
| Epsilon | 0.00000001 | Ignore numerically negligible changes in the rate |

PyTorch defines the patience and threshold behavior above.
See [ReduceLROnPlateau documentation](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
The initial best evaluation establishes the scheduler baseline; it is not a bad epoch.
The first reduction changes 0.001 to 0.0003. Later reductions retain the same factor and floor.

Scheduler patience is independent of early-stopping patience.
The scheduler must not reset, extend, or replace the stopping mechanism.
Checkpoint selection still accepts any strict macro-F1 improvement. The scheduler uses its stated threshold.
A tiny positive improvement can therefore select a checkpoint without resetting the scheduler plateau count.
Do not change that distinction after inspecting candidate results.

## Rate timing and evidence

Each epoch trains using its recorded actual optimizer rates.
The existing validation pass produces its macro-F1, and diagnostics capture the rates used for that completed epoch.
The scheduler then evaluates that score and can set a smaller rate for the next epoch.
A reduction after epoch k cannot explain the training updates or predictions already produced during epoch k.

Enabled schedules add learning_rates_used, learning_rates_next_epoch, and scheduler_reduced to the saved training history.
The next-epoch field is absent in effect, represented by None, when the cap or stopping rule ends training.
No scheduling step is needed after a final or stopped epoch.
The existing per-epoch evidence schema continues to record actual rates used, preserving the meaning of its rate plots.

Inspect reductions before interpreting the experiment. A schedule that never changes a rate has not tested a lower-rate phase.
Read reference/candidate class trade-offs beside NLL and confidence.
Intermediate checkpoints are evaluation snapshots, not exact optimizer/scheduler/RNG resume states.
The grouped runner still refuses existing results. Automatic training resume remains unsupported.

## Run locally

Open [notebook 22](../notebooks/22_transformer_plateau.ipynb).
Read the What/Why explanation before each code cell.
Verify the completed constant-0.001 reference, matching MPS device, original fold assignments, and fixed controls.
Keep setup disabled for the existing environment.
Prepare the candidate only once, then run both no-gradient forward checks.
Enable RUN_TRAINING when ready. Leave RUN_PREPARATION false once its manifest exists.

Fresh clones need the development features and completed instrumented constant-0.001 reference artifacts.
The completed constant-0.0003 experiment is context for this study; it is not the primary paired reference.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-plateau-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-plateau-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-plateau/fold-01/gru.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-plateau/fold-04/gru.yaml --check
```

Start optimization only through the explicit command or gated notebook cell:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-plateau-grouped.yaml --train
```

After both fits complete:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/transformer-diagnostics/summary.json \
  --candidate artifacts/transformer-plateau/summary.json \
  --comparison scheduler --output artifacts/transformer-plateau/paired-comparison.json
uv run --locked --group gru --group notebook python -m sleep_classifier.experiment_evidence \
  --output-dir artifacts/transformer-plateau --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/22_transformer_plateau.ipynb
```

HTML export skips setup, preparation, training, and recursive export. It never starts optimization.
Keep source notebooks output-free and executed reports as local HTML.

## Decision and limits

Compare selected fold macro-F1 first, then Wake/Light/Deep/REM precision, recall, and F1.
Inspect selected NLL, training progress, loss swings, confidence, and actual learning-rate changes as secondary evidence.
Require that the rate reductions occur early enough to affect later trained epochs.
Do not judge success from smoother loss, later selected checkpoints, or fewer confident errors alone.

The scheduler uses validation feedback in addition to validation checkpoint selection.
These folds therefore remain development data, not an independent generalization estimate.
Their eleven participants form a post-hoc diagnostic subset; a useful result requires broader validation before global adoption.
Do not add AdamW, weight decay, label smoothing, augmentation, extra epochs, or a new architecture to this comparison.
If the bounded scheduling test gives little benefit, return to the planned causal patch encoder rather than expanding the optimizer grid.

## Where regularization fits

A penalty such as L2 changes the objective. AdamW applies weight decay separately from the adaptive loss-gradient update.
For Adam, those are different operations; see the [decoupled weight decay paper](https://arxiv.org/abs/1711.05101).
Dropout deliberately introduces stochastic regularization without adding a written penalty term.
Early stopping restricts fitting time. Architecture, capacity, data diversity, and optimization bias also affect generalization.
A scheduler changes step sizes; it does not impose a new loss penalty or guarantee improved generalization.
Keep each mechanism's test separate so its effects remain attributable.

## Historical preparation checks

These checks describe the state before Timo ran training. The measured review supersedes pending-result status.

On 2026-11-12, the two selected folds were prepared without model optimization.
All 52 copied feature CSVs match the completed constant-0.001 reference byte for byte.
Resolved fold recipes differ only by adding lr_scheduler.
Both MPS no-gradient forward checks passed with 2,884 parameters and 64 × 30 × 6 input windows.
Notebook 22 exported from a fresh kernel with optimization skipped; candidate results and scheduler history remain pending.
All 458 tests, Ruff lint, and whitespace checks pass.
On this checkout, leave RUN_PREPARATION false and enable RUN_TRAINING when ready.
