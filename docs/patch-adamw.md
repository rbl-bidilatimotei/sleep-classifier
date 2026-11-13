# Patch Transformer weight-decay experiment

Status: prepared for local training by Timo. No participant-data AdamW results exist yet.

## Question and control

The [patch-size comparison](experiments/patch-size-comparison-2026-11-13/README.md) loses the primary score despite a small REM F1 gain.
Retain the replicated three-epoch patch model. Stop the local patch-size grid.
Its late training/validation NLL gap motivates one explicit parameter-regularization experiment.

Compare legacy Adam with grouped AdamW at weight decay 0.01.
AdamW applies decay separately from its gradient moments. See the [PyTorch AdamW definition](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html).
It does not add an L2 term to the logged cross-entropy loss.
Keep the validation loss definition unchanged. A lower NLL alone is not success when macro-F1 declines.

Before any candidate training, compare Adam with grouped zero-decay AdamW on synthetic data.
The reusable `ZeroDecayControl` runs twenty updates with active dropout and clipped gradients.
It compares losses, clipped gradients, updated parameters, and optimizer moments after each update.
It restores CPU, CUDA, and MPS random states when those backends are available.
No participant observations enter this control. It does not fit preprocessing.
The notebook runs it on CPU and the selected device. A failed check stops execution.
The same control is available through a reproducible command:

```bash
uv run --locked --group gru python -m sleep_classifier.optimizer_control \
  --config configs/transformer-patch-adamw.yaml --device mps \
  --output artifacts/transformer-patch-adamw-zero-control-cli.json
```

A failed control exits with a nonzero status.

The training cell repeats the selected-device check before launching the training command.

Both CPU and native MPS checks pass exactly on this checkout with PyTorch 2.15.0.
Every recorded maximum difference is zero. Each check uses twenty synthetic updates.
This is bounded numerical evidence, not a proof of identical participant-data trajectories.
Reusing the measured Adam reference is conditional on this check and the fixed recipe/device checks.
If the control fails on another environment, investigate before training. Do not silently reuse the reference.

## Fixed recipe

| Setting | Reused reference | Candidate |
| --- | --- | --- |
| Optimizer | Adam, zero decay | AdamW, decay 0.01 |
| Decay groups | Original single group | Decay / no_decay |
| Bias and LayerNorm decay | Zero | Zero |
| Initial rate / betas / epsilon | 0.001 / (0.9, 0.999) / 1e-8 | Same |
| Scheduler / OPAL | None / none | Same |
| Architecture | Three-epoch patch Transformer | Same |
| Raw context / features | 30 completed epochs / six features | Same |
| Width / heads / layers / feedforward | 16 / 2 / 1 / 32 | Same |
| Dropout / parameters | 0.2 / 2,756 | Same |
| Training seed / split seed | 42 / 42 | Same |
| Development folds | All five original folds | Same |
| Epoch cap / patience / batch size | 40 / 12 / 64 | Same |
| Gradient clipping | Norm 1.0 | Same |
| Loss / sampling | Unweighted cross-entropy / natural | Same |
| Checkpoint selection | Best validation macro-F1 | Same |
| Epoch diagnostics | Enabled with training evaluation | Same |

The policy exempts all bias parameters, including attention and recurrent biases, and all LayerNorm parameters.
It decays other trainable parameters, including the learned positional table.
Shared parameters enter one group only. Frozen parameters do not enter AdamW groups.
For this patch model, the decay group has 2,560 parameters and the exempt group has 196.
Preflight output and saved optimizer provenance list the exact names, counts, and effective decay for each group.
Do not substitute a blanket rule that decays every tensor or excludes every one-dimensional parameter.
This declared ownership policy is part of the experiment.

Reuse `artifacts/transformer-patch-grouped-validation/summary.json`.
Train only `configs/transformer-patch-adamw-grouped.yaml`.
The candidate writes to `artifacts/transformer-patch-adamw-grouped-validation`.
Maximum new participant-data work is five fits and 200 epochs. The three final-test participants remain reserved.

## Run locally

Open [notebook 26](../notebooks/26_patch_adamw.ipynb).
Read each What/Why explanation before its code cell.
Keep setup disabled in the current environment.
Run both synthetic controls and inspect their results.
Prepare the candidate once. Keep preparation disabled after its manifest exists.
Inspect the five-fold plan, effective parameter groups, and no-gradient checks.
Enable `RUN_TRAINING` when ready. Keep `RUN_PREPARATION` false after preparation.
The notebook never retrains the reference.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-adamw-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-adamw-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-patch-adamw-grouped-validation/fold-01/gru.yaml --check
```

The notebook checks all five folds. Its gated training cell repeats the synthetic device control first.
Existing result directories block accidental reruns. Automatic resume is unsupported.
Fresh clones also require the completed reference artifacts and prepared development features.
Keep the comparison on MPS to match this measured reference.

## Evaluation

Primary: compare paired mean fold macro-F1, then inspect all five fold differences.
Report participant differences, accuracy, Wake/Light/Deep/REM precision, recall, F1, support, and confusion matrices.
Inspect selected NLL and the gap between training-evaluation and validation NLL.
Keep checkpoint selection unchanged. Do not choose replacement checkpoints from the NLL curves.

Use `comparison="weight_decay"` for the paired and temporal reports.
This policy requires an unchanged model/base recipe and effective AdamW parameter-group evidence.
Other comparison policies reject optimizer changes.
Inspect short-history and transition slices, with their class support shown.
These slices remain exploratory and do not change eligibility or model inputs.

The hypothesis is that explicit decay reduces overfitting while preserving stage detection.
The previous lower-rate and scheduler experiments improved some loss behavior while losing classification scores.
Weight decay can also fail. Do not claim success from a smoother or lower loss curve alone.
A useful first-seed result requires replication before replacing the working development reference.
Repeated development-fold reuse cannot establish independent final-test performance.

After training, export the completed report:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/26_patch_adamw.ipynb
```

The fresh-kernel HTML export skips setup, preparation, participant training, and recursive export.
It can run the bounded synthetic control and no-gradient forward checks.
Keep source notebooks output-free. Store executed HTML and large run artifacts under ignored artifacts.
Small reviewed control evidence is recorded under docs/experiments.

## Preparation evidence

All five candidate folds were prepared on 2026-11-13 without participant-data optimization.
All 130 feature CSV files match the measured reference byte for byte.
The five resolved recipes differ only in the optimizer mapping. Both recipes resolve to MPS.
All five no-gradient checks pass with 2,756 parameters, raw inputs 64 × 30 × 6, and logits 64 × 4.
The notebook displays both synthetic controls and the effective decay groups.
Fresh-kernel HTML export completes with participant training and preparation skipped.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.

Validation: 621 tests pass without warnings. Ruff lint and whitespace checks pass.
