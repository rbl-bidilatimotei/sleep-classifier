# Training-only label smoothing

Status: completed locally by Timo and reviewed on 2026-11-16.
The [measured comparison](experiments/patch-label-smoothing-2026-11-16/README.md) improves accuracy and selected NLL but loses mean fold macro-F1.
Retain the original unsmoothed Adam/P3 reference.

## Why this intervention?

The [AdamW comparison](experiments/patch-adamw-2026-11-16/README.md) changes mean fold macro-F1 by only +0.00075.
Most selected NLL improvement comes from one fold selecting an earlier checkpoint.
Retain the original Adam, three-epoch patch model as the working reference.
Its persistent probability-overfitting pattern motivates one controlled loss comparison.

Test cross-entropy with label smoothing 0.05 during training only.
The [PyTorch definition](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html) mixes the original target with a uniform class distribution.
For four classes, the training target assigns 0.9625 to the true class and 0.0125 to each other class.
This changes target probabilities. It does not resample epochs, rebalance class frequencies, or change eligibility.
Ground-truth validation labels remain unchanged.

The hypothesis is that modest target softening reduces overconfident errors without losing stage detection.
The [original label-smoothing study](https://arxiv.org/abs/1906.02629) motivates this test but provides no dataset-specific guarantee.
We have not established that incorrect labels cause our errors. Smoothing is not proof of label noise.

## Fixed comparison

| Setting | Reused reference | Candidate |
| --- | --- | --- |
| Training loss | Cross-entropy, smoothing 0 | Cross-entropy, smoothing 0.05 |
| Validation / training-evaluation NLL | Hard-label cross-entropy | Same hard-label cross-entropy |
| Optimizer / decay | Original Adam / zero | Same |
| Model / parameters | Three-epoch patch Transformer / 2,756 | Same |
| Raw history / features | 30 completed epochs / six features | Same |
| Width / heads / layers / feedforward | 16 / 2 / 1 / 32 | Same |
| Dropout / learning rate | 0.2 / 0.001 | Same |
| Epoch cap / patience / batch size | 40 / 12 / 64 | Same |
| Seed / fold split seed | 42 / 42 | Same |
| Folds | All five original development folds | Same |
| Sampling / class weighting | Natural / none | Same |
| Clipping / scheduler / OPAL | Norm 1.0 / none / none | Same |
| Diagnostics / checkpoint selection | Enabled / best validation macro-F1 | Same |

Do not combine smoothing with AdamW, EMA, a scheduler, or changed model dimensions in this experiment.
The zero-smoothing path must retain the legacy training behavior.
Reuse `artifacts/transformer-patch-grouped-validation/summary.json`.
The candidate uses `configs/transformer-patch-label-smoothing-grouped.yaml`.
Its artifacts go to `artifacts/transformer-patch-label-smoothing-grouped-validation`.
Only five fresh fits are needed, with a maximum of 200 epochs. The final-test participants remain reserved.

## Loss interpretation

The smoothed training objective is not ordinary hard-label NLL.
Report it under an explicit training-objective label.
Continue computing validation loss and fixed-weight training-evaluation NLL from hard targets.
Those two evaluation-mode NLL curves remain directly comparable in definition.
Do not interpret the gap between the smoothed stochastic objective and hard-label NLL as an ordinary generalization gap.
Dropout and changing within-epoch weights already affect the stochastic curve; smoothing adds a different loss definition.

The paired report uses `comparison="label_smoothing"`.
It requires the same model, optimizer, rate, diagnostics, and base recipe, with only training smoothing allowed to differ.
Effective loss provenance must show an unsmoothed evaluation criterion.
Other comparison policies reject unapproved loss changes.
Keep macro-F1 checkpoint selection unchanged. Do not select checkpoints from post-hoc NLL plots.

## Run locally

Open [notebook 27](../notebooks/27_patch_label_smoothing.ipynb).
Read the What/Why note before each code cell.
Keep setup disabled in the existing environment.
Prepare the candidate once. Keep preparation disabled after its manifest exists.
Inspect the fold plan, frozen recipe, loss definitions, and five no-gradient checks.
Enable RUN_TRAINING when ready. The reference is reused and never retrained automatically.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-label-smoothing-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-label-smoothing-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-patch-label-smoothing-grouped-validation/fold-01/gru.yaml --check
```

The notebook checks all five folds. Training requires the explicit gated cell or this command:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-label-smoothing-grouped.yaml --train
```

Existing result directories prevent accidental reruns. Automatic resume is unsupported.
Fresh clones require prepared development features and the completed reference artifacts.
Keep MPS to match the measured reference device.

## Decision criteria

Primary: paired mean fold macro-F1, followed by individual fold and participant differences.
Report Wake/Light/Deep/REM precision, recall, F1, support, accuracy, and confusion matrices.
Inspect selected hard-label NLL and training-evaluation versus validation NLL trajectories.
Check matched history and transition slices, with their supports shown.
Keep the small temporal slices exploratory. They do not change model inputs or eligibility.

A useful first-seed result requires replication before replacing the working reference.
Lower NLL alone is insufficient. Smoothing may reduce confidence without improving stage decisions.
If the primary result remains neutral, review feature information and participant-specific errors before extending the regularization grid.
No development result establishes independent final-test performance.

Export the report after training:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/27_patch_label_smoothing.ipynb
```

HTML export skips setup, preparation, participant training, and recursive export.
Keep source notebooks output-free. Executed HTML and large run artifacts remain local under ignored artifacts.

## Historical preparation evidence

The following checks describe the state before Timo ran participant-data training.

All five candidate folds were prepared on 2026-11-16 without participant-data optimization.
All 130 feature tables match the reused reference byte for byte.
The five resolved MPS recipes differ only in the loss mapping.
All five no-gradient checks pass with 2,756 parameters, inputs 64 × 30 × 6, and logits 64 × 4.
Each check records training smoothing 0.05 and evaluation smoothing zero.
Fresh-kernel notebook export completes with preparation and participant training skipped.
The full suite passes 661 tests. Ruff lint and whitespace checks pass.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.
