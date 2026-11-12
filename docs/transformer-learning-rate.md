# Fixed-recipe learning-rate diagnostic

Status: completed locally by Timo.
See the [measured results](experiments/transformer-learning-rate-2026-11-12/README.md).
The lower rate improves pooled NLL but loses mean fold macro-F1; retain 0.001 as the stage-classification reference.

## Question and hypothesis

Does reducing the constant Adam learning rate from 0.001 to 0.0003 improve the two-fold diagnostic trade-off?
The [completed diagnostic repeat](experiments/transformer-diagnostics-2026-11-12/README.md) identified a growing fixed-weight training/validation NLL gap.
It also identified persistent errors with falling true-stage probabilities and frequent late gradient clipping.
The evidence does not prove that learning rate causes the overfitting.
The hypothesis is that smaller update steps reduce short-term probability swings without sacrificing selected macro-F1.

## Fixed controls and scope

Reuse the completed instrumented seed-42 reference in artifacts/transformer-diagnostics.
Train only the lower-rate candidate on the same original folds 01 and 04.
Generate those partitions from the original five-fold split, not a new two-fold split.
The maximum new work is two fits and eighty total epochs.

| Setting | Reference | Candidate |
| --- | --- | --- |
| Constant Adam learning rate | 0.001 | 0.0003 |
| Seed and split seed | 42 | 42 |
| Context / width / dropout | 30 / 16 / 0.2 | Unchanged |
| Epoch cap / patience | 40 / 12 | Unchanged |
| Batch / gradient clipping | 64 / 1.0 | Unchanged |
| Loss and sampling | Unweighted cross-entropy, natural sampling | Unchanged |
| Checkpoint objective | Validation macro-F1 | Unchanged |
| Evidence controls | Per-epoch predictions, training evaluation, intermediate epochs 13–17 | Unchanged |

All model settings, features, eligibility rules, scaling, participant assignments, and device remain fixed.
Each new fit starts from a fresh model and fits its scaler on training participants only.
The recipe comparison permits only learning_rate and storage-path differences.
The three final-test participants remain unused. OPAL stays disabled.

This is a post-hoc eleven-person diagnostic subset, not a complete development-cohort benchmark.
A useful result motivates broader validation before adopting a global recipe.
A negative result within forty epochs does not establish worse asymptotic performance at the lower rate.
Do not simultaneously change patience, epochs, dropout, sampling, scheduler, or checkpoint selection.

## Run locally

Open [notebook 21](../notebooks/21_transformer_learning_rate.ipynb).
Read the What/Why explanation before each code cell.
Validate the reference, original fold assignments, device, and fixed recipe before preparation.
Keep setup disabled for the existing local environment.
Prepare the candidate only once, then run both no-gradient preflights.
Enable RUN_TRAINING when ready. Leave RUN_PREPARATION false after preparation completes.
The reference is reused and never automatically retrained.

Fresh clones require development features and the completed instrumented reference artifacts.
Use MPS to match the measured local reference device. A different device requires a separately agreed comparison.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-learning-rate-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-learning-rate-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-learning-rate/fold-01/gru.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-learning-rate/fold-04/gru.yaml --check
```

Start model optimization only through the explicit command or gated notebook cell:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-learning-rate-grouped.yaml --train
```

Existing model results block a rerun. Automatic resume is unsupported.
Inspect retained evidence after a failure before choosing a recovery action.
A report-only execution skips setup, preparation, training, and recursive export.
It never starts a model fit.

## Evaluation and decision

Primary: compare each fold's selected macro-F1, then the mean paired fold difference.
Retain the same checkpoint-selection rule. NLL remains a diagnostic and secondary score.
Show Wake, Light, Deep, and REM precision/recall/F1 beside pooled accuracy.

Inspect selected-checkpoint probability NLL, evaluation-mode training NLL, and each full loss trajectory.
Compare the largest upward NLL steps, confidence behavior, and pre-clipping gradient/clipped-step statistics.
The same epoch numbers do not represent the same optimization progress at different learning rates.
A smoother validation curve caused only by slower learning is insufficient evidence of improved generalization.
Do not choose the candidate from minimum NLL alone or select a separate recipe for each fold.

The strict learning_rate policy verifies model metadata, all other recipe fields, matched target identities, folds, devices, coverage, and metric hashes.
It preserves the subset scope in the report. Partial candidate evidence remains inspectable without fabricating paired scores.
After both candidate fits complete, use the notebook or CLI:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/transformer-diagnostics/summary.json \
  --candidate artifacts/transformer-learning-rate/summary.json \
  --comparison learning_rate --output artifacts/transformer-learning-rate/paired-comparison.json
uv run --locked --group gru --group notebook python -m sleep_classifier.experiment_evidence \
  --output-dir artifacts/transformer-learning-rate --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/21_transformer_learning_rate.ipynb
```

The local HTML and large evidence files stay under artifacts. Curate reviewed findings under docs/experiments after training.
See the [evidence protocol](transformer-diagnostics.md) for NLL decomposition, checkpoint limitations, and file semantics.

If this bounded check gives little benefit, return to the planned small causal patch encoder rather than expanding the learning-rate grid.
If it helps, verify the benefit beyond these diagnostic folds before adopting the setting.

## Historical preparation checks

These checks describe the pre-training state. The measured report supersedes pending-result status.

On 2026-11-12, both candidate folds were prepared without model optimization.
All 52 participant feature CSV files match the completed reference byte for byte.
The resolved fold recipes differ only at learning_rate: 0.001 versus 0.0003.
Both MPS forward checks pass with 2,884 parameters and 64 × 30 × 6 input windows.
Notebook 21 exported from a fresh kernel with training and preparation skipped.
It shows complete reference evidence and a prepared candidate, with no paired candidate result yet.
All 399 tests, Ruff lint, and whitespace checks pass.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.
