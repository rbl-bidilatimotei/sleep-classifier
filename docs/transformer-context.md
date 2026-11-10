# Context-ten versus context-thirty Transformer experiment

Status: completed by Timo on local MPS. See the [measured context results](experiments/transformer-context-2026-11-10/README.md).

## Question and controls

Does fifteen minutes of available causal history improve the classifier over five minutes?
Reuse the measured context-ten, width-16, seed-42, budget-40/12 reference.
The [three-seed budget results](experiments/transformer-budget-seeds-2026-11-10/README.md) support this working development recipe.
The reference remains unchanged. Train only the five new candidate folds.

| Setting | Measured reference | Candidate |
| --- | ---: | ---: |
| Maximum context epochs | 10 | 30 |
| Maximum available span | 5 minutes | 15 minutes |
| Learned position capacity | 10 | 30 |
| Trainable parameters | 2,564 | 2,884 |
| Representation / feedforward width | 16 / 32 | 16 / 32 |
| Maximum epochs per fold | 40 | 40 |
| Early-stopping patience | 12 | 12 |

The longer learned position table adds `(30 - 10) × 16 = 320` parameters.
This compares context recipes; it does not isolate history from positional capacity in a parameter-matched experiment.
Additional positions also change initialization draws. Identical early training histories are not expected.
Keep one layer, two heads, residual/feedforward dropout 0.1, and attention-weight dropout zero.
Keep six features, coverage rules, seed 42, batch 64, Adam learning rate 0.001, and gradient clipping one.
Keep the same five participant folds and validation macro-F1 checkpoint selection.
Keep the MPS device for local comparison. OPAL stays disabled.
Maximum new work is five fits and 200 epochs. Every fit uses a fresh model and training-only scaler.
The candidate does not resume any reference checkpoint.

## Causal history and target population

Classify the current completed thirty-second epoch using that epoch and earlier observations.
Windows never cross participant or contiguous-segment boundaries. Missing, rejected, and unscored intervals break history.
Keep all eligible targets, including short-history epochs. Do not require a complete thirty-epoch window.
Use the existing padding masks and causal attention masks.
The model classifies the final valid position and excludes right padding.

Notebook 16 reports full-history fractions and mean available lengths without fitting a model.
The current development tables contain 22,708 eligible targets.
About 95.35% have more than ten available epochs; 88.66% have a full thirty-epoch history.
The average available length increases from 9.77 to 28.15 epochs under the respective caps.
These diagnostic values describe the current development tables, not final-test observations or classification results.
Both model recipes must produce predictions for the exact same participant/time/target population.
The paired reporter refuses a population change instead of silently comparing only complete windows.

## Run locally

Open [notebook 16](../notebooks/16_transformer_context.ipynb).
Inspect configs/transformer-context-30.yaml and configs/transformer-context-30-grouped-validation.yaml.
Run the no-gradient preflight and inspect the fold plan and history diagnostic.
Enable RUN_PREPARATION once for fresh candidate output directories.
Then enable RUN_TRAINING and execute the training cell interactively.
Leave preparation disabled when a candidate manifest already exists on this checkout.
Source notebook flags default to disabled preparation and training.
HTML export skips setup, preparation, and training regardless of those flags.

Equivalent commands from the repository root:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer-context-30.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-context-30-grouped-validation.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-context-30-grouped-validation.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-context-30-grouped-validation.yaml --train
```

Candidate outputs use artifacts/transformer-context-30-grouped-validation.
Existing result directories are refused. Partial artifacts remain after failure; automatic resume is not supported.
Fresh clones need prepared development features and the measured reference artifacts to reproduce the paired report.
The source setup cell can clone the PR branch in another environment; local checkouts are not switched automatically.

## Verify and interpret results

Use the explicit context comparison policy after all five fits complete:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/transformer-budget-grouped-validation/summary.json \
  --candidate artifacts/transformer-context-30-grouped-validation/summary.json \
  --comparison context \
  --output artifacts/transformer-context-comparison/summary.json
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/16_transformer_context.ipynb
```

The default architecture and training-budget policies retain their previous restrictions.
Context mode permits a longer history and the corresponding learned position table.
It rejects changes to width, depth, dropout, optimization, budget, coverage, provenance, or evaluation population.
It verifies the parameter difference against the position-table formula and the grouped epoch cap against each fold recipe.

Use mean paired fold macro-F1 as the primary comparison. Report every fold, participant, and class trade-off.
Inspect REM precision and F1 beside recall. Check Deep, Light, and Wake behavior too.
Inspect histories, selected epochs, completed epochs, and measured runtime before attributing an effect to temporal information.
The seed-42 comparison is exploratory development evidence. Fold validation also selects checkpoints.
Do not claim statistical significance or new-participant generalization from these results.
The three final test participants remain reserved.

Review the measured result before choosing seed replication, more work on this encoder, or another architecture.
The [architecture assessment](architecture-options.md) keeps the later options and their task assumptions explicit.
Do not tune different folds separately or expand the experiment during execution.

## Local preparation evidence

On 2026-11-10, five context-thirty folds were prepared on MPS without optimization.
The complete grouped plan matches the measured context-ten reference, including seed, epoch cap, and checkpoint rule.
All 130 copied feature CSV files match the reference byte for byte.
The configuration comparison confirms that only the output path, context length, and position capacity changed.
Base and frozen fold-one no-gradient preflights passed with 64 × 30 × 6 inputs and four output logits.
The candidate has 2,884 parameters. Synthetic MPS forward/backward updates passed at context ten and thirty.
Notebook 16 executed from a fresh kernel and exported as HTML without training.
Before training, it reported the reference complete and the candidate not_started.
The completed report now validates both experiments and includes a post-hoc history-length diagnostic.
The five fits are complete. Keep preparation and training disabled when reviewing saved results.
All 298 tests and lint pass, including strict context-policy checks and completed synthetic notebook reporting.
