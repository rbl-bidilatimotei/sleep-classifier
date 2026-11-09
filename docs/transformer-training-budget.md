# Width-16 Transformer training-budget experiment

Status: prepared for Timo to run locally. No participant-data budget training has run.

## Question and evidence

Does a larger training budget improve the fixed width-16 recipe on the same development folds?
The [capacity experiment](experiments/transformer-capacity-2026-11-09/README.md) lost macro-F1 in all five folds.
Width 16 remains the reference. Its folds 01 and 03 reached the twenty-epoch cap.
That supports testing a longer budget. It does not establish that more training will improve generalization.

| Setting | Measured reference | Candidate |
| --- | ---: | ---: |
| Maximum epochs per fold | 20 | 40 |
| Patience without improved validation macro-F1 | 5 | 12 |
| Maximum epochs across five folds | 100 | 200 |
| Representation / feedforward width | 16 / 32 | 16 / 32 |
| Parameters | 2,564 | 2,564 |

Change the cap and patience together as one budget policy. This experiment cannot separate their individual effects.
Keep one layer, two heads, context ten, residual/feedforward dropout 0.1, and attention-weight dropout zero.
Keep six features, coverage rules, participant folds, seed 42, batch size 64, and Adam learning rate 0.001.
Keep gradient clipping one and validation macro-F1 checkpoint selection. OPAL remains disabled.
Use the same resolved device as the local MPS reference.
The candidate fits a fresh model and training-only scaler in each fold. It does not resume reference checkpoints.
Reuse the completed width-16 artifacts. Do not rerun or edit their configurations.
Forty epochs and patience twelve are a bounded first test; do not extend them during the run.

## Run the experiment

Open [notebook 14](../notebooks/14_transformer_training_budget.ipynb).
Use the current checkout locally. The optional setup cell can clone the PR branch on another machine.
Fresh clones need the prepared source features and measured reference artifacts before producing a paired comparison.
The notebook reports missing candidate artifacts as not_started.

1. Inspect the configurations and fold plan.
2. Run the no-gradient preflight.
3. Enable RUN_PREPARATION once for a fresh output directory.
4. Enable RUN_TRAINING and execute the training cell interactively.
5. Inspect the saved paired scores, class plots, participant scores, and validation histories.
6. Export the HTML report.

Leave RUN_PREPARATION disabled if a candidate manifest already exists locally.
Source notebooks default to disabled preparation and training. Report export skips both cells regardless of their flags.
Training logs each completed epoch, training loss, validation loss, macro-F1, and accuracy.
Early stopping restores the best validation macro-F1 checkpoint rather than the final epoch.

Equivalent commands from the repository root:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer-budget-40-12.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-budget-grouped-validation.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-budget-grouped-validation.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-budget-grouped-validation.yaml --train
```

Outputs use artifacts/transformer-budget-grouped-validation. Existing output directories are refused.
Partial results are retained after failure; automatic resume is not supported.

## Validate the comparison

The default paired-report policy still requires equal training budgets for architecture experiments.
This experiment explicitly selects training_budget mode.
That mode permits larger epoch and patience settings while requiring the same model configuration and parameter count.
It rejects other model or training recipe changes. Output and preparation paths may differ.
The reporter still checks metric hashes, participant splits, provenance, coverage, device, and exact held-out epoch alignment.
It verifies each grouped epoch cap against the fold recipe.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/transformer-grouped-validation/summary.json \
  --candidate artifacts/transformer-budget-grouped-validation/summary.json \
  --comparison training_budget \
  --output artifacts/transformer-budget-comparison/summary.json
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/14_transformer_training_budget.ipynb
```

## Interpret the result

Use the mean paired fold macro-F1 difference as the primary score.
Report all folds, participant gains and losses, and every class precision/recall/F1.
Inspect completed epochs, cap hits, and selected epochs beside validation histories.
Check whether a selected improvement occurs after epoch twenty or after a wait exceeding the old patience.
A best epoch before twenty can still benefit from longer patience.
Compare training and validation losses before attributing changes to insufficient training.

A larger budget offers more validation checkpoint opportunities. A higher development score alone does not establish better generalization.
MPS runs may differ numerically; identical initial histories are not promised.
One seed cannot establish robustness. Repeated seeds can follow if the measured improvement warrants them.
The folds share training participants and also select checkpoints. These are development comparisons, not independent outer-test estimates.
Keep the final three test participants reserved. Curate measured results only after Timo completes the run.

## Local preparation evidence

The width-16 MPS preflight passed without optimization on 2026-11-09.
The candidate has 2,564 parameters. Only the output path, epoch cap, and patience differ from the reference configuration.
Participant membership and checkpoint-selection rules match the measured reference plan.
Five fold tables and their manifest were prepared without training.
On this checkout, leave RUN_PREPARATION disabled and enable RUN_TRAINING when ready.
Notebook 14 was executed from a fresh kernel and exported as HTML.
It reports the measured reference as complete and the candidate as not_started.
Synthetic tests exercise the completed comparison and plot cells without training a participant-data model.
