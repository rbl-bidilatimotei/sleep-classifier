# Shared-seed Transformer budget replication

Status: completed by Timo on local MPS. See the [measured three-seed results](experiments/transformer-budget-seeds-2026-11-10/README.md).

## Question

Does the longer width-16 training budget help across initialization and data-loader seeds?
The [seed-42 budget experiment](experiments/transformer-budget-2026-11-09/README.md) improved REM recall by 8.54 percentage points.
Mean fold macro-F1 increased by 0.01497. Deep F1 declined slightly, and training work more than doubled.
Replicate this paired comparison before changing the model, context, or features again.

| Training seed | Reference | Candidate | Action |
| --- | --- | --- | --- |
| 42 | 20 epochs / patience 5 | 40 epochs / patience 12 | Reuse completed results |
| 43 | 20 epochs / patience 5 | 40 epochs / patience 12 | Train both recipes |
| 44 | 20 epochs / patience 5 | 40 epochs / patience 12 | Train both recipes |

Each recipe uses the same five development folds. New work is twenty fits and at most 600 epochs.
Reference fits request at most 200 epochs; candidate fits request at most 400 epochs.
Keep split seed 42, width 16, feedforward width 32, context ten, and the existing six features.
Keep Adam, learning rate 0.001, batch 64, dropout, gradient clipping, coverage, and checkpoint selection fixed.
Use the local MPS device for comparison with the measured reference.
Every fit starts from a fresh model and fits its scaler on training participants only.
OPAL stays disabled. The final three test participants remain reserved.

## Configurations and shared code

configs/transformer-budget-seeds.yaml lists three explicit reference/candidate grouped-config pairs.
The seed-42 pair has reuse enabled. Preparation and training exclude that pair.
Each new grouped YAML has a separate model YAML and output directory.
These short configurations keep the complete recipes visible without adding a configuration-inheritance system.
The notebook uses the existing grouped_validation CLI for planning, preparation, and training.
BudgetSeedReport only plans work and reads results; it cannot train a model.
Its report validates each pair through the existing training-budget policy.
It also rejects changed folds, changed recipes across seeds, duplicate seeds, shared outputs, and incorrect recorded training seeds.

## Run locally

Open [notebook 15](../notebooks/15_transformer_budget_seeds.ipynb).
Inspect the work estimate and participant-fold equality before training.
Run the no-gradient candidate preflight.
Enable RUN_PREPARATION once for new directories, then enable RUN_TRAINING.
The training cell runs seed 43 reference/candidate, followed by seed 44 reference/candidate.
Every completed epoch logs training loss, validation loss, macro-F1, and accuracy.

Fresh checkouts need source development features and the completed seed-42 artifacts to reproduce the full report.
Leave RUN_PREPARATION disabled if the new manifests already exist locally.
Source notebooks default to disabled preparation and training.
Report export skips setup, preparation, and training even when their flags are enabled.

Inspect the study plan without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.budget_seed_report \
  --config configs/transformer-budget-seeds.yaml --plan
```

Each explicit grouped configuration also works with the standard commands:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-budget-reference-seed-43-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-budget-reference-seed-43-grouped.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-budget-reference-seed-43-grouped.yaml --train
```

Use the matching candidate and seed-44 configurations for the remaining runs.
The notebook contains the complete guarded command loop; no manual edits to measured configurations are needed.
Existing results are refused. The first failed command stops the loop and leaves completed artifacts intact.
Do not rerun the full loop after a partial failure. Inspect the failed run before choosing a recovery action.
Automatic resume remains unsupported.

## Inspect the results

The notebook shows every grouped run as complete, failed, or not_started.
It reports completed seed pairs while others remain pending. It writes the study aggregate only after all pairs finish.
For each seed, report the mean paired fold macro-F1 effect, participant gains, all fold differences, and pooled class scores.
Plot mean fold macro-F1 and REM recall by seed and recipe.
Keep REM precision and F1 beside recall when assessing the gain.
Inspect Deep, Light, and Wake trade-offs too.

After all runs finish:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.budget_seed_report \
  --config configs/transformer-budget-seeds.yaml --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/15_transformer_budget_seeds.ipynb
```

The study JSON is saved under artifacts/transformer-budget-seed-comparison.
The report includes all three validated paired reports and the mean/sample deviation of their seed-level effects.
Do not pool repeated predictions across seeds as additional independent epochs.
The same 26 participants repeat, and the fold training sets overlap.
Three training seeds describe initialization sensitivity; they do not estimate variation across new participant populations.
Validation folds select checkpoints, and longer budgets offer more selection opportunities.
These remain descriptive development results without a significance claim.

## Decision after replication

Check the direction and size of the budget effect for each seed.
A gain restricted to seed 42 would weaken the present budget recommendation.
Consistent REM gains with stable macro-F1 would support keeping the longer budget as a development reference.
Consider causal context or feature ablations only after reviewing these results.
Do not extend the budget again or tune individual folds during replication.

## Local preparation evidence

The four new grouped runs were prepared on local MPS without optimization on 2026-11-09.
All twenty fold assignments match the seed-42 reference.
All 520 copied feature CSV files match the reference byte for byte.
The base candidate preflight and fresh-kernel notebook report passed without training.
Before training, the report showed seed 42 complete and seeds 43 and 44 not_started.
The completed report now verifies all three pairs.
The four new grouped runs are complete. Keep preparation and training disabled when reviewing saved results.
All 275 tests pass, including synthetic completed-study notebook reporting and cross-seed mismatch checks.
