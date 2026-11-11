# Shared-seed dropout-0.2 replication

Status: completed locally by Timo and reviewed on 2026-11-11.
See the [measured results](experiments/transformer-dropout-seeds-2026-11-11/README.md) and [cell-by-cell guide](notebook-19-guide.md).

## Question and fixed controls

Does the dropout-0.2 validation benefit repeat with seeds 43 and 44?
The [seed-42 dropout result](experiments/transformer-dropout-2026-11-11/README.md) improved pooled accuracy, REM F1, and selected-checkpoint NLL.
Mean fold macro-F1 gained only 0.00539; two folds and twelve participants lost macro-F1.
Replicate this modest effect before changing the established dropout default.

| Seed | Dropout-0.1 reference | Dropout-0.2 candidate | New fits |
| --- | --- | --- | ---: |
| 42 | Reuse completed context-thirty run | Reuse completed dropout run | 0 |
| 43 | Reuse completed context-thirty run | Train five new folds | 5 |
| 44 | Reuse completed context-thirty run | Train five new folds | 5 |

New work is ten fits with a 400-epoch cap. No reference is retrained.
Both recipes retain context thirty, width 16, feedforward width 32, one layer, two heads, and 2,884 parameters.
Only residual/feedforward dropout differs between recipes: 0.1 versus 0.2.
Attention-weight dropout stays zero for the existing MPS-compatible implementation.
Keep the six features, coverage rules, participant folds, split seed 42, natural sampling, and unweighted cross-entropy fixed.
Keep forty epochs, patience twelve, batch 64, Adam learning rate 0.001, gradient clipping one, and validation macro-F1 checkpoint selection.
Each new candidate starts from a fresh model and fits its scaler on training participants only.
Use local MPS for comparison with the measured references. OPAL remains disabled.
The three final test participants stay reserved.

## Reused artifacts and shared interfaces

configs/transformer-dropout-seeds.yaml declares architecture-recipe comparison and per-recipe reuse flags.
All reference flags are true. The seed-42 candidate is also reused.
Only candidate seeds 43 and 44 appear in PairedSeedReport.new_runs().
The notebook uses the existing grouped_validation CLI; no trainer, model, sampler, or comparison-policy implementation changes.
Before preparation or training, it verifies every reused grouped artifact.
Missing or damaged references block execution rather than triggering automatic reference training.
Checks verify metric hashes, saved prediction scores, participant populations, training seeds, recipes, model metadata, and devices.
The complete report also requires consistent per-role recipes and fold plans across seeds.

Fresh clones need source development features and all four reused grouped artifacts.
Those are context-thirty dropout-0.1 results for seeds 42, 43, and 44, plus the seed-42 dropout-0.2 result.
New candidates use artifacts/transformer-dropout-seeds/seed-43/candidate and the matching seed-44 directory.

## Original run protocol

The candidates are complete on this checkout. Do not rerun preparation or training for the review.
Run the report cells with RUN_TRAINING and RUN_PREPARATION false.

## Run locally

Open [notebook 19](../notebooks/19_transformer_dropout_seeds.ipynb).
Inspect the plan: only candidate seeds 43 and 44 should appear, requesting ten fits and 400 maximum epochs.
Run the no-gradient preflight and reused-artifact checks.
Enable RUN_PREPARATION once for fresh output directories.
Then enable RUN_TRAINING and execute the training cell interactively.
The training cell runs dropout-0.2 seed 43, followed by dropout-0.2 seed 44.
Leave preparation disabled when both new manifests already exist locally.
Source notebook preparation and training flags default to false.
HTML export skips setup, preparation, training, and recursive export regardless of those flags.

Plan without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-dropout-seeds.yaml --plan
```

Each new candidate also works through the standard commands:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer-dropout-02-seed-43.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-dropout-02-seed-43-grouped.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-dropout-02-seed-43-grouped.yaml --train
```

Use the matching seed-44 configuration for its candidate. The notebook contains the complete guarded command loop.
Existing results are refused. The first failed command stops the loop and preserves completed artifacts.
Automatic resume is unsupported. Inspect a partial failure before choosing a recovery action.
Do not rerun the complete loop after an earlier candidate has already finished.

## Reports and decision

The notebook distinguishes reused and new runs, and complete, failed, and not_started results.
It reports seed 42 while the new candidates remain pending. The study aggregate requires all three pairs to finish.
Report paired mean fold macro-F1, every fold difference, participant gains/losses, and pooled class precision/recall/F1.
Inspect REM gains beside Wake behavior and Deep recall. Do not judge the candidate from pooled accuracy alone.
The saved-probability diagnostic recomputes selected-checkpoint pooled NLL and descriptive confidence statistics.
A lower NLL does not by itself establish calibration, and confidence on different error sets is not a controlled comparison.
A smaller train/validation gap caused only by higher stochastic training loss is not success.
Keep checkpoint selection on macro-F1; do not switch to NLL after viewing the results.

After both new candidates complete:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-dropout-seeds.yaml --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/19_transformer_dropout_seeds.ipynb
```

The aggregate is saved under artifacts/transformer-dropout-seed-comparison.
It includes all three validated paired reports and the mean/sample deviation of seed-level macro-F1 effects.
The same 26 development participants repeat across seeds. Fold training sets overlap, and validation selects checkpoints.
These are descriptive initialization repeats, not independent outer tests or significance evidence.
Do not expand the dropout grid, resample data, or tune individual folds during this replication.
Review the measured trade-offs before changing the default or choosing the next architecture experiment.

## Historical preparation evidence

This records the state before Timo ran training. The completed review below supersedes pending-run status.

On 2026-11-11, only the two new dropout-0.2 grouped candidates were prepared on MPS without optimization.
Their ten fold assignments match the seed-42 plan apart from training seed.
Only the seed and output location differ from the measured dropout-0.2 model recipe.
All 260 copied feature CSV files match the seed-42 reference byte for byte.
All four reused grouped artifacts passed validation before and after preparation.
The plan contains only candidate seeds 43 and 44: ten fits and 400 maximum epochs.
Notebook 19 executed from a fresh kernel and exported as HTML without training.
It shows all reused results complete and the two new candidates not_started.
The base candidate no-gradient MPS preflight passed with a 64 × 30 × 6 batch and 2,884 parameters.
At preparation, all 313 tests and lint passed, including shared synthetic completed-context/dropout-study reports and probability diagnostics.
At that stage, preparation was complete and training required explicit opt-in.

## Completed review

All three seeds improve mean fold macro-F1 and selected-checkpoint pooled NLL.
The mean paired macro-F1 gain is 0.00394; only eight of fifteen seed/fold cells improve.
Seed 44 loses accuracy and REM F1. Stronger dropout provides a modest, mixed development benefit.
Notebook cells 20–23 inspect the epoch-fourteen question and expose optional read-only checkpoint replay.
Report export skips replay and displays its existing result when available.
The replay restores saved scaling and never fits preprocessing or updates weights.
The proposed next work is a two-fold instrumented repeat, not another seed grid.

The updated notebook executed from a fresh kernel with training and preparation skipped.
All 317 tests and Ruff lint pass after the review changes.
The executed HTML report remains local; curated JSON and PNG evidence are tracked in docs/experiments.
