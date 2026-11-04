# Paired GRU seed comparison

Status: prepared for Timo to run. No measured results exist yet.

The first follow-up selected patience twelve at macro-F1 0.5001, versus reference 0.4760.
That comparison used one training seed and three validation participants.
Check whether the gain persists across training initialization and loader order before increasing model complexity.

## Fixed protocol

Use configs/gru-seeds.yaml with shared seeds 42, 43, and 44.
Train reference and patience twelve for each seed, creating six fresh models.
Keep the twenty-epoch cap, model size, features, split, scaling procedure, and resolved device fixed.
Use the same Adam optimizer and best-validation-macro-F1 checkpoint rule.
The suite resolves the device once and records every generated configuration.
It refuses existing output directories. It preserves partial results on failure but does not resume them.
Maximum work is 120 epochs; early stopping can reduce this total.
Seed 42 is repeated deliberately to keep this comparison self-contained.

Inspect the plan:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.gru_suite --config configs/gru-seeds.yaml --plan
```

Launch the six runs when ready:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.gru_suite --config configs/gru-seeds.yaml --train
```

Open notebook 09 for optional setup, planning, training, and graphical reporting.
Select experiment/gru-seed-comparison before merge; use main after merge.
Report execution skips setup and training. Source notebooks remain output-free.

## Interpretation

Report each run's class scores, accuracy, macro-F1, best epoch, and completed epochs.
The completed summary includes recipe means, sample standard deviations, minimums, and maximums for macro-F1.
It also includes patience-minus-reference macro-F1 for each matching seed and the mean and sample deviation of those differences.
Inspect REM precision beside recall and F1. Inspect participant results in each run's complete metrics file.
A positive mean difference alone does not establish a reliable improvement with three seeds.
Do not treat epochs or overlapping windows as independent participants.
This experiment measures training-seed sensitivity, not participant generalization or statistical significance.
Use grouped development folds for participant sensitivity later. Keep final test observations reserved.

## Following milestone

Use this comparison to choose a documented Adam reference for a separate Codex-powered OPAL pilot.
Agree on the account and token budget before live agent calls.
The seed suite makes no language-model calls and does not use OPAL.
