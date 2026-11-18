# Replicate the window-local HR feature recipe across seeds

Status: completed locally by Timo. See the [measured replication](experiments/window-local-hr-seeds-2026-11-18/README.md).

## Question

Does the [seed-42 HR feature gain](experiments/window-local-heart-rate-2026-11-17/README.md) repeat under different training randomness?
Mean fold macro-F1 improves by 0.01316, mainly through fold 01. Pooled REM precision, recall, and F1 improve.
Wake F1, Deep recall, and transition scores decline. Replicate the combined feature recipe before promotion or another feature change.

## Reuse and new work

| Training seed | Original Adam/P3 reference | HR deviation + availability | New fits |
| --- | --- | --- | ---: |
| 42 | Reuse original patch result | Reuse notebook 29 result | 0 |
| 43 | Reuse original patch result | Train five fresh candidate folds | 5 |
| 44 | Reuse original patch result | Train five fresh candidate folds | 5 |

The study requires ten new candidate fits, capped at 400 epochs in total.
Reuse every reference and the completed seed-42 candidate. Missing reused results must block execution rather than trigger retraining.
The seed-43/44 reference output paths contain candidate because that was their role in the earlier patch-architecture experiment.
Do not rename or modify those artifacts.

## Fixed recipe

Change only the training seed and output paths from the measured HR candidate.
Keep the split seed at 42, with the same five folds, 26 development participants, and three reserved final-test participants.
Training seeds change initialization, minibatch order, and stochastic dropout behavior. They do not create independent participant cohorts.

- Thirty raw epochs and six original prepared input features
- Window-local deviation from up to ten strictly previous HR means, requiring four earlier epochs
- Training-only deviation normalization and an unscaled binary availability flag
- Eight effective model channels and 2,852 parameters; the reference retains six channels and 2,756 parameters
- Three-epoch patches, width 16, two heads, one layer, feedforward width 32, and dropout 0.2
- Original Adam 0.001, no scheduler or agent, unsmoothed unweighted cross-entropy, and natural sampling
- Forty epochs, patience twelve, batch 64, norm clipping one, and validation macro-F1 checkpoint selection
- The same per-epoch diagnostics, predictions, intermediate checkpoints, and MPS device

Fit both normalization stages separately on each fold's training participants. Keep unavailable-history targets eligible.
Derive both channels inside each supplied window; do not extend the effective information boundary.
Restore fitted normalization from checkpoints during inference. See the [feature definition](window-local-heart-rate.md).

## Run locally

Open [notebook 30](../notebooks/30_window_local_hr_seeds.ipynb).
Read the What/Why explanation before each code cell.
Keep RUN_SETUP false in the existing environment.
Keep RUN_PREPARATION false after the two candidate manifests exist.
Inspect the reuse matrix, fixed-recipe checks, and ten no-gradient checks.
Set RUN_TRAINING to true when ready. Execute the training cell interactively.
It trains candidate seed 43, then candidate seed 44. It never retrains the references or candidate seed 42.

Source notebooks default to training disabled. Automated HTML export skips setup, preparation, training, and recursive export.
Existing result directories block accidental reruns. Automatic resume is unsupported.
After a partial failure, inspect retained artifacts before choosing recovery. Do not blindly rerun the two-candidate loop.

Plan without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-window-local-hr-seeds.yaml --plan
```

A fresh clone also needs development feature tables and these completed measured artifacts:

- artifacts/transformer-patch-grouped-validation
- artifacts/transformer-patch-seeds/seed-43/candidate
- artifacts/transformer-patch-seeds/seed-44/candidate
- artifacts/transformer-window-local-hr-grouped-validation

Prepare each new candidate only when its output does not already exist:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-window-local-hr-seed-43-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-window-local-hr-seed-44-grouped.yaml --prepare
```

The notebook runs no-gradient checks on every prepared fold.
Run one seed independently, if needed:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-window-local-hr-seed-43-grouped.yaml --train
```

Use the matching seed-44 config for its training command.
Outputs use artifacts/transformer-window-local-hr-seeds/seed-43/candidate and the corresponding seed-44 directory.

## Evaluation and decision

Use the explicit window_features comparison policy for each paired seed.
Validate matching predictions, supports, participant folds, recorded recipes, budgets, device, and metric hashes.
The notebook compares planned settings against measured settings before preparation or training.
The existing paired reporter also validates the completed candidate's normalization provenance and effective model channels.

Report paired mean fold macro-F1 for each seed, then the three-seed mean effect and sample deviation.
The complete aggregate requires all three completed pairs. Missing candidate scores remain pending.
Do not treat fifteen seed/fold cells as independent population trials or the sample deviation as a confidence interval.
Inspect each fold and participant beside every stage's precision, recall, F1, support, and confusion matrix.
Check whether REM gains persist without hiding Wake or Deep regressions.

Inspect saved evaluation-mode NLL curves beside training loss and macro-F1.
NLL measures the negative log probability assigned to the true class. Lower values indicate better probability loss.
The notebook recomputes selected pooled NLL from saved probabilities and validates their normalization and predicted class.
Confidence-on-error summaries compare changing error sets; keep them descriptive.

Use the same history bins: 1–4 epochs, 5–29 epochs, and full thirty-epoch history.
Check the same stage-transition definition. History bins are disjoint; transitions overlap them.
Save per-class slice scores and supports. Do not average slice macro-F1 values to reconstruct pooled macro-F1.
Repeated development use remains adaptive exploration. Keep the final test untouched.

Both new channels remain bundled. A replicated gain would support this recipe, without isolating deviation, availability, capacity, or initialization effects.
No candidate promotion or next feature change is decided before reviewing the new results.

Export the HTML report without training:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/30_window_local_hr_seeds.ipynb
```

The executed report stays under ignored artifacts/notebooks. Source notebooks, configs, tests, and this protocol remain tracked.

## Preparation evidence

The [preparation snapshot](experiments/window-local-hr-seeds-preparation-2026-11-17/README.md) records ten MPS checks and unchanged feature-table hashes.
All 799 tests, Ruff, whitespace checks, and fresh-kernel HTML export pass.
No new candidate training ran. Keep RUN_PREPARATION false in the current checkout; set RUN_TRAINING true when ready.

## Measured outcome

All three seeds improve mean fold macro-F1; the mean paired seed effect is +0.01834.
REM precision, recall, and F1 improve at each seed. Wake F1 declines at each seed.
The combined HR recipe becomes the working development reference. Next, isolate deviation with an availability-only control at fixed model shape.
