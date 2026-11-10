# Shared-seed context-thirty replication

Status: prepared for Timo to run locally. No new participant-data context replication training has run.

## Question and fixed settings

Do the context-thirty gains repeat with seeds 43 and 44?
The [seed-42 result](experiments/transformer-context-2026-11-10/README.md) improved macro-F1 in all five folds.
Wake and Deep improved, while REM F1 declined. Nine of 26 participants lost macro-F1.
Replicate that trade-off before selecting a default or changing architecture.

| Seed | Context-ten reference | Context-thirty candidate | New fits |
| --- | --- | --- | ---: |
| 42 | Reuse completed budget-40/12 run | Reuse completed context-thirty run | 0 |
| 43 | Reuse completed budget-40/12 run | Train five new folds | 5 |
| 44 | Reuse completed budget-40/12 run | Train five new folds | 5 |

New work is ten fits, with a maximum of 400 epochs. No reference is retrained.
Keep width 16, feedforward width 32, one layer, two heads, six features, and the same five participant folds.
Keep split seed 42, forty epochs, patience twelve, batch 64, Adam learning rate 0.001, dropout, and gradient clipping one.
Keep coverage, causal window construction, and validation macro-F1 checkpoint selection fixed.
Every new fit starts from a fresh model and fits its scaler on training participants only.
The candidate has 2,884 parameters; the context-ten reference has 2,564.
This positional-capacity difference is unchanged from the seed-42 experiment.
Use local MPS for comparison with the existing references. OPAL remains disabled.
Keep all eligible targets, including short histories. Participant and contiguous-segment boundaries still reset history.

## Configurations and reuse safeguards

configs/transformer-context-seeds.yaml selects comparison: context and lists three explicit grouped-config pairs.
Each recipe has its own reuse flag. All reference flags are true; only candidate seeds 43 and 44 have false flags.
Only those two candidates appear in the new-work plan.
The notebook uses the existing grouped_validation CLI for preparation and training.
PairedSeedReport plans work and validates results; it cannot train a model.
BudgetSeedReport and its old CLI remain compatibility entry points for earlier budget notebooks.
The budget study retains its default training_budget policy and pair-level reuse behavior.

Before new work, the notebook validates every reused grouped result.
Missing or failed reused runs block execution rather than triggering automatic reference training.
Checks verify metric hashes, saved prediction scores, participant populations, seeds, recipes, model metadata, and devices.
Complete study reporting also requires consistent per-role recipes and fold plans across seeds.
The existing context policy verifies unchanged optimization and the expected learned-position parameter difference within each pair.

## Run locally

Open [notebook 17](../notebooks/17_transformer_context_seeds.ipynb).
Inspect the plan: only candidate seeds 43 and 44 should appear, with ten total fits and 400 maximum epochs.
Run the no-gradient preflight and the reused-result checks.
Enable RUN_PREPARATION once for fresh candidate directories.
Then enable RUN_TRAINING and execute the training cell interactively.
The training cell runs candidate seed 43, followed by candidate seed 44.
Leave preparation disabled when both new manifests already exist locally.
Source notebooks default to disabled preparation and training.
HTML export skips setup, preparation, training, and recursive export regardless of those flags.

Fresh clones need source development features and all four reused grouped artifacts before starting.
The reused artifacts are seed-42 context ten and thirty, plus seed-43 and seed-44 context ten at budget 40/12.
Their files stay unchanged; new candidates use artifacts/transformer-context-seeds/seed-43/candidate and the matching seed-44 directory.
The first failed training command stops the loop. Completed runs remain intact; automatic resume is unsupported.
Do not rerun the full loop after a partial failure. Inspect the failed run before choosing a recovery action.

Plan without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-context-seeds.yaml --plan
```

Each candidate also works through the standard commands:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer-context-30-seed-43.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-context-30-seed-43-grouped.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-context-30-seed-43-grouped.yaml --train
```

Use the matching seed-44 configuration for its candidate. The notebook contains the complete guarded command loop.

## Report and decision

The notebook distinguishes reused runs from new runs and complete results from not_started or failed results.
It reports seed 42 while the new candidates remain pending. The aggregate requires all three pairs to complete.
Show paired mean fold macro-F1, every fold difference, participant gains/losses, and class precision/recall/F1.
Inspect Wake and Deep improvements beside REM losses. Read recall with precision and F1.
The plots compare macro-F1, REM recall, and Wake/Deep/REM F1 across seeds and contexts.
Do not select only the strongest fold or tune individual folds separately.

After both candidates complete:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-context-seeds.yaml --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/17_transformer_context_seeds.ipynb
```

The aggregate is saved under artifacts/transformer-context-seed-comparison.
It includes each validated pair and the mean/sample deviation of seed-level paired macro-F1 effects.
The same 26 development participants repeat across seeds. Fold training sets overlap, and validation selects checkpoints.
These results describe initialization sensitivity; they are not independent outer-test estimates or significance evidence.
The three final test participants remain reserved.
Review the measured trade-offs before choosing more work on this encoder or an adapted architecture.

## Local preparation evidence

On 2026-11-10, only the two new grouped candidates were prepared on MPS without optimization.
All ten fold assignments match the completed seed-42 context-thirty plan apart from the training seed.
Only the seed and output location differ from the context-thirty model recipe.
All 260 copied feature CSV files match the seed-42 reference byte for byte.
All four reused grouped artifacts passed validation before and after preparation.
Base and frozen seed-43 fold-one no-gradient MPS preflights passed with 64 × 30 × 6 inputs and four logits.
The candidate still has 2,884 parameters. The legacy budget CLI plan remains functional.
Notebook 17 executed from a fresh kernel and exported as HTML without training.
It shows the reused runs complete and the two new candidates not_started.
On this checkout, leave RUN_PREPARATION disabled and enable RUN_TRAINING when ready.
All 306 tests and lint pass, including per-recipe reuse guards and synthetic completed-context-study reporting.
