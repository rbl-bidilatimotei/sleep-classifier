# Replicate the three-epoch patch recipe across seeds

Status: prepared for Timo to run locally. Only seed 42 has measured patch-model results.

## Question

Do the patch model's mean fold macro-F1 and REM gains repeat under different initialization and training randomness?
The [seed-42 comparison](experiments/causal-patch-transformer-2026-11-13/README.md) improves mean fold macro-F1 by 0.01334.
Four folds and fourteen participants improve. REM precision, recall, and F1 all rise.
The small first-two-epoch-history slice loses performance, and transition scores are essentially flat.
Replicate the recipe before changing patch size, overlap, capacity, or optimization.

## Reuse and new work

| Training seed | Per-epoch Transformer reference | Patch candidate | New fits |
| --- | --- | --- | ---: |
| 42 | Reuse measured dropout-0.2 result | Reuse notebook 23 result | 0 |
| 43 | Reuse measured dropout-0.2 result | Train five fresh folds | 5 |
| 44 | Reuse measured dropout-0.2 result | Train five fresh folds | 5 |

The two new candidates require ten fits with a maximum of 400 total epochs.
No reference or seed-42 candidate is retrained.
The seed-43/44 reference paths contain candidate because that was their role in the earlier dropout study.
They serve as references in this new patch comparison; the existing artifacts are not renamed or modified.

Keep patch size three, raw context thirty, width sixteen, two heads, one layer, feedforward width 32, and dropout 0.2.
Keep Adam 0.001, no scheduler or agent, unweighted cross-entropy, natural sampling, batch 64, clipping one, and budget 40/12.
Keep validation macro-F1 checkpoint selection and all evidence controls unchanged.
The patch model remains a joint-channel causal adaptation with 2,756 parameters.
Every new model starts fresh and fits scaling on its own training participants only.

Training seed changes initial weights, minibatch order, and stochastic dropout behavior.
Split seed remains 42. All six study configurations retain the same five participant assignments.
The same 26 development people repeat across seeds; three final-test participants remain reserved.
This does not create three independent cohorts or fifteen independent population trials.

## Integrity checks and required inputs

configs/transformer-patch-seeds.yaml declares all reused and new recipes explicitly.
PairedSeedReport.new_runs() must return only candidate seeds 43 and 44.
Before preparing or training, the notebook verifies reused metric hashes, saved prediction scores, populations, fold plans, model metadata, and devices.
It compares each planned role's model settings against its saved measured recipe, excluding only seed, storage paths, and separately checked device resolution.
Missing or inconsistent reused data blocks execution instead of triggering automatic retraining.

Fresh clones need the unscaled development features plus these measured grouped artifacts:

- artifacts/transformer-dropout-grouped-validation
- artifacts/transformer-patch-grouped-validation
- artifacts/transformer-dropout-seeds/seed-43/candidate
- artifacts/transformer-dropout-seeds/seed-44/candidate

New outputs are artifacts/transformer-patch-seeds/seed-43/candidate and the corresponding seed-44 directory.
Use MPS to match the measured local references.

## Run locally

Open [notebook 24](../notebooks/24_patch_transformer_seeds.ipynb).
Read each What/Why explanation and inspect the reuse matrix and work budget.
Keep setup disabled in the existing environment.
Enable RUN_PREPARATION only when both candidate studies still need their first preparation.
Then run the ten no-gradient fold checks.
Enable RUN_TRAINING when ready and execute the training cell interactively.
The cell trains candidate seed 43, then candidate seed 44.
Keep RUN_PREPARATION false after the two manifests exist.

Plan without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-patch-seeds.yaml --plan
```

Each candidate also supports the ordinary grouped CLI:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-seed-43-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-patch-seeds/seed-43/candidate/fold-01/gru.yaml --check
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-seed-43-grouped.yaml --train
```

Use the matching seed-44 configuration for its candidate.
The notebook contains both guarded commands and checks every prepared fold.
Existing results block a rerun. Automatic resume is unsupported.
After a partial failure, inspect retained evidence before choosing recovery; do not rerun the complete two-candidate loop blindly.

## Reports and decision

Seed 42 can be reported while the new candidates remain pending. Missing scores are not replaced by zero.
The full aggregate requires all three reference/candidate pairs to complete.
Report the paired mean fold macro-F1 effect for every seed, then its three-seed mean and sample standard deviation.
Those descriptive quantities are not a confidence interval or an independent-test result.

Show each fold difference and participant gains/losses beside every class's precision, recall, F1, and support.
Show REM precision and F1 beside recall. Do not select the model from accuracy or REM recall alone.
Recompute selected pooled NLL from saved normalized class probabilities and retain confidence summaries as secondary diagnostics.
A lower NLL alone does not establish calibration or improved hard-label classification.

After both new candidates finish:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/transformer-patch-seeds.yaml --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/24_patch_transformer_seeds.ipynb
```

The aggregate is saved under artifacts/transformer-patch-seed-comparison.
Automated HTML export skips setup, preparation, training, and recursive export.
Source notebooks remain output-free. Executed HTML, predictions, and checkpoints remain local under artifacts.
Curate the reviewed multi-seed findings under docs/experiments after training.

A repeated macro-F1/REM gain would support retaining the patch recipe for the next focused question.
Then consider short-history handling or a bounded patch-size comparison.
A mixed result should remain a mixed result; do not select a favorable seed or silently combine new settings.
Do not inspect final-test observations during this replication.

## Local preparation evidence

On 2026-11-13, only the two new candidate studies were prepared without model optimization.
All 260 copied feature CSV files match the seed-42 patch folds byte for byte.
Resolved model recipes differ only by training seed. Fold assignments, device, and diagnostic controls are unchanged.
All four reused grouped artifacts passed the public reused-result validation.
Notebook 24 exported from a fresh kernel with optimization skipped and ten successful MPS forward checks.
The checks use 2,756-parameter patch models with raw input shape 64 × 30 × 6 and output shape 64 × 4.
The report shows seed 42 complete and the two new candidates not_started; no missing scores are invented.
All 522 tests, Ruff lint, and whitespace checks pass.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.
