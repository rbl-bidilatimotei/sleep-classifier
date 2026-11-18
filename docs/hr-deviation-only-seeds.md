# Replicate the availability-flag ablation across seeds

Status: prepared for local training by Timo. New deviation-only seeds 43 and 44 have no scores yet.

## Question

The [seed-42 flag ablation](experiments/hr-deviation-only-2026-11-18/README.md) loses mean fold macro-F1 by 0.00577.
Four folds decline; fold 05 improves. REM and Deep recall rise, but precision falls, reducing their F1 scores.
Light loses correct predictions, while Wake F1 and full-history macro-F1 are nearly unchanged.
Keep the full-HR recipe for now. Repeat the exact ablation before concluding whether this smaller, mixed effect persists.

This study tests uncertainty about the flag's conditional contribution, rather than searching for a lucky better score.
The earlier full-HR model already varies across training seeds. One small negative effect does not prove necessity or irrelevance.

## Reuse and new work

| Seed | Full-HR reference | Deviation-only candidate | New fits |
| --- | --- | --- | ---: |
| 42 | Reuse completed result | Reuse notebook 32 result | 0 |
| 43 | Reuse completed result | Train five new folds | 5 |
| 44 | Reuse completed result | Train five new folds | 5 |

Ten new fits are capped at 400 epochs in total. Do not retrain any reference or the measured seed-42 candidate.
Missing or inconsistent reused evidence must block the study rather than trigger automatic retraining.
Only training seeds and output paths change from the measured deviation-only candidate.

## Fixed recipe

- Availability mask: window_features.masked_channels contains only heart_history_available
- Original six prepared features and unchanged standardized HR deviation
- Thirty raw epochs, ten-epoch past reference, and minimum four previous epochs
- Eight internal channels, 2,852 nominal parameters, and the same parameter layout in each pair
- P3, width 16, two heads, one layer, feedforward width 32, and dropout 0.2
- Original Adam 0.001, no scheduler or agent, unsmoothed unweighted loss, and natural sampling
- Forty epochs, patience twelve, batch 64, norm clipping one, and macro-F1 checkpoint selection
- The same diagnostics, intermediate checkpoints, and MPS device

Fit normalization separately on each fold's training participants. Corresponding reference and candidate states must match exactly.
Keep unavailable deviations zero, retain all eligible targets, and restore the saved mask and state during inference.
The control still receives valid lengths and positional information, so availability remains derivable from existing inputs.
Zeroing its explicit channel changes representation rather than sensor information.
Its projection weights become inactive; equal nominal parameter counts do not imply equal functional capacity.

Split seed remains 42. All six configurations preserve the same five participant folds, 26 development people, and three reserved test participants.
Training seeds affect initialization, minibatch order, and dropout. They do not create new participant cohorts.

## Run locally

Open [notebook 33](../notebooks/33_hr_deviation_only_seeds.ipynb).
Read each What/Why explanation and inspect the reuse matrix, fixed-recipe guards, and ten no-gradient checks.
Keep RUN_SETUP false in the current environment.
Keep RUN_PREPARATION false once both candidate manifests exist.
Set RUN_TRAINING true when ready. Run the training cell interactively.
It trains candidate seed 43, then candidate seed 44. It never trains reused results.

Each operation derives its config from a validated experiment object.
Training logs its config, output directory, and masked channel, then calls the shared readiness guard before launching the CLI.
A failed readiness check or command stops the loop. Existing outputs remain intact; automatic resume is unsupported.
After partial failure, inspect retained artifacts before choosing recovery. Do not blindly rerun the entire two-candidate loop.

Fresh clones require development features and these measured grouped artifacts:

- artifacts/transformer-window-local-hr-grouped-validation
- artifacts/transformer-window-local-hr-seeds/seed-43/candidate
- artifacts/transformer-window-local-hr-seeds/seed-44/candidate
- artifacts/hr-deviation-only-grouped-validation

Use MPS to match those references. The notebook blocks a device mismatch instead of silently mixing backends.
Candidate outputs use artifacts/hr-deviation-only-seeds/seed-43/candidate and the corresponding seed-44 directory.
Study outputs use artifacts/hr-deviation-only-seed-comparison.

Plan without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_seed_report \
  --config configs/hr-deviation-only-seeds.yaml --plan
```

Prepare each new candidate only when its output directory does not exist:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/hr-deviation-only-seed-43-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/hr-deviation-only-seed-44-grouped.yaml --prepare
```

The notebook checks every prepared fold. An explicit standalone training command for one candidate is:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/hr-deviation-only-seed-43-grouped.yaml --train
```

Use the matching seed-44 config for its training command.

## Evaluation and decision

Use the explicit window_feature_ablation policy in the shared-seed reporter.
Require the same declared mask across folds and seeds, and identical controls apart from the mask and storage paths within each pair.
Validate saved metric hashes, actual prediction scores, populations, coverage, fitted state, constructor metadata, budgets, and device.
The notebook also compares configured settings with recorded reused recipes before any new work.

Report candidate-minus-full-HR mean fold macro-F1 for each seed, then the three-seed descriptive mean and sample deviation.
Negative effects favor retaining the flag. Do not replace missing candidate scores with zero or publish the full aggregate early.
Show every fold and participant effect beside stage precision, recall, F1, support, confusion matrices, NLL, and saved curves.
Pay particular attention to Light losses and Deep false positives. A recall increase alone does not establish improved classification.

Use the same history bins: 1–4, 5–29, and full thirty-epoch context. Keep stage transitions as an overlapping secondary slice.
Do not average slice macro-F1 scores to reconstruct pooled scores or attribute a whole fold effect to startup from pooled slices alone.
The full aggregate requires all three completed pairs.

If primary losses repeat, retain the flag and stop this ablation series.
If effects vary or average near zero, report the uncertainty rather than claiming equivalence or selecting a favorable seed.
Keep the full-HR working reference until the complete comparison is reviewed.
Repeated development participants do not make fifteen independent population trials; the sample deviation is not a confidence interval.
Keep the final test untouched. No new participant-data training is authorized during preparation.

Export the HTML report without training:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/33_hr_deviation_only_seeds.ipynb
```

Export skips setup, preparation, training, and recursive export. Keep source outputs clear and generated HTML under ignored artifacts/notebooks.

## Preparation evidence

The [preparation snapshot](experiments/hr-deviation-only-seeds-preparation-2026-11-18/README.md) records ten actual MPS checks and matching fitted states.
All 937 tests, Ruff, whitespace checks, and fresh-kernel HTML export pass.
No new participant training ran. Keep RUN_PREPARATION false in the current checkout; set RUN_TRAINING true when ready.
