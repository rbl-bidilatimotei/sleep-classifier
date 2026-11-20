# Replicate the MPS motion effect across training seeds

Status: seeds 43 and 44 are prepared for local training. No new candidate model has been trained.

## Question and reused evidence

The [matched seed-42 comparison](experiments/fixed-lag-motion-mps-2026-11-20/README.md) improves mean fold macro-F1 by 0.00680.
Its gains are uneven across folds, with fold 05 contributing most. Selected NLL worsens, and startup slices decline.
Replicate the same motion recipe across initialization seeds before changing another setting.

Use training seeds 42, 43, and 44 on the same participant folds.
Reuse all three completed full-HR MPS references and the completed seed-42 MPS motion candidate.
Train only seed-43 and seed-44 motion candidates: ten new fits, capped at 400 completed epochs in total.
Keep the earlier CPU motion run intact and outside the seed aggregate.
No three-seed conclusion is available until both new candidates finish all five folds.

## Fixed controls

Change only the training seed and new output paths relative to the completed MPS candidate.
Require explicit MPS for both new candidates. The measured references also used MPS.
Keep split seed 42, all five participant folds, the same 26 development participants, and 22,708 eligible targets per seed.
Keep the final three test participants reserved. Do not read their observations.

Retain the immutable motion snapshot, one-second lag, 0.2-second maximum observation age, and within-epoch backward selection.
Retain seven raw inputs, nine effective channels, both ten-epoch HR channels, and minimum HR history four.
Fit scaling and motion imputation on each fold's training targets only. Do not introduce an availability flag or support cutoff.
Retain context 30, P3 patches, width 16, two heads, one layer, feedforward width 32, and dropout 0.2.
Retain forty epochs, patience twelve, batch size 64, learning rate 0.001, gradient clip 1.0, weighted cross-entropy, and zero-decay Adam.
Retain the diagnostics and best-validation-macro-F1 checkpoint rule. Add no scheduler, smoothing, sampling change, or agent.
Each motion candidate has 2,900 parameters; its no-motion reference has 2,852.
That necessary projection difference remains part of the feature-recipe intervention.

## Configuration and artifact identity

The study config is configs/fixed-lag-motion-mps-seeds.yaml.
It declares only four reused runs and two new runs. All reference reuse flags are true; only the seed-42 candidate reuse flag is true.

| Seed | Reference | Motion candidate |
| --- | --- | --- |
| 42 | Reuse transformer-window-local-hr-grouped.yaml | Reuse fixed-lag-motion-mps-grouped.yaml |
| 43 | Reuse transformer-window-local-hr-seed-43-grouped.yaml | New fixed-lag-motion-mps-seed-43-grouped.yaml |
| 44 | Reuse transformer-window-local-hr-seed-44-grouped.yaml | New fixed-lag-motion-mps-seed-44-grouped.yaml |

New candidate artifacts live under artifacts/fixed-lag-motion-mps-seeds/seed-43/candidate and its seed-44 counterpart.
Study-level reports live under artifacts/fixed-lag-motion-mps-seed-comparison.
Existing references, seed-42 MPS motion, and CPU motion artifacts remain unchanged.

The seed reporter uses the strict motion_features paired policy. It does not expose a device exception.
A reused motion candidate is validated against its paired reference, rather than through the legacy six-input self-comparison.
Cross-seed signatures require matching motion definition, audit-config hash, and snapshot-manifest hash.
Fold-specific fitted values may vary across folds; they do not become source identity.
Other seed, recipe, fold, population, capacity, and source checks remain active.

## Run locally

Open [notebook 38](../notebooks/38_motion_mps_seeds.ipynb). Restart the kernel before its first run.
Each code cell explains what it does and why. Source execution switches default to false.
The ten folds are already prepared in this checkout. Keep RUN_SETUP and RUN_PREPARATION false here.
Enable RUN_CHECK for optional no-gradient checks. Enable RUN_TRAINING when ready for both new candidate runs.
The training cell derives each command from its validated experiment and checks the prepared device against MPS.
Existing results block accidental retraining. Preserve partial evidence if a run fails.

A fresh checkout needs development features, all reused results, and the matching verified motion snapshot and config.
Use the optional setup cells to clone the working branch and install the locked uv groups.
Prepare the two candidates only once:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/fixed-lag-motion-mps-seed-43-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/fixed-lag-motion-mps-seed-44-grouped.yaml --prepare
```

For report-only execution, keep every execution switch false.
The notebook can display the completed seed-42 pair while the other pairs remain pending.
It writes no complete three-seed aggregate until all pairs finish.
HTML export skips setup, preparation, training, and recursive export.

## Review after training

Compare each seed's mean paired fold macro-F1 first. Then inspect fold and participant effects.
Show pooled accuracy, precision/recall/F1 per stage, error destinations, NLL, and temporal slices.
Check whether fold-05 gains recur, whether Wake precision improves, and whether Light recall or startup performance declines.
Inspect both REM→Light and REM→Deep errors beside REM precision and recall.
Keep the original checkpoint rule rather than selecting an epoch after inspecting curves.

Report the mean and sample deviation of the three seed effects descriptively.
The same participants and folds repeat; fifteen seed-fold cells are not fifteen independent populations.
Neither a positive seed mean nor the seed deviation establishes final-test performance or a confidence interval.
Review replication before promoting the motion recipe or changing the architecture, loss, or regularization.
Timo runs all new training locally. This preparation performs no participant-data optimization.
