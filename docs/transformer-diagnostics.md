# Instrumented seed-42 Transformer repeat

Status: completed locally by Timo and reviewed on 2026-11-12.
See the [measured results](experiments/transformer-diagnostics-2026-11-12/README.md).
Keep preparation and training disabled when rebuilding the completed report.

## Question

Which targets, classes, and participants contribute to validation-loss rebounds near epochs thirteen through seventeen?
The [dropout replication](experiments/transformer-dropout-seeds-2026-11-11/README.md) found no common epoch-fourteen switch.
Earlier runs retained scalar histories and selected checkpoints. Those cannot reconstruct the probability changes behind a historical rebound.
This repeat collects the missing evidence without changing the model recipe.

## Fixed recipe and bounded scope

Use seed 42, context thirty, width sixteen, dropout 0.2, and the original five-fold split seed 42.
Keep one layer, two heads, feedforward width 32, batch 64, and the same six causal features.
Keep plain Adam learning rate 0.001, unweighted cross-entropy, natural sampling, and gradient clipping at one.
Keep the forty-epoch cap, patience twelve, and validation macro-F1 checkpoint selection.
The three final test participants remain reserved. OPAL stays disabled.

Select only original folds 01 and 04 after generating the original five-fold assignments.
Do not generate a new two-fold split. The study requires two fresh fits and at most eighty total epochs.
Their eleven held-out people are a diagnostic subset, not a complete 26-participant validation result.
Do not rank this subset's pooled score against complete five-fold results.
The selection is post-hoc and targets a debugging question; it is not an independent performance benchmark.

The diagnostic configuration adds evidence controls and output locations only.
The notebook verifies recipe equality against the measured seed-42 dropout-0.2 recipe before preparation or training.
It also checks both selected partitions against their original fold definitions.

## Evidence collected

| Evidence | Frequency | Purpose |
| --- | --- | --- |
| Target probabilities, predicted/true stage, confidence, NLL, participant, and time | Every completed epoch, compressed CSV | Reconstruct which predictions changed and inspect any observed rebound |
| Class and participant NLL sums, means, counts, and pooled contributions | Every completed epoch, JSON | Separate group size from group difficulty and decompose total loss |
| Class precision/recall/F1 and four-class confusion matrix | Every completed epoch | Check stage trade-offs alongside changes in probability loss |
| Training NLL in evaluation mode | Every completed epoch, extra read-only training pass | Compare fixed end-of-epoch training and validation weights with dropout disabled |
| Ordinary stochastic training loss | Every completed epoch | Retain the original optimization history; this includes changing weights and active dropout |
| Actual learning rates and pre-clipping gradient mean/max/clipped-step fraction | Every completed epoch | Check optimizer behavior and whether gradients frequently hit the clipping limit |
| Portable intermediate checkpoints | Completed epochs 13, 14, 15, 16, and 17 | Re-evaluate current states before best-checkpoint restoration |
| Recipe, provenance, device, and runtime versions | Once per fit | Identify the evidence and reproduce its context |

The recorder reuses the existing validation pass. It does not add another validation forward pass.
The extra training evaluation pass adds compute, synchronization, and disk overhead. This is a diagnostic run, not a speed benchmark.
A separate DataLoader generator protects the training random stream during the extra pass.
Synthetic active-dropout tests require exact equality of histories, selected weights, predictions, best epoch, and final Torch RNG state.
This verifies the current CPU implementation. It does not promise bitwise cross-device or cross-version reproduction.
Intermediate checkpoints are evaluation snapshots, not optimizer/RNG snapshots for exact training resume.

Early stopping remains active. If a fold stops before an inspected epoch, the notebook reports that comparison as unavailable.
A completed epoch record is written after its linked artifacts. Files use atomic replacement and include content hashes.
Earlier complete records remain inspectable after a later failure. Interrupted temporary files do not become completed epoch records.
The grouped runner refuses existing results; automatic training resume remains unsupported.

## Run locally

Open [notebook 20](../notebooks/20_transformer_diagnostics.ipynb).
Read the short What/Why explanation before every code cell.
Keep RUN_SETUP false for the existing project environment.
Inspect the fixed recipe, original fold assignments, and two-fit budget.
Enable RUN_PREPARATION only if the new study directory has not been prepared.
Run both fold preflights. They perform forward passes without optimization.
Enable RUN_TRAINING and run the training cell when ready.
Leave preparation disabled once its manifest exists.

Commands for a fresh study:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-diagnostics-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-diagnostics-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-diagnostics/fold-01/gru.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-diagnostics/fold-04/gru.yaml --check
```

Start training only through the explicit command or gated notebook cell:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-diagnostics-grouped.yaml --train
```

Inspect evidence during or after the run without training:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.experiment_evidence \
  --output-dir artifacts/transformer-diagnostics --report
uv run --locked --group gru --group notebook python -m sleep_classifier.experiment_evidence \
  --output-dir artifacts/transformer-diagnostics --report \
  --compare-fold fold-04 --before-epoch 15 --after-epoch 16
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/20_transformer_diagnostics.ipynb
```

The automated HTML export skips setup, preparation, training, and recursive export.
It can inspect existing partial or complete evidence. It never starts optimization.
Source notebooks remain output-free. Executed HTML and large diagnostic artifacts stay local under artifacts.
Track small curated findings under docs/experiments after the run is reviewed.

## Reading the loss decomposition

NLL averages the negative natural logarithm of each true-class probability. Lower is better.
The per-target evidence uses a 1e-12 numerical floor before the logarithm.
The original logits-based criterion loss is retained separately as validation_criterion_nll.
The two can differ under extreme probability underflow, so inspect both when diagnosing large losses.

A participant's contribution is its NLL sum divided by the complete fold target count.
Contributions add to the pooled fold NLL. Class contributions add independently to the same total.
Do not add the participant and class decompositions together; they describe the same targets in two ways.
A small group can have high average loss and still contribute little to the total rise.

The epoch comparison joins identical participant/time/target identities and checks all linked hashes.
For each group, contribution_delta = (after NLL sum - before NLL sum) / total targets.
Positive values increase the pooled loss; negative values offset it. The contributions sum to the total NLL difference.
The largest-target table ranks absolute changes, including both improvements and regressions.
Use the sign when interpreting the table. The top twenty targets need not explain the entire change.
Confidence on errors uses changing error sets. It is descriptive, not a calibration score or a controlled matched-error comparison.
High-confidence error fraction counts incorrect predictions with maximum probability at least 0.8, divided by all validation targets.

Inspect epochs thirteen through seventeen first, then compare any other observed rebound through the CLI or class interface.
Check whether the rise concentrates in a few participants, particular true stages, or widespread increasingly confident errors.
Compare fixed-weight training and validation NLL before interpreting the generalization gap.
Gradient norms describe the training updates before clipping, not validation gradients.
Keep these diagnostic measures separate from macro-F1 checkpoint selection.

## Local evaluation foundation

The first implementation is an evidence store and inspector, usable across the supported neural model architectures.
The recorder owns the schema. The trainer supplies observations. The report reads and verifies artifacts without importing the trainer.

```text
Grouped experiment
  -> fixed-recipe trainer
     -> epoch diagnostics JSON + prediction CSV.gz + selected intermediate .pt
  -> evidence inspector
     -> status and curves
     -> matched epoch comparisons
     -> notebook tables and local report files
```

This provides the data contract for a later local evaluation workspace.
Useful later views include run selection, participant/stage filters, prediction timelines, confidence plots, and checkpoint comparisons.
Add an interactive interface after the first instrumented run confirms which views are useful.
Keep the artifact reader and metrics independent of that interface.
An experiment browser does not need to change the training pipeline or add a hosted service.
Docker, inference APIs, and economical CI remain separate roadmap milestones.

## Interpretation limits and next decision

A fixed-recipe repeat can locate contributions to a loss rebound. It cannot prove the biological or statistical cause by itself.
Check whether its scalar history matches the earlier run before linking new evidence to the old epoch-fourteen observation.
Runtime changes and device kernels can affect numerical repeatability even with the same seed.
Avoid selecting another learning rate, sampler, architecture, or checkpoint objective from these two folds alone.
After reviewing the evidence, choose one bounded intervention or return to the planned small causal patch encoder comparison.

## Historical preparation evidence

The following records the state before Timo ran the experiment.

On 2026-11-11, both selected folds were prepared without optimization.
All 52 copied participant feature CSV files match the corresponding original folds byte for byte.
The resolved model recipes match those original folds exactly after excluding the diagnostics section.
Both MPS forward checks passed with 2,884 parameters, input shape 64 × 30 × 6, and output shape 64 × 4.
Fold 01 has 18,121 training and 4,587 validation targets. Fold 04 has 18,281 training and 4,427 validation targets.
No result or epoch-evidence directory exists for either fold before training.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.

The prepared-state notebook exported successfully from a fresh kernel with optimization cells skipped.
Its status tables show both folds prepared and epoch comparisons pending.
The synthetic evidence figure was rendered and visually inspected; it is not a participant-data training result.

All 373 tests pass, including populated notebook views, signed loss decomposition, and unchanged synthetic training trajectories.
Ruff lint and whitespace checks pass. The report CLI leaves unprepared output paths absent, so inspection cannot block later preparation.

## Completed review

Both trajectories and selected checkpoints exactly match the original seed-42 results.
Fixed-weight training NLL falls while validation NLL rises, supporting overfitting in probability quality.
Matched persistent errors lose true-class probability; different people and stages drive different spikes.
The next proposal tests learning rate 0.0003 on the same two folds with all other settings fixed.
