# Controlled Transformer capacity experiment

Status: prepared for Timo to execute. No participant-data capacity training has run.

## Question

Does a wider single-layer Transformer improve development macro-F1 on the same held-out epochs?
The [previous comparison](experiments/transformer-grouped-2026-11-09/README.md) improved four folds but only half the participants.
Deep recall improved more than REM recall. Larger models may increase overfitting rather than improve these stages.

| Setting | Measured reference | Candidate |
| --- | ---: | ---: |
| Representation width | 16 | 32 |
| Feedforward width | 32 | 64 |
| Parameters | 2,564 | 9,220 |
| Layers | 1 | 1 |
| Heads | 2 | 2 |
| Context epochs | 10 | 10 |
| Residual/feedforward dropout | 0.1 | 0.1 |
| Attention-weight dropout | 0 | 0 |

Change the two widths together as one capacity recipe. Head dimension also increases because the head count stays fixed.
This cannot isolate representation width from feedforward width or head dimension.
Keep six features, coverage rules, folds, seed 42, MPS, batch 64, Adam learning rate 0.001, and gradient clipping one.
Keep patience five, twenty epochs, and validation macro-F1 checkpoint selection.
Use the completed width-16 Transformer as the reference. Do not retrain it or change its artifacts.
OPAL stays disabled. Maximum new work is five fits and 100 epochs.

## Run locally

Open [notebook 13](../notebooks/13_transformer_capacity.ipynb).
Run the preflight and inspect the fold plan first.
Enable RUN_PREPARATION once for the new output directory.
Then enable RUN_TRAINING and execute the training cell yourself.
Keep both flags disabled for report-only execution.

Equivalent commands from the repository root:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer-width-32.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-capacity-grouped-validation.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-capacity-grouped-validation.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-capacity-grouped-validation.yaml --train
```

Preparation freezes new fold configurations and hashes. Existing output directories are refused.
If the manifest already exists on this checkout, leave preparation disabled.
Fresh clones need source data and the measured reference artifacts to reproduce the paired report.
The notebook reports missing artifacts explicitly. Curated Git summaries cannot replace verified predictions.

## Decide from the measured result

Use the mean paired fold macro-F1 difference as the primary comparison.
Report all folds, pooled class precision/recall/F1, and participant differences.
Show how many participants improve, including losses hidden by pooled scores.
Check Wake and Light trade-offs when interpreting Deep and REM improvements.
Do not select individual folds or checkpoints after inspecting class outcomes.
Do not retune different folds independently.

A positive mean with mixed participant results is provisional evidence.
A negative result rejects this capacity recipe under the fixed training budget, not all larger Transformers.
Check training curves before attributing a difference to capacity alone.
The learning rate is held fixed; this is not a separately optimized comparison for each size.
Consider repeated seeds before adopting a winner. Do not expand the search automatically.
The folds select checkpoints and share training participants, so no independent-test or significance claim follows.
Repeated development tuning increases selection bias. The final three test participants stay reserved.

## Reports

Notebook 13 saves the validated paired JSON under artifacts/transformer-capacity-comparison.
Its plots compare width 16 with width 32 using the shared reporting implementation.
Export HTML without optimization:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/13_transformer_capacity.ipynb
```

Only output-free source notebooks are tracked. Curate measured results after Timo completes the experiment.

## Local preparation evidence

On 2026-11-09, the width-32 MPS preflight passed without optimization.
The full grouped plan exactly matched the measured width-16 plan.
Five fold tables and their manifest were prepared without training.
Only the output path and the two width settings differ in the model recipe.
On this checkout, leave RUN_PREPARATION disabled and enable RUN_TRAINING when ready.
Notebook 13 was executed from a fresh kernel and exported as HTML.
It correctly reported the reference as complete and the candidate as not_started.
