# Patch-size comparison

Status: completed locally by Timo and reviewed on 2026-11-13.
The [measured comparison](experiments/patch-size-comparison-2026-11-13/README.md) loses mean fold macro-F1 despite a small REM F1 gain.
Retain the replicated patch-size-three reference.

## Question and evidence

The [three-seed replication](experiments/patch-transformer-seeds-2026-11-13/README.md) supports patch size three as the working development reference.
All three seeds gain mean fold macro-F1, and every seed improves REM precision, recall, and F1.
However, Deep F1 declines for two seeds. Short-history and Deep-transition scores decline in every seed.
These small post-hoc slices identify a question, rather than establish a causal explanation.

Test whether finer patches retain these gains while improving temporal detail.
Use two epochs per patch instead of three. Do not change the raw history or optimizer.
This experiment compares architecture recipes; it cannot isolate a single mechanism from initialization and parameter-layout changes.

## Fixed protocol

| Setting | Reused reference | New candidate |
| --- | --- | --- |
| Epochs per patch | 3 | 2 |
| Seconds per full patch | 90 | 60 |
| Full-history tokens | 10 | 15 |
| Trainable parameters | 2,756 | 2,740 |
| Raw context / inputs | 30 completed epochs / six features | Same |
| Training seed / split seed | 42 / 42 | Same |
| Development folds | All five original folds | Same |
| Width / heads / layers / feedforward | 16 / 2 / 1 / 32 | Same |
| Dropout / Adam rate | 0.2 / 0.001 | Same |
| Scheduler / weighting / sampling | None / unweighted / natural | Same |
| Epoch cap / patience | 40 / 12 | Same |
| Checkpoint selection | Validation macro-F1 | Same |
| Diagnostics | Enabled, including training evaluation | Same |

The input projection shrinks while the position table grows. The net change is sixteen fewer parameters.
Finer patches require more attention tokens. Do not assume a training speed improvement.
The newest valid epoch stays in the final slot of its patch. Partial-history padding remains causal.

Reuse `artifacts/transformer-patch-grouped-validation/summary.json`.
Prepare only `configs/transformer-patch-size-2-grouped.yaml`.
The candidate writes to `artifacts/transformer-patch-size-2-grouped-validation`.
Maximum new work is five fits and 200 epochs. The three final-test participants remain reserved.

## Run locally

Open [notebook 25](../notebooks/25_patch_size_comparison.ipynb).
Read each What/Why note before its code cell.
Keep setup disabled in the current environment.
Prepare the candidate once. Keep preparation disabled after its manifest exists.
Review the fold plan and no-gradient forward checks.
Enable training only when ready.
The notebook reuses the reference and never starts reference training.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-size-2-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-size-2-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-patch-size-2-grouped-validation/fold-01/gru.yaml --check
```

The notebook checks all five folds. Start training through its gated cell or this explicit command:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-size-2-grouped.yaml --train
```

Existing results block accidental reruns. Automatic resume is unsupported.
Fresh clones also need the completed reference artifacts and prepared source features.

## Interpretation

Primary: compare paired mean fold macro-F1, then inspect all five fold differences.
Report participant differences and pooled class precision, recall, F1, and support.
Check accuracy, confusion matrices, selected NLL, and the epoch curves.
NLL measures the probability assigned to the true class. Lower NLL does not guarantee higher macro-F1.

Use the reusable temporal-slice report for both models on identical eligible targets.
History bins are disjoint: epochs 1–2, epochs 3–29, and full context at epoch 30 or later.
History counts restart at participant and gap-induced segment boundaries.
The transition slice overlaps these bins. It contains targets whose label changes from the preceding contiguous epoch.
Ground-truth previous labels define evaluation slices only. They are never model inputs.
Show support and each class score, especially Deep transitions and short-history Deep targets.
Small slices remain exploratory. Do not select a model from one favorable class or fold.

Keep checkpoint selection unchanged. Do not choose a different checkpoint after inspecting these slices.
A useful seed-42 result requires replication before replacing the working reference.
Repeated reuse of these development folds cannot establish independent final-test performance.

Export the report after training:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/25_patch_size_comparison.ipynb
```

The fresh-kernel HTML export skips setup, preparation, training, and recursive export.
Keep source notebooks output-free. Store executed HTML and large run files under ignored artifacts.

## Historical preparation evidence

These checks describe the state before Timo trained the candidate. The measured report supersedes pending status.

All five candidate folds were prepared on 2026-11-13 without optimization.
All 130 copied feature CSV files match the measured reference byte for byte.
Resolved fold recipes differ only in patch size. Both use MPS.
All five no-gradient checks pass with 2,740 parameters, raw input shape 64 × 30 × 6, and logits shape 64 × 4.
The fresh-kernel notebook HTML export completes with preparation and training skipped.
The temporal report reproduces the reference supports: 295 short-history targets, 2,279 intermediate targets, and 20,134 full-context targets.
Its overlapping transition slice contains 1,398 targets. Reference self-comparison gives zero score differences in every slice.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.

Validation: 546 tests pass. Ruff lint and whitespace checks pass. No candidate optimization ran.
