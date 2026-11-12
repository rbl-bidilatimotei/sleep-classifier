# Small causal patch Transformer

Status: prepared for Timo to train locally. No participant-data patch-model training result exists yet.

## Why change representation now?

The lower constant rate and plateau schedule improved some probability-loss behavior, but lost stage-classification scores.
The [scheduler review](experiments/transformer-plateau-2026-11-12/README.md) also showed smoother trajectories without improved macro-F1.
Stop that local optimizer grid and test a different temporal representation.
The question is whether jointly embedding short ordered subsequences helps stage detection across all development participants.

The design borrows temporal patching from [PatchTST](https://arxiv.org/abs/2211.14730).
That paper also uses channel-independent encoding and studies forecasting and representation learning.
This model jointly embeds six related wearable features and predicts a completed sleep-stage interval.
It is a PatchTST-inspired causal classifier, not a faithful reproduction or a transfer of the paper's forecasting performance claims.
See the [official implementation](https://github.com/yuqinie98/PatchTST) for the original design.

## Architecture and causal contract

Thirty completed thirty-second epochs provide fifteen minutes of history.
A nonoverlapping patch contains three ordered epochs, or ninety seconds.
Flatten their six-feature vectors in chronological order to form eighteen input values per token.
A learned linear projection maps each token to width sixteen. We do not average away the three epoch vectors.

```mermaid
flowchart LR
    A[30 completed epochs × 6 features] --> B[10 chronological patches × 18 values]
    B --> C[Linear projection + learned patch positions]
    C --> D[One causal Transformer layer, width 16]
    D --> E[Final valid patch → four stage logits]
```

The remaining attention settings stay fixed: two heads, feedforward width 32, GELU, and residual/feedforward dropout 0.2.
Attention-weight dropout stays zero for the existing MPS implementation.
The patch wrapper reuses the existing causal Transformer backbone.

Each history is right-aligned to its prediction target before patch formation.
The newest valid raw epoch always occupies the final slot of the final valid patch.
For five valid epochs x0 through x4 and patch size three, the tokens are:

    [zero, x0, x1], [x2, x3, x4]

The leading zero represents missing early history in already normalized feature space.
It is not a synthetic sensor observation or a future value.
There is no extra within-patch missingness indicator; partial-history treatment is an explicit part of this architecture.
Right-padded raw values are masked to zero before projection and cannot affect a valid prediction.
Padded patch keys and future patch keys are excluded from attention.
Each patch may mix all of its completed epochs. Only the final valid patch produces a target prediction.
Thus the causal claim applies at the completed target's end, not at every hypothetical instant within an unfinished patch.

Keep participant boundaries, gap-induced sequence breaks, and training-only scaling unchanged.
Do not use centered windows, future labels, per-night future statistics, or bidirectional future context.
Cold starts and histories shorter than three epochs remain eligible under the existing rules.
The architecture groups the same available inputs; it does not change eligibility or add sensor information.

## Fixed comparison

| Setting | Measured reference | Candidate |
| --- | --- | --- |
| Model type | transformer | patch_transformer |
| Raw context | 30 epochs | 30 epochs |
| Features | Six causal summaries | Same six |
| Temporal tokens at full history | 30 | 10 patches |
| Epochs per token | 1 | 3 |
| Width / heads / layers / feedforward | 16 / 2 / 1 / 32 | Unchanged |
| Dropout | 0.2 | 0.2 |
| Adam rate / scheduler | 0.001 / none | Unchanged |
| Epoch cap / patience / selection | 40 / 12 / validation macro-F1 | Unchanged |
| Trainable parameters | 2,884 | 2,756 |

The input projection grows from six to eighteen inputs, while the position table shrinks from thirty to ten entries.
Their net parameter change is −128. There are no extra parameters in the patch-packing operation.
This is an architecture-recipe comparison, not a parameter-matched proof that patching alone causes any score change.
The new initialization layout also changes the seeded weight draws, despite the shared training seed.
The attention score map shrinks from 900 to 100 entries per head at full history.
That does not establish a ninefold training speedup. Measure runtime and do not compare instrumented fits as controlled speed benchmarks.

## Evaluation scope and artifacts

Use all five original development folds, split seed 42 and training seed 42.
Reuse the completed constant-rate dropout-0.2 reference:

    configs/transformer-dropout-grouped-validation.yaml
    artifacts/transformer-dropout-grouped-validation/summary.json

Train only five fresh candidate fits, capped at 200 total epochs.
The full comparison scores the same 22,708 eligible targets from 26 development participants.
Three final-test participants remain reserved. The candidate's scalers fit each fold's training participants only.
Validation also selects checkpoints, so these are development scores rather than an independent final-test estimate.

The original full reference did not record per-epoch diagnostics.
The candidate enables the same optional evidence collection verified in the diagnostic repeat.
That read-only instrumentation preserved exact original MPS trajectories on the two measured diagnostic folds.
It is an observability addition, not an optimizer intervention. Do not fabricate unavailable reference epoch evidence for other folds.
The original prediction CSVs and selected checkpoint metrics remain the primary paired reference.

The notebook verifies every reference metric hash, saved score, target population, current fold plan, and resolved device.
It requires all training controls and Transformer dimensions to remain fixed, with only the stated architecture and observation controls differing.
The grouped manifest then freezes the source configuration and prepared feature hashes.

## Run locally

Open [notebook 23](../notebooks/23_causal_patch_transformer.ipynb).
Read the What/Why explanation before each code cell.
Inspect the five-fold plan, architecture, parameter counts, and raw-context versus patch-token distinction.
Keep setup disabled in the existing environment.
Prepare the candidate only once, then run its forward checks without optimization.
Enable RUN_TRAINING when ready. Keep RUN_PREPARATION false after the manifest exists.
The reference is reused and is never automatically retrained.

Fresh clones require source development features and the completed five-fold reference artifacts.
Use MPS for the paired comparison with the measured local reference.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-grouped.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-grouped.yaml --prepare
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/transformer-patch-grouped-validation/fold-01/gru.yaml --check
```

The notebook checks all five prepared folds. Start optimization only through the explicit command or gated cell:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-patch-grouped.yaml --train
```

Automatic resume is unsupported. Existing results block a rerun; inspect partial evidence before choosing recovery.
After all folds complete:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/transformer-dropout-grouped-validation/summary.json \
  --candidate artifacts/transformer-patch-grouped-validation/summary.json \
  --comparison architecture --output artifacts/transformer-patch-grouped-validation/paired-comparison.json
uv run --locked --group gru --group notebook python -m sleep_classifier.experiment_evidence \
  --output-dir artifacts/transformer-patch-grouped-validation --report
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/23_causal_patch_transformer.ipynb
```

Automated HTML execution skips setup, preparation, training, and recursive export.
Keep source notebooks output-free. Large predictions, checkpoints, and executed HTML stay local under artifacts.
Curate small reviewed results under docs/experiments after Timo runs training.

## Decision criteria

Primary: paired mean fold macro-F1 and each individual fold difference.
Then inspect participant gains/losses, Wake/Light/Deep/REM precision, recall, and F1, accuracy, confusion matrices, and selected probability loss.
Check startup behavior and stage-transition errors, because coarser tokenization can discard useful fine-scale distinctions.
Do not promote a model from one favorable fold, a smoother curve, or a lower pooled NLL alone.
A useful first-seed result motivates seed replication; it is not a final-test claim.
A negative result is informative about this patch size and joint-channel architecture, not all PatchTST variants.

Tests require patch-size-one equivalence, chronological packing, correct partial histories, padding invariance, causal attention, finite gradients, and checkpoint restoration.
The shared training/inference interfaces continue accepting six-feature raw epoch histories; patching remains inside the model.

## Local preparation evidence

On 2026-11-12, all five candidate folds were prepared without model optimization.
All 130 copied participant feature CSV files match the original full-fold reference byte for byte.
Resolved recipes preserve every training setting after excluding model type, patch size, and validated observation controls.
All five MPS no-gradient forward checks passed with 2,756 parameters, raw input shape 64 × 30 × 6, and four output logits.
The model internally packs the full history into ten tokens with eighteen values each.
Notebook 23 exported from a fresh kernel with training and preparation skipped.
All 517 tests, Ruff lint, and whitespace checks pass. No candidate model result directories exist before Timo runs training.
On this checkout, keep RUN_PREPARATION false and enable RUN_TRAINING when ready.
