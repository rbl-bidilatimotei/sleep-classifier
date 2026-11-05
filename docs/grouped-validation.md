# Grouped development validation

Status: prepared for Timo to run. No measured fold-training results exist yet.

## Question

Does the fixed GRU recipe perform consistently across held-out development participants?
The earlier comparisons repeatedly used three validation participants.
Training seeds measure initialization variation; they do not replace participant-disjoint evaluation.

## Protocol

Use the 26 participants already assigned to baseline training and validation.
Keep the three final-test participants reserved.
Shuffle the sorted development participant IDs with split seed 42 and assign five folds.
Each participant appears in one validation fold and four training folds.
Fold sizes are six or five validation participants, with twenty or twenty-one training participants.
All windows from a participant stay in that participant's partition.

Use the unchanged 32-unit GRU, ten-epoch context, six features, Adam, seed 42, patience five, and twenty-epoch cap.
OPAL is disabled. Train a fresh model and fit a fresh training-only scaler for each fold.
Use the same resolved device across all folds.
At most five model fits and 100 training epochs are requested.
Do not alter the recipe after inspecting individual folds in this run.

The existing prepared feature tables are causal and unscaled.
Preparation copies their full rows and segment metadata into fold-specific derived directories.
It does not reload raw sensor files or final-test observations.
The original feature tables and baseline split remain unchanged.
Input and derived-file hashes guard against stale or modified preparation.

## Evaluation boundary

Each fold's validation data selects its best checkpoint and stopping time.
The pooled predictions are therefore development cross-validation results, not an unbiased independent outer-test estimate.
Do not present them as final-test performance or compare their pooled score directly with the earlier three-person score as an improvement.
Different participant populations are being evaluated.
For a later model comparison, use the same fixed folds and preparation protocol for both architectures.
Nested selection or separate stopping participants would be needed for independent outer-fold evaluation.

## Commands

Inspect the fold plan without loading feature observations or training:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/gru-grouped-validation.yaml --plan
```

Prepare fold-specific tables and configurations without training:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/gru-grouped-validation.yaml --prepare
```

Launch all five fits when ready:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/gru-grouped-validation.yaml --train
```

Use notebook 11 for optional environment setup, planning, preparation, training, and saved-result plots.
Training and preparation are skipped during HTML report export.
Preparation refuses existing output directories. Training requires unchanged prepared inputs and refuses existing fold results.
Failures preserve partial summaries. Resume is not supported; choose a fresh output directory for a full repeat.

## Read the results

Inspect each fold and participant, not only the average.
Report mean fold macro-F1 and its sample standard deviation alongside pooled eligible-epoch scores.
Fold means weight folds equally; pooled metrics combine all retained validation epochs.
Report coverage, class support, per-class precision/recall/F1, and confusion matrices.
A participant with more eligible epochs contributes more to pooled epoch metrics.
Do not treat overlapping epochs or repeated fold training sets as independent experimental replicates.

After this run, review failure patterns and decide whether to compare a small causal Transformer on these same folds.
Keep model capacity changes, feature changes, and evaluation changes distinguishable.

## Preparation verified locally

On 2026-11-05, all five development folds were prepared and frozen to MPS.
Fold sizes are 20/6, then four folds of 21/5 training/validation participants.
Fold one's no-gradient preflight passed with 18,121 training and 4,587 validation examples.
Its batch shape was 64 × 10 × 6, with four output logits and 3,972 parameters.
No participant-data optimization or grouped accuracy result exists yet.
On this checkout, skip preparation and enable only the notebook's training flag when ready.
A fresh checkout must prepare its own derived folders first.
