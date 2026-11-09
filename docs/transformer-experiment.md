# Small causal Transformer comparison

Status: completed by Timo on local MPS. See the [measured results](experiments/transformer-grouped-2026-11-09/README.md).

## Question and controls

Does a small attention model improve the grouped development results over the fixed GRU recipe?
Reuse the same 26 development participants, five fold assignments, six features, and coverage rule.
Keep seed 42, batch size 64, context ten, Adam learning rate 0.001, gradient clipping one, patience five, and twenty-epoch cap.
Use the same resolved device in both experiments. The measured local GRU reference used MPS.
The candidate runs five fresh models and fits a fresh training-only scaler in every fold.
Maximum work is five fits and 100 epochs. OPAL is disabled.

## Architecture

| Setting | GRU reference | Transformer candidate |
| --- | --- | --- |
| Temporal mechanism | One unidirectional GRU | One causal attention encoder |
| Width | 32 hidden units | 16 representation dimensions |
| Attention heads | Not applicable | 2 |
| Feedforward width | Not applicable | 32 |
| Context | Up to 10 completed epochs | Up to 10 completed epochs |
| Positions | Recurrent order | Learned positions within the window |
| Residual/feedforward dropout | None | 0.1 |
| Attention-weight dropout | Not applicable | 0 |
| Parameters | 3,972 | 2,564 |

This is a model-recipe comparison, including capacity and regularization differences.
It cannot isolate attention as a single causal factor or establish that Transformers generally outperform GRUs.
Do not tune the candidate separately on each fold.

The model projects the six features, adds learned position embeddings, and applies causal attention plus a padding mask.
It classifies the final valid timestep. Future timesteps cannot influence earlier representations.
Padding remains excluded from attention keys and checkpoint restoration returns evaluation-mode models.
The implementation uses [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoder.html)
and [TransformerEncoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html).
Synthetic tests verify causal behavior, padded/unpadded equivalence, checkpoint restoration, and local MPS forward/backward updates.
The default training-mode test uses a padded 64 × 10 × 6 batch.
Attention-weight dropout is fixed to zero on every device because this MPS runtime rejects nonzero attention dropout.
Residual and feedforward dropout remain 0.1; there is no silent CPU fallback.
CUDA remains unverified. Cross-device numerical equality is not promised.

## Reused engineering interfaces

ModelFactory validates architecture settings and creates models and checkpoint metadata.
The dataset, collation, scaling, optimization loop, early stopping, and grouped runner remain shared.
The generic neural_training entry point exposes that existing trainer; legacy gru_training commands remain valid.
Old GRU checkpoint files and prediction columns remain readable.
New Transformer files use model_checkpoint.pt and a generic prediction column.
This is a focused shared interface, not a broad package redesign.

## Run the candidate

Inspect configs/transformer.yaml and configs/transformer-grouped-validation.yaml first.
Use notebook 12, or run these commands from the repository root.

Preflight without optimization:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer.yaml --check
```

Inspect the grouped plan:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-grouped-validation.yaml --plan
```

Prepare new fold tables without training:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-grouped-validation.yaml --prepare
```

Start training when ready:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-grouped-validation.yaml --train
```

Existing output directories are refused. Preparation must be repeated in a fresh output directory after settings change.
Source feature tables and measured GRU results remain unchanged.

## Compare matched results

After all folds finish:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/gru-grouped-validation/summary.json \
  --candidate artifacts/transformer-grouped-validation/summary.json \
  --output artifacts/transformer-comparison/summary.json
```

The report verifies metric hashes, fold membership, common settings, coverage, provenance, and exact participant/time/target alignment.
It recomputes scores from saved predictions and supports both legacy and generic prediction columns.
Report paired fold differences, pooled class scores, and participant differences rather than selecting only the best fold.
Mean differences and sample deviations are descriptive. Fold training sets overlap; they are not independent trial replicates.
Both models select checkpoints on their development validation folds. These remain development results, not independent outer-test estimates.
The final test participants remain reserved.

## Runtime and future refactoring

Keep the existing Python environment for this comparison.
Treat Python 3.14 as a separate migration with wheel, MPS, notebook-kernel, and target-platform checks.
Use SOLID and DRY to identify real shared responsibilities, rather than adding a class or interface for every operation.
Meaningful specialized methods can remain when they clarify a responsibility.
Broader data/configuration refactoring should follow the interfaces that stabilize across these model families.

## Local preparation evidence

On 2026-11-06, the candidate folds were prepared on MPS without optimization.
Their complete participant plans match the measured GRU plans exactly.
The default-model and fold-one preflights passed with 64 × 10 × 6 inputs and four output logits.
Fold one contains 18,121 eligible training and 4,587 validation epochs.
The candidate has 2,564 parameters. At preparation time, no participant-data Transformer checkpoint or classification result existed.
The five candidate folds are now complete. Do not rerun training into their existing output directories.
The existing gru dependency group supplies the shared PyTorch runtime; no new training library is required.
