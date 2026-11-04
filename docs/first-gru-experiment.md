# First causal GRU experiment

Status: Timo completed the first local CPU run on 2026-11-03.
See the [recorded results](experiments/first-gru-2026-11-03/README.md).

## What the weighting experiment changed

Class weighting kept all 19,928 retained training epochs.
It weighted each sample's contribution to logistic-regression loss using training-class frequencies.
It did not undersample, oversample, or generate synthetic observations.
Per-class precision and recall remain valid metrics. Precision also depends on class prevalence.
Aggregate accuracy and weighted averages can hide poor minority-class performance.
The large recall trade-off was observed, not guaranteed by weighting.

Merged PR #6 records the measured comparison.
No additional EDA-only milestone is required before the first GRU experiment.

## Controlled starting point

Keep the six prepared sensor features, participant split, and coverage rule fixed.
Use unweighted cross-entropy for the first GRU run, matching the unweighted baseline objective.
The architecture has one unidirectional GRU layer with 32 hidden units and a four-class linear head.
There are 3,972 trainable parameters. No attention or bidirectional context is used.

Each example predicts one retained target using its current epoch and up to nine previous epochs.
The maximum context is ten epochs, covering five minutes.
Early examples use shorter histories. Right-padding is ignored through packed sequences.
All 19,928 training and 2,780 validation targets remain available, matching baseline coverage.
Sequences never cross participant boundaries, rejected intervals, unscored intervals, or missing timestamps.
Each window starts with a fresh recurrent state; windows are not a continuous carried-state streaming implementation.

A GRU improvement would combine architecture and history effects.
A later context-length-one GRU can isolate the contribution of longer history more directly.
Do not attribute any future gain entirely to temporal context without that comparison.

## Engineering boundaries

`CausalSequenceDataset` adapts prepared feature frames into PyTorch sequence examples.
`SequenceBatchCollator` pads variable-length batches and preserves sequence lengths.
`CausalGRUClassifier` converts the final recurrent representation into four logits.
`GRUExperiment` handles scaling, loaders, epochs, validation, and artifacts through separate methods.
`EarlyStopping` selects and restores the state with the best validation macro-F1.
The source-specific label mapping remains in feature preparation.

The scaler fits eligible training features only. It is reused for validation.
Validation observations choose the checkpoint but do not update model weights or scaler statistics.
Final test observations remain reserved.
The epoch-start convention remains an explicit preparation assumption.

## Configuration and execution

The first recipe is `configs/gru.yaml`: 20 maximum training epochs, batch size 64, learning rate 0.001, and patience five.
The optimizer is Adam. Gradient norms are clipped at one.
Seed 42 controls initialization and training-loader shuffling.
Seed control does not guarantee identical floating-point results across hardware or PyTorch versions.
Record the installed versions and resolved device with each run.

```bash
uv sync --locked --group gru --group notebook
uv run --locked --group gru --group notebook python -m sleep_classifier.gru_training --config configs/gru.yaml --check
```

Preflight loads prepared development data, fits the training-only scaler, and runs a no-gradient forward pass.
It does not create an optimizer, checkpoint, or validation accuracy report.
The measured local preflight confirms 19,928 training and 2,780 validation examples, four output logits, and 3,972 parameters.
The initial run used CPU. Auto now selects CUDA, then Apple MPS, then CPU.
Explicit unavailable devices raise an error. MPS passed synthetic forward, backward, padding, and Adam checks with PyTorch 2.15.0.
CUDA execution remains unverified. Small models may run faster on CPU; measure each device before claiming acceleration.

Timo can start training explicitly:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.gru_training --config configs/gru.yaml --train
```

Use notebook 07 for checked setup, preparation, preflight, and operator-controlled training commands.
Use the local checkout or set SLEEP_CLASSIFIER_ROOT in another notebook environment.
Existing checkouts are not reset or switched automatically.

## Artifacts and reporting

Training will write `artifacts/gru-context-10/metrics.json`, `gru_checkpoint.pt`, and `validation_predictions.csv`.
The checkpoint includes CPU model weights, architecture settings, scaler mean/scale lists, feature names, and context length.
The saved metrics include epoch history, best epoch, resolved device, package versions, coverage, and per-class and per-participant results.
Checkpoint selection uses validation macro-F1 and restores the selected state before final validation export.
Training logs show epoch number, losses, validation accuracy, and macro-F1.
A new run overwrites the configured directory; choose another output directory when preserving comparisons.

Notebook 07 plots loss curves, validation scores, and a normalized confusion matrix when saved results exist.
Report generation skips setup, preparation, and training cells. It can run preflight and display saved results.
Source notebooks remain output-free; executed reports are HTML only.
Publish selected measured figures under docs/experiments after the operator-run experiment.
The first run reached 56.69% validation accuracy and 0.4760 macro-F1.
See [controlled follow-up experiments](gru-followup.md) before choosing another configuration.
