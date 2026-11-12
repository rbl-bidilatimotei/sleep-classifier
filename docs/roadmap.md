# Project roadmap

Updated on 2026-11-11. Each milestone uses a separate reviewed work chunk.
Training remains operator-controlled. Keep final test participants reserved until model choices are fixed.

## Completed foundation

- Audit the cohort and freeze participant splits
- Prepare causal features with explicit coverage rules
- Measure majority, logistic-regression, and class-weighted references
- Train the first GRU and compare five controlled settings on MPS
- Verify OPAL's offline keep wrapper with synthetic inputs
- Replicate the GRU patience comparison across three seeds
- Complete a live Codex-powered OPAL pilot with one applied optimizer change
- Complete five-fold grouped development validation for the fixed GRU recipe
- Complete the matched five-fold causal Transformer comparison
- Complete the width-32 capacity test; retain width 16 as the development reference
- Complete the width-16 budget test; measure REM gains with forty epochs and patience twelve
- Replicate paired budget gains across seeds 42, 43, and 44
- Complete the seed-42 context-thirty comparison; measure Wake/Deep gains and a REM trade-off
- Replicate context-thirty gains across three seeds and adopt it as the working development reference
- Complete seed-42 dropout 0.1 versus 0.2; measure a modest macro-F1 gain and lower selected-checkpoint NLL
- Replicate dropout 0.2 across three seeds; measure small mean macro-F1 and NLL gains with mixed class trade-offs

The [shared-seed results](experiments/gru-seeds-2026-11-04/README.md) and
[successful OPAL retry](experiments/opal-retry-2026-11-05/README.md) record the current findings.
OPAL remains optional. Conventional Adam is the next reference.

## Next experiment milestone

Completed: the two-fold diagnostic repeat reproduced original histories and selected checkpoints exactly.
Its fixed-weight training/validation curves and matched persistent errors support probability overfitting.
See the [measured diagnostic results](experiments/transformer-diagnostics-2026-11-12/README.md).

Completed: constant 0.0003 improves selected NLL but loses mean macro-F1 and REM/Deep recall on the two diagnostic folds.
Retain the measured constant-0.001 stage-classification reference.
See the [learning-rate results](experiments/transformer-learning-rate-2026-11-12/README.md).

Proposed next: start at 0.001 and reduce by factor 0.3 after validation macro-F1 plateaus.
Keep the same model, seed, folds, evidence controls, sampling, loss, clipping, and budget 40/12.
Maximum new work is two fits and eighty epochs. This is a targeted optimization comparison, not a global recipe decision.
If this bounded check provides little benefit, return to the planned small causal patch encoder.
The local evidence store supports later experiment browsing and prediction timelines.

## Model and evaluation milestones

- Check participant sensitivity with grouped development folds
- Compare causal feature additions and context lengths
- Compare a small causal Transformer after establishing a reproducible GRU reference
- Assess Mamba only after checking runtime support and the experiment's purpose
- Evaluate the reserved test set once model choices are fixed

## Later engineering milestones

These items are roadmap work, outside the next experiment milestone.

| Milestone | Scope | Readiness condition |
| --- | --- | --- |
| Local evaluation workspace | Browse runs, participants, stages, confidence, timelines, and checkpoint evidence through the existing reader | The first instrumented run validates the evidence schema and useful views |
| Portable inference | Load a versioned checkpoint, scaler, feature schema, and causal history | Model recipe and prediction contract are defined |
| FastAPI service | Validate timestamps, coverage, and session boundaries; return probabilities, labels, and model version | Offline and streaming predictions agree on the same completed epochs |
| Docker | Build a locked CPU training/inference environment; mount data and artifacts externally | Script and inference interfaces are stable |
| Economical CI | Run lint, unit tests, and synthetic inference checks; cache dependencies and avoid duplicate branch/PR jobs | Check commands and dependency groups are stable |
| Delivery pipeline | Build and publish versioned images; use a manual deployment gate | A deployment target and release policy exist |

Keep model training and dataset downloads outside routine CI.
Use scheduled or manual jobs for longer validation when needed.
Native macOS execution remains the MPS path; Docker does not provide Apple's MPS accelerator.
Keep the first API aligned with the selected model's actual inputs.
Do not accept steps as a meaningful feature until a trained recipe uses them.
Specify warm-up and missing-data behavior before exposing session predictions.

The shared-seed milestone is prepared in [notebook 09](../notebooks/09_gru_seed_comparison.ipynb).
See its [fixed protocol](gru-seed-comparison.md) before launching the six runs.

The [seed results](experiments/gru-seeds-2026-11-04/README.md) are recorded.
The next [Codex-powered OPAL pilot](opal-pilot.md) is prepared in notebook 10 and remains operator-controlled.

Grouped development validation is completed and reproducible through [notebook 11](../notebooks/11_grouped_validation.ipynb).
Its [protocol](grouped-validation.md) and the [learning review](learning-review.md) separate observed results from hypotheses.

The causal Transformer comparison is completed in [notebook 12](../notebooks/12_transformer_comparison.ipynb).
See its [fixed protocol](transformer-experiment.md) before training.
