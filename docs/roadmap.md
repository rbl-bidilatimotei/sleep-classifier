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

Prepared: instrumented seed-42 dropout-0.2 repeat for original folds 01 and 04 only.
Keep the measured recipe fixed. Record per-epoch target probabilities, class/participant NLL, confidence, actual learning rates, and gradients.
Save intermediate checkpoints at completed epochs thirteen through seventeen.
The budget is two fits and eighty epochs. Timo runs training locally.
The [diagnostic protocol](transformer-diagnostics.md) and [notebook 20](../notebooks/20_transformer_diagnostics.ipynb) prepare this run.
The reusable evidence store and inspector form the first local evaluation layer.
A later interface can add run/participant/stage selection and prediction timelines after these records prove useful.
The [replication review](experiments/transformer-dropout-seeds-2026-11-11/README.md) motivates this work.
A causal patch encoder remains a later architecture option after the targeted diagnostic.

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
