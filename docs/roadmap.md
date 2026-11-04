# Project roadmap

Updated on 2026-11-04. Each milestone uses a separate reviewed work chunk.
Training remains operator-controlled. Keep final test participants reserved until model choices are fixed.

## Completed foundation

- Audit the cohort and freeze participant splits
- Prepare causal features with explicit coverage rules
- Measure majority, logistic-regression, and class-weighted references
- Train the first GRU and compare five controlled settings on MPS
- Verify OPAL's offline keep wrapper with synthetic inputs

The [measured GRU comparison](experiments/gru-followup-2026-11-04/README.md) is the current experiment reference.
The real-data trials used Adam. Live reactive optimization remains untested.

## Next experiment milestone

Compare the reference and patience-twelve recipes with three shared seeds.
Report best-checkpoint macro-F1, class scores, participant scores, and variation across seeds.
Use fixed features and participants. Record all runs, including weaker results.
Prepare the scripts, YAML settings, and notebook for Timo to run.

Prepare a separate Codex-powered OPAL pilot against an identical Adam recipe.
Reuse OPAL's experimental Codex backend rather than maintain a fork.
Agree on the account and token budget before launching live agent calls.
Preserve generated programs and decision records with the experiment provenance.
See the [OPAL assessment](reactive-optimizer-assessment.md) for integration and execution requirements.

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
