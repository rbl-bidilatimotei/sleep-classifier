# Test a slower strictly past HR reference

Status: completed locally by Timo. See the [measured findings](experiments/hr-history-20-2026-11-19/README.md).

## Question

The [completed flag replication](experiments/hr-deviation-only-seeds-2026-11-18/README.md) gives no consistent benefit from availability removal.
Keep both HR channels and stop that ablation series.
The earlier combined recipe remains the working reference, with consistent Light/Deep and Wake/transition trade-offs.

Test one representation hypothesis: a slower recent HR reference may give a more useful relative-level signal.
Compare up to twenty previous epoch means, ten minutes, against the current ten previous means, five minutes.
This is a proposed experiment, not an established physiological claim or promised improvement.
Use one declared candidate horizon rather than an open-ended validation sweep.

## Information boundary and example

Both models receive the same thirty completed raw epochs and six prepared input channels.
The internal transform adds deviation and availability, giving eight channels before patch projection.
Every token uses only earlier observations inside its supplied window. Exclude the current epoch from its reference average.
Reset at each window boundary and existing participant/segment boundary.
Require at least four previous epochs in both models; the availability flag therefore stays identical.
Keep all startup targets eligible.

For ten means of 50 BPM, then ten means of 60 BPM, followed by a current mean of 70 BPM:

| Reference horizon | Past means used at the current position | Past mean | Raw deviation | Availability |
| --- | ---: | ---: | ---: | ---: |
| Ten epochs | Most recent ten | 60 BPM | +10 BPM | 1 |
| Twenty epochs | All twenty | 55 BPM | +15 BPM | 1 |

These example deviations are raw BPM differences. The model standardizes them using training-only statistics.
Do not precompute session-wide summaries and then slice thirty-epoch windows.
The oldest token could otherwise carry twenty extra epochs, expanding the effective context from thirty to fifty epochs.
The existing window-local transform prevents that confound.
The candidate changes the reference summary within existing observations; it adds no sensor data or raw context.

## Fixed controls and normalization

| Setting | Reused full-HR reference | Candidate |
| --- | --- | --- |
| Maximum previous HR means | 10 | 20 |
| Minimum previous means / availability | 4 / unmasked | Same |
| Raw context / raw channels / internal channels | 30 / 6 / 8 | Same |
| Parameters and trainable initialization layout | 2,852 | Same |
| Original six-feature scaler | Training-only fit | Same fitted values |
| Deviation fitting targets | One available training target per window | Same count and source |
| Deviation fitted mean and scale | Ten-epoch definition | May change with the new definition |

The two raw deviation definitions agree when at most ten previous epochs exist.
Their standardized values can still differ there because their training-fitted means and scales can differ.
Do not require complete normalization states to match across this experiment.
Require the same original HR scale, available fitting-target count, training-only origin, and valid state schema.
Checkpoints record and restore the declared horizon and its own fitted normalization without refitting during inference.

Trainable initial tensors remain matched; fitted normalization buffers may differ intentionally.
Keep P3, width 16, two heads, one layer, feedforward width 32, and dropout 0.2.
Keep original Adam 0.001, unsmoothed unweighted loss, natural sampling, batch 64, and norm clipping one.
Keep forty epochs, patience twelve, macro-F1 checkpoint selection, diagnostics, intermediate checkpoints, and MPS.
Keep training seed 42, split seed 42, all five development folds, and all eligible targets fixed.

Reuse the completed full-HR seed-42 reference. Train five fresh candidates, capped at 200 total epochs.
The horizon, its resulting deviation normalization, and output paths are the only intended changes.
The three final-test participants remain reserved.

## Run locally

Open [notebook 34](../notebooks/34_hr_history_horizon.ipynb).
Read each What/Why explanation. Inspect the reused reference, ten-versus-twenty definition, and five no-gradient checks.
Keep setup disabled in the current environment.
Keep preparation disabled after the candidate manifest exists.
Set RUN_TRAINING true when ready. Run the training cell interactively.
It uses the validated experiment's config path and readiness guard; it never trains the reference.

Source notebooks default to training disabled. Existing output directories block accidental reruns; automatic resume is unsupported.
Fresh clones require development features and artifacts/transformer-window-local-hr-grouped-validation.
Use MPS to match that measured reference.

Prepare only when the candidate output directory does not exist:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/hr-history-20-grouped.yaml --prepare
```

Check all prepared folds through the notebook. For one fold:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config artifacts/hr-history-20-grouped-validation/fold-01/gru.yaml --check
```

Train only through the gated cell or this explicit command:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/hr-history-20-grouped.yaml --train
```

## Evaluation and decision

Use the explicit window_history paired policy. It permits only the declared longer reference within unchanged raw context.
Require the same feature names, unmasked channels, model layout, parameter count, optimizer/loss controls, populations, and coverage.
Validate metric hashes, actual prediction scores, recorded constructors, devices, budgets, and normalization provenance.
The notebook also checks the planned settings against the saved reference recipe.

Primary: candidate-minus-reference mean fold macro-F1, with all fold and participant differences.
Inspect Wake, Light, Deep, and REM precision, recall, F1, support, confusion matrices, NLL, and comparable loss curves.
Check the existing Wake/transition trade-offs beside aggregate gains.
Use the same history bins: 1–4, 5–29, and full thirty-epoch context, plus the overlapping transition slice.
Do not add slice macro-F1 scores to reconstruct pooled or fold scores.

A useful seed-42 gain requires replication before changing the working reference.
A negative result supports retaining the ten-epoch definition. Do not select a favorable fold or lower-NLL checkpoint post-hoc.
A different normalization state alone is not a performance finding.
Repeated development selection remains adaptive exploration. Keep the final test untouched.

Export HTML without training:

```bash
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/34_hr_history_horizon.ipynb
```

Export skips setup, preparation, training, and recursive export. Keep source outputs clear and generated HTML under ignored artifacts/notebooks.

## Preparation evidence

The [preparation snapshot](experiments/hr-history-20-preparation-2026-11-18/README.md) records five MPS preflights and synthetic CPU/MPS checks.
All 1,022 tests, Ruff, whitespace checks, and fresh-kernel report export pass.
That preparation snapshot predates Timo’s completed run. Reuse the measured artifacts rather than retraining them.

## Measured outcome

Twenty past epochs lose mean fold macro-F1 by 0.00201; pooled macro-F1 and NLL are nearly unchanged.
Deep and REM F1 improve modestly, while Wake and Light F1 decline. Retain the ten-epoch full-HR reference.
Next, audit fixed-lag motion diagnostics and their sampling support before another feature-training experiment.
