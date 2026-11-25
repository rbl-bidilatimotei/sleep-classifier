# Label-free raw-sensor preparation audit

Status: development audit completed on 2026-11-24. No model was loaded, fitted, or modified.
Read the [measured findings](experiments/label-free-sensor-audit-2026-11-24/README.md) alongside notebook 44.

## Purpose

The research preparation uses label timestamps to define epochs. It also excludes epochs without scored targets before assigning model history segments.
A live caller has no sleep-stage labels. The input schedule and history resets must therefore use session timing and sensor coverage.

This milestone separates those concerns without changing the measured research pipeline or its fitted models.
`EpochFeatureBuilder.build()` retains its existing target mapping, output columns, coverage rules, and label-dependent eligibility.
The new `build_sensor_epochs()` accepts only explicit epoch times, heart rate, and acceleration. It rejects target or stage columns.
The audit compares this sensor-only view with saved research preparation after constructing the candidate rows.
The model and its feature definitions stay fixed. No imputation, scaling, or optimizer fitting occurs.

## Clock and sensor rules

The YAML declares the first session-relative epoch start: zero seconds for this dataset audit.
The clock does not read that value from sleep labels. A future caller must supply a synchronized session origin.
Intervals follow `[t, t + 30)`. A sample at the right edge belongs to the next interval.
The offline replay closes epochs only through the earlier final timestamp from the two sensor files.
This conservative rule excludes unfinished tails. It is a proxy for completed input, not an implementation of live arrival or lateness handling.
An explicit session-end event may support another tail policy in a later contract.

The shared implementation preserves these rules:

- Collapse exact duplicate timestamps. Reject conflicting values.
- Use only observations inside each completed interval for the six base summaries.
- Keep the selected coverage rule from frozen preparation provenance.
- Compute fixed-lag motion on the same interior one-second ticks, with observation age at most 0.2 seconds.
- Retain NaN for unavailable motion diagnostics. The audit does not fit or apply replacement values.
- Start a new sensor segment after a rejected epoch. Do not reset because a sleep label is absent.
- Cap the diagnostic history length at the model's existing thirty-epoch context.

Steps remain excluded, as in the working model. The model still computes its fitted heart-rate transformation during inference.
This audit compares raw channels and context identities; it does not generate the two derived heart-rate channels or predictions.
For contiguous segments, history length and current epoch identify the retained timestamp window.
Context-cause diagnostics inspect that window for covered unscored epochs and clock-only epochs. Cause counts can overlap.

## Scope and input protection

The source validates the frozen split against saved preparation metadata before any observation access.
Only the twenty-six development participants can supply raw heart-rate, motion, or prepared files.
The loader refuses raw labels, steps, reserved-test identifiers, unexpected prepared tables, checksum changes, and observation-path symlinks.
Raw sensors must match the recorded publisher digests in preparation provenance.
The historical motion snapshot must match its saved hashes, membership, and fixed-grid definition.
Labels in saved development preparation are used only for retrospective attribution after sensor-only rows exist.
Reserved identifiers in split metadata are permitted; their observations are not opened.

The YAML resolves paths relative to its own directory. It identifies existing preparation, raw sensors, the frozen split, and the historical motion snapshot.
It also declares the session origin and a fresh audit output path. It contains no training or model settings.
The default output is `artifacts/label-free-sensor-preparation-audit`.

## Run or inspect

Install the locked analysis environment:

```bash
uv sync --locked --inexact --group baseline --group notebook --group dev
```

Inspect the metadata-only plan:

```bash
uv run --locked --inexact --group baseline python -m sleep_classifier.sensor_preparation_audit \
  --config configs/label-free-sensor-audit.yaml
```

Run a fresh development audit:

```bash
uv run --locked --inexact --group baseline python -u -m sleep_classifier.sensor_preparation_audit \
  --config configs/label-free-sensor-audit.yaml --run
```

The command refuses an existing output. Choose a new configuration and output path for an intentional repeat.
Notebook 44 reuses an existing snapshot after checking its configuration and output hashes.
Keep RUN_AUDIT false to inspect the completed report. Set TIMELINE_PARTICIPANT to inspect one development session.
Every code cell explains its purpose. Setup, raw audit, and recursive HTML export cells carry skip-execution tags.

Export a fresh report without rerunning raw preparation:

```bash
uv run --locked --inexact --group baseline --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/44_label_free_sensor_audit.ipynb
```

The HTML appears at `artifacts/notebooks/44_label_free_sensor_audit.executed.html`.
Source notebooks stay output-free. Local snapshots and HTML remain ignored; selected JSON evidence and one plot are committed.
A new clone needs the documented raw and reference artifacts to run the audit.
The snapshot loader verifies stored outputs; it does not revalidate current raw inputs or claim a new audit execution.
The manifest records configuration, output, and implementation hashes. The summary records input hashes and explicit no-training/test-read flags.

## Interpretation and next boundary

`status=passed` means numerical features and sensor coverage agree on shared epochs.
It does not mean epoch sets, model contexts, classification scores, or clinical performance are equivalent.
Three previously eligible targets fall outside this audit's conservative tail boundary. Keep them visible when comparing later predictions.
Do not silently evaluate only an easier intersection or feed the old label-derived reset mask into a live stream.

The next proposed milestone is frozen-model replay with sensor-only eligibility and resets.
Restore each fold's saved imputation and scaling. Make predictions first, then join development targets for scoring.
Report changes on shared scored epochs and disclose additional unscored predictions and excluded boundary targets.
Use a matching bundle/runtime or a separately identified, verified export. Do not bypass the exact-runtime guard.
This replay can measure the consequences of removing label-dependent resets without any training or final-test access.

The [frozen-model replay](sensor-only-replay.md) is now prepared in notebook 45.
A one-fold inference preflight passed; the full five-fold policy comparison remains opt-in.
