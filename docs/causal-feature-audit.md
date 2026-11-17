# Causal feature audit and local evaluation view

Status: implemented and executed without model training. Snapshot created on 2026-11-16; reviewed on 2026-11-17.
See the [measured findings](experiments/causal-feature-audit-2026-11-17/README.md).

## Purpose and boundary

The recent regularization experiments do not justify replacing the original unsmoothed Adam/P3 reference.
Inspect persistent Deep/REM-to-Light errors before another architecture, optimizer, or loss change.
Build reusable evaluation components while answering that specific question.

This audit reads saved development predictions and checksum-verified development motion files.
It never constructs a model, updates model parameters, or changes the training feature schema.
The three final-test participants are excluded before observation access.
Their identifiers are read as metadata solely to enforce that boundary.
Existing eligibility, participant splits, scalers, predictions, and raw files stay unchanged.

The audit aligns the original reference with patch size two, AdamW, and label smoothing at seed 42.
A persistent error means true Deep or REM with Light predicted by every included recipe.
These related models share participants and training data. Agreement is not independent replication.

## Components and artifact contract

```text
Frozen grouped artifacts → AuditEvidenceLoader
Development motion files → VerifiedMotionSource
                       ↓
              CausalAuditFeatures
                       ↓
       rows.csv + summary.json + manifest.json
                       ↓
      AuditVisualization → local dashboard + figures
                       ↓
            notebook 28 → executed HTML
```

| Component | Responsibility |
| --- | --- |
| AuditEvidenceLoader | Validate plans, saved scores, features, probabilities, hashes, and one-to-one target alignment |
| VerifiedMotionSource | Enforce the development allowlist and verify publisher checksums before parsing sensor observations |
| CausalAuditFeatures | Compute strictly past heart-rate context and within-epoch motion differences |
| CausalFeatureAudit | Orchestrate the audit, summarize evidence, and publish a hashed snapshot |
| AuditVisualization | Render offline timelines, target tables, and descriptive plots |

The output manifest records configuration, implementation, and output hashes.
The summary records the source-artifact and raw-file hashes used during creation.
The cached reader verifies the saved outputs and configuration without rereading raw observations.
A cached report is a historical snapshot; it does not revalidate every original source file on each view.
Do not edit snapshot files in place. Use a new output directory for a new audit definition.

## Diagnostic definitions

### Strictly past heart-rate context

For each participant and contiguous eligible segment, use at most ten previous epoch heart-rate means.
Exclude the current epoch from the past mean. Require at least four past epochs.
Compute current epoch mean minus this strictly past mean.
At startup or after a segment reset, unavailable values remain missing.
Keep the past count and availability flag alongside the diagnostic value.
Do not drop these epochs or silently replace missing history with zero.

The current interval is complete at prediction time. Its own mean is therefore available to the subtraction.
No whole-night normalization, centered window, future observation, or previous true stage enters the feature.
This is a heart-rate trend diagnostic, not physiological HRV derived from beat-to-beat intervals.

### Within-epoch axis-vector change

Use adjacent accelerometer observations inside each completed interval [t, t + 30 seconds).
Compute the Euclidean norm of each axis-vector difference.
Exclude pairs with nonpositive time differences or gaps greater than 0.2 seconds.
A 1e-8-second numerical tolerance handles timestamp rounding at that threshold.
Never join pairs across epoch boundaries. Reuse the existing exact-duplicate/conflict policy.

Average the admissible pair differences. This is not a time derivative or jerk.
Keep valid pair counts, valid fractions, and observed median/maximum gaps.
No admissible pairs means an unavailable mean, while observed zero movement remains zero.

Pair-difference amplitudes depend on sampling cadence and sensor behavior.
Inspect the gap evidence before comparing participants or proposing this as a model input.
This audit does not normalize cadence or claim that a new motion feature improves sleep classification.

## Read and run

Open [notebook 28](../notebooks/28_causal_feature_audit.ipynb).
Read each What/Why note before its code cell.
Keep RUN_SETUP and RUN_AUDIT false on the current checkout.
The completed cached snapshot already exists locally.
Run the remaining cells to inspect tables and figures without reading raw observations again.

For a fresh checkout, supply the existing development data and completed grouped-study artifacts first.
Inspect the metadata-only plan:

```bash
uv run --locked --group baseline --group notebook python -m sleep_classifier.causal_feature_audit \
  --config configs/causal-feature-audit.yaml --plan
```

Run the development-only raw audit once:

```bash
uv run --locked --group baseline --group notebook python -u -m sleep_classifier.causal_feature_audit \
  --config configs/causal-feature-audit.yaml --run
```

The command logs each permitted participant and rejects an existing output directory.
There is no training option in this CLI.
Do not remove model artifacts or reserved data to rerun it. Select a new audit output directory instead.

Export the report:

```bash
uv run --locked --group baseline --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/28_causal_feature_audit.ipynb
```

Fresh-kernel report execution skips setup, raw analysis, and recursive export.
Keep source notebooks output-free and store executed HTML only under ignored artifacts.

## Local viewer

The snapshot contains `artifacts/causal-feature-audit/dashboard.html`.
It embeds its data and scripts, requires no server, and makes no external requests.
Select a participant, prediction recipe, window length, and window start.
Stage/error filters affect the table only. Timelines retain their actual elapsed-time gaps.
Feature paths break at segment boundaries and unavailable values.

The target table includes selected predictions, raw predicted-stage/true-stage probabilities, heart-rate history, motion support, and sampling gaps.
These are raw model probabilities, not a claim of calibrated uncertainty.
With all recipes shown, the table uses the original reference prediction.
The notebook links the existing dashboard; it does not rewrite hashed snapshot files.

Static plots show one median per participant and stage rather than treating epochs as independent people.
Error-count plots are counts, not participant difficulty rankings or rates.
Training-fold summaries use each fold's training participants only.
Matched error summaries compare persistent errors with reference-correct targets within the same participant and true stage.
All label-conditioned summaries are post-hoc diagnostics, never model inputs.

## Platform scope and next work

This PR delivers a local artifact contract and an offline audit viewer.
It complements existing per-epoch evidence readers; it does not replace them or introduce a hosted service.
Later PRs can add a run catalog, selected-checkpoint/epoch navigation, sampling-aware feature views, and evidence bookmarks.
Keep compute and training behind explicit commands as the workspace grows.

Review the measured audit before selecting a feature ablation.
An apparent visual separation is not a performance result, and persistent errors do not prove an information ceiling.
A future ablation must keep the original participant folds and target population fixed.
Fit transforms on training participants only and preserve missing-history information.
The final test remains reserved.

## Verification limits

Tests cover causality, scope exclusion, checksum changes, snapshot tampering, and offline JavaScript control behavior.
The generated plots were inspected visually.
The agent browser blocked local-file URLs, so actual browser layout was not verified through that tool.
No alternate browser route was used. Offline render-function tests do not replace browser layout review.

Validation: 727 tests, Ruff lint, whitespace checks, and fresh-kernel notebook export pass. Both notebook figures have descriptive alternative text.
