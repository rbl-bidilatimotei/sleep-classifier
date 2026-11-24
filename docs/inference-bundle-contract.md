# Versioned inference-bundle contract

Status: reviewed contract 1.0.0, 2026-11-23. Exporter and loader implementation follows on 2026-11-24.
See the [operating guide](evaluation-bundles.md) for publication and self-contained loading. No service is added.
Read the [model card](model-card.md) for measured performance and limitations.

## Purpose and scope

Package one evaluation checkpoint with its feature semantics, provenance, model card, and reproducible runtime identity.
Version 1 covers the current seven-input motion and heart-rate research profile on the validated MPS backend.
It supports evaluation-fold artifacts only. A deployment-model role needs a separately approved fitting and release procedure.
Do not select the highest-scoring fold as a deployment model. Do not treat the five folds as an implicit ensemble.

The bundle consumes prepared, completed-epoch summaries. Raw heart-rate readings, x/y/z events, and steps are outside this interface.
Labels are not input fields. A feature adapter must establish coverage, timestamps, units, and missing-data behavior before inference.
The [chronological audit](experiments/chronological-inference-2026-11-23/README.md) validates this prepared-feature boundary.
It does not validate a raw-event adapter or a different inference backend.

## Bundle directory layout

```text
<bundle_id>/
  manifest.json
  checkpoint.pt
  MODEL_CARD.md
  uv.lock
```

The manifest follows [manifest.schema.json](../schemas/inference-bundle/v1/manifest.schema.json).
Every listed file has a fixed relative name and a lowercase SHA-256 digest.
The manifest itself has no self-hash. A release registry can record its digest separately.
Do not include raw data, participant tables, cached motion audits, credentials, or absolute local paths.
The provenance identifies the source fold and recorded result hashes without embedding participant observations.

The exporter must create a sanitized inference checkpoint from the original training checkpoint.
Training checkpoints include training_config with absolute local paths. Do not copy those bytes unchanged into a bundle.
Allow only state_dict, model_type, model_config, features, model_features, context_epochs, scaler_mean, scaler_scale, window_feature_state, and motion_features.
Reject any retained field that contains a local path. Omit optimizer state, training_config, and training-data references.
Keep all retained tensors and fitted values identical. Serialization can change the checkpoint digest.
Record the original digest as provenance.source_checkpoint_sha256 and the packaged digest as files.checkpoint.sha256.
Verify the original digest against the recorded chronological audit before export.
The checkpoint contains weights, constructor metadata, feature order, context length, and fitted preprocessing state.
It must retain scaler_mean, scaler_scale, window_feature_state, and motion_features metadata.
It must also retain model_type, model_config, features, model_features, context_epochs, and state_dict.
The loader must compare those fields with the manifest and the selected feature profile.
The constructor must match width 16, two heads, one layer, feedforward width 32, P3 patches, and dropout 0.2.
The stored checkpoint remains authoritative for numerical fitted values. Never refit transforms while exporting or loading.
A raw-sensor adapter can use the stored motion imputation value without a local training-data cache.

## Versions and identity

| Field | Meaning |
| --- | --- |
| schema_version | Manifest representation version; the first supported value is 1.0.0 |
| feature_contract_version | Prepared-feature semantics; the first supported value is 1.0.0 |
| bundle_id | Unique identifier for one immutable assembled artifact |
| artifact_role | evaluation_fold; no production readiness is implied |
| runtime.source_revision | Exact runtime source commit, separate from the training provenance |
| files.checkpoint.sha256 | Identity of the packaged inference checkpoint bytes |
| provenance | Fold, training seed, source manifest hash, result hash, and final-test exclusion |

Keep an exported bundle immutable. Any bundle-content change requires a new bundle_id, including manifest, runtime, provenance, weights, fitted values, documentation, or lock changes.
Input units, aggregation, missingness, label mapping, and reset semantics define feature compatibility.
Change the feature-contract major version when those meanings change incompatibly.
Change the manifest major version when required field meanings change incompatibly.
Additive optional metadata can use a minor version after the loader explicitly supports it.
Version 1 readers reject unknown versions and unknown fields; no forward compatibility is assumed.

The manifest records runtime Python requirements and project version. It also records the exact source revision and validated backend.
Use the packaged lock with that source revision to reconstruct the runtime.
A project version alone does not identify source bytes. A digest proves byte identity, not publisher authenticity.
Trust and distribution controls remain a separate release decision.

## Input feature profile 1.0.0

Supply one [epoch request](../schemas/inference-bundle/v1/epoch-request.schema.json) per completed epoch.
The request contains participant_id, segment_id, time, and features. The feature object requires exactly these seven names.
JSON object order is irrelevant. The manifest's raw_features array defines the order used during normalization and model construction.

| Feature | Definition | Unit |
| --- | --- | --- |
| heart_rate_mean | Mean readings within the epoch | beats/minute |
| heart_rate_std | Population standard deviation, ddof=0 | beats/minute |
| heart_rate_min | Minimum reading | beats/minute |
| heart_rate_max | Maximum reading | beats/minute |
| motion_magnitude_mean | Mean Euclidean magnitude of the same-epoch x/y/z vectors | source g |
| motion_magnitude_std | Population standard deviation of those magnitudes, ddof=0 | source g |
| motion_grid_change_mean | Mean supported neighboring one-second grid vector-change magnitude | source g |

Use the source g representation, including gravity, recorded in [preparation](session-04.md#prepared-representation).
Do not silently substitute m/s², displacement, or jerk. An adapter conversion requires separate validation.
Current evidence does not establish a calibrated cross-device acceleration conversion.
For fixed-lag motion, use ticks at seconds 1 through 29 of the epoch.
At each tick, select the latest same-epoch observation with age between zero and 0.2 seconds, including the implemented 1e-8 tolerance.
Use only neighboring tick pairs with valid observations and strictly increasing source timestamps. One supported pair is sufficient.
Never interpolate or use future observations.
See the [implemented motion policy](fixed-lag-motion-ablation.md#missingness-and-scaling) and [source algorithm](../src/sleep_classifier/motion_grid.py).

Data preparation assumes time is the start of [time, time + 30 seconds).
The earliest prediction availability is time + 30 seconds, after all required summaries arrive.
Physical sensor timing and transport delay require separate validation. The interface does not forecast the next epoch.
Use finite seconds on a consistent session-relative clock. Negative session-relative values are permitted.
Do not mix epoch-start timestamps with epoch-end timestamps.

The caller supplies finite, already prepared values in original units. The predictor applies the saved scaler exactly once.
An unsupported fixed-lag motion value uses the checkpoint's saved training-target imputation mean before scaling.
A genuinely observed zero remains zero. Do not replace missing motion with zero.
Apply the frozen moderate sensor-support rule before requesting inference:

| Check | Required support |
| --- | --- |
| Deduplicated HR observations | At least three per epoch |
| Largest HR observation gap | At most fifteen seconds |
| Motion observations | At least one |
| Occupied one-second motion bins | At least 80% |
| Largest motion observation gap | At most five seconds |

Largest observation gaps include the distance from each epoch boundary to its nearest observation.
Remove exact duplicate sensor rows before aggregation. Reject conflicting readings at the same timestamp.
Do not average conflicts or silently choose one reading.
Offline scoring also requires a scored label. Live inference must check sensor support without requiring a ground-truth label.
A finite summary alone does not establish sufficient support.
If a core heart-rate or magnitude summary is unsupported, do not invent an imputation policy.
Skip prediction for that interval and break history before the next supported epoch.
The current predictor rejects missing keys, extra keys, null values, non-finite values, and values that overflow float32 after scaling.
JSON schemas cannot verify physiological support or availability times. Those require adapter-level checks.

The model adds two internal HR channels. Do not supply them in the request.
The deviation compares the current mean with at most ten strictly previous means inside the retained window.
At least four previous means are required. Otherwise, the derived deviation and availability flag are zero.
Available deviations use the checkpoint's fitted delta mean and scale.

## History and response

Use one predictor instance per active stream. Retain at most thirty completed epochs, including the current epoch.
Reset on participant changes, segment changes, or timestamp increments outside thirty seconds with absolute tolerance 1e-8.
A duplicate or backward timestamp within the active participant is an error, even when its segment changes.
Call reset() to begin an explicit new session at an earlier timestamp.
An invalid request must not advance history. Do not silently reuse another participant's state.
Reset-reason precedence is participant_change, then segment_change, then time_gap.
Same-participant ordering errors are checked before segment changes. Initialization and explicit reset return reset_reason null.
Recompute the full retained window after each input. Re-derive HR channels after eviction and right-align P3 patches with the current target.
Do not cache derived channels or patch boundaries from an earlier window.

The [response schema](../schemas/inference-bundle/v1/epoch-response.schema.json) matches StreamingEpochPredictor.predict_epoch.
It contains participant_id, time, prediction, probabilities, history_epochs, and reset_reason.
Probability indices map to wake 0, light 1, deep 2, and REM 3.
Light combines N1 and N2. Deep includes source N3 and legacy N4 when present.
The predicted class is the first argmax of the four probabilities, matching PyTorch behavior for ties.
The response timestamp identifies the classified epoch start. It does not record actual wall-clock completion time.
A separate service may add a completion timestamp under a separately versioned API contract.

Probabilities must be finite, lie between zero and one, and sum to one within absolute tolerance 1e-6.
JSON Schema checks field shapes and scalar bounds. A future loader or adapter must check normalization, argmax, and request/response identity.
Reject duplicate object keys and decoded non-finite values, including exponent overflow. Strict JSON encoding must reject NaN and Infinity.
JSON Schema uses mathematical integers. A future JSON adapter must normalize validated segment IDs to Python int and reject booleans.
The Python predictor already enforces finite inputs and finite model probabilities.

## Exporter and loader acceptance checks

1. Verify the bundle role and exact supported versions.
2. Reject missing files, changed hashes, unexpected entries, symlinks, and paths outside the bundle directory.
3. Verify the declared runtime revision, Python compatibility, and explicit backend.
4. Load only approved, sanitized checkpoint sources with weights_only enabled.
5. Match constructor, feature order, class count, context, and parameter count against the manifest.
6. Check finite scaler vectors, positive scales, and consistent HR normalization buffers.
7. Check stored motion definition and its fitted imputation metadata.
8. Compare training and validation provenance with the source fold record before export.
9. Require probability differences at most 2e-6 and zero class disagreements in startup, reset, eviction, and saved-prediction checks.
10. Preserve source artifacts and refuse to overwrite an existing bundle identifier.

Resolve packaged model-card links against runtime.source_revision. Verify each target exists and rewrite relative links to revision-pinned repository URLs.
Hash the rewritten MODEL_CARD.md bytes, not the original repository document.
These are implementation acceptance requirements. Structural schema validation alone does not satisfy them.
The [exporter and loader](evaluation-bundles.md) implement these checks. Legacy checkpoint-path inference remains available.

## Examples and checks

The [example manifest](../examples/inference-bundle/manifest.example.json) illustrates one reference fold's packaging metadata.
It is not an assembled bundle. The checkpoint, model card, and lock are not copied into the example directory.
Fold 01 illustrates the contract without ranking or selecting a deployment model.
Packaged-file hashes are explicit zero placeholders until an exporter assembles the files. They do not identify exported artifacts.
The original checkpoint digest remains recorded separately as verified provenance.
The example request and response are synthetic and are not measured model outputs.
No patient observations or model weights are committed with these examples.

Install the locked test dependencies and validate the contracts:

```bash
uv sync --locked --group dev --inexact
uv run --locked --group dev --inexact pytest -q tests/test_inference_bundle_contract.py
```

The tests verify the schemas, examples, rejected incompatible records, and agreement with current runtime feature names and recipe settings.
They do not open the final-test data, load research checkpoints, or train a model.

## Rollback

Retain previously validated immutable bundles and their recorded runtime revisions.
Roll back by selecting a prior validated bundle_id together with its runtime revision. Do not replace files inside the current bundle.
Reset active stream state when switching bundles. Only replay original supported summaries when they satisfy the restored feature contract and support rules.
Previously imputed motion can contain the old checkpoint's mean, even in unscaled history.
Recreate missing fixed-lag values with the restored checkpoint's mean before scaling.
Reset when the original missingness provenance is unavailable.
A changed scaler or model can make prior normalized history incompatible. Never carry normalized history silently across versions.
Record the selected bundle identity alongside each inference session.

## Next implementation boundary

Verify the evaluation-bundle implementation before extending its input boundary.
Next, audit the raw-sensor-to-epoch adapter on development data.
Define a separate deployment-model fitting protocol before any final-test decision. Training requires Timo's explicit authorization.
Docker, FastAPI, and economical CI remain later scoped milestones.

The [label-free preparation audit](label-free-sensor-audit.md) verifies raw summary formulas and exposes label-dependent research reset boundaries.
It does not extend the version-one prepared-epoch request schema or change the recorded models.
A later sensor adapter must define clock, coverage, end-of-session, and reset semantics without reading sleep labels.
Measure that policy's prediction impact with frozen development checkpoints before claiming end-to-end equivalence.
