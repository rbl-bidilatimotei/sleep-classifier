# Notebook 43: operator export and loading

Date: 2026-11-24. Runtime: `1c223c2687308ed2c818d2a4ecdc37bd7b0eae91`. Backend: MPS.

## Observed result

Export and loading passed for `motion-seed42-fold01-evaluation-v1`.
This explicitly selected evaluation fold is not a deployment-model selection.

| Check | Result |
| --- | --- |
| Validation epochs / participants | 4,587 / 6 |
| Maximum probability difference | 7.450580596923828e-7 |
| Absolute tolerance | 2e-6 |
| Predicted-class disagreements | 0 |
| History lengths and reset reasons | All match |
| History starts / full-context epochs | 29 / 4,101 |
| Weights unchanged / gradients created | Yes / no |
| Training performed / final-test observations read | No / no |

[Export verification](export-verification.json) records the measured parity and source hashes.
The bundle contains four files: manifest, sanitized checkpoint, model card, and environment lock.
The checkpoint retains fitted preprocessing and inference weights. It removes training-only fields and local paths.
The original checkpoint remains unchanged. The exported model has 2,900 parameters and thirty-epoch context.
Its seven raw summaries produce nine effective inputs through the saved heart-rate transformation.

[Notebook inference](notebook-inference.json) records one synthetic request and model integrity.
That request exercises the interface; its predicted REM class is not an accuracy result.

## Self-contained runtime check

[Wheel verification](wheel-verification.json) records a separate-process check of the actual exported bundle.
All thirty-three synthetic responses match exactly, including history eviction beyond thirty epochs.
The process loads the matching wheel and copied bundle with access to original data, artifacts, source modules, and Git blocked.
Weights remain unchanged, evaluation mode stays enabled, and no gradients are created.
The bundle still needs its compatible Python environment, runtime dependencies, and validated MPS backend.
This is inference portability evidence, not a Linux or CPU validation.

The matching wheel is preserved locally under `artifacts/evaluation-runtime-wheel/1c223c26/`.
The manifest SHA-256 is `b0700cb28736032acc347dfafc5eec68aab628573bb4cf6f80e4c4b4c0a96d89`.
The wheel and bundle binaries remain ignored. Committed JSON records preserve their identities and measured results.

## Report inspection after later commits

A later commit changes the source revision. The existing bundle remains pinned to its measured revision.
Notebook inspection now checks schema, layout, lock metadata, and file hashes without deserializing weights.
It can report historical evidence from a newer checkout. It does not validate checkpoint semantics or runtime compatibility.
Actual model loading, inference, and export reuse retain the exact-runtime gate.
Use the preserved matching wheel or clean matching checkout for this bundle's inference.
The executed operator report is preserved locally under `artifacts/recovery/evaluation-bundle-2026-11-24/`.
The refreshed report-only HTML does not rerun export or model inference.

## Recommended next milestone

Audit a label-free raw-sensor-to-epoch adapter without training. Keep the working model and final test unchanged.
Start with synthetic timestamp, duplicate, coverage, missingness, and reset tests. Then compare development features and predictions.
Derive completed thirty-second intervals from the session clock, using only current or earlier sensor observations.
Restore saved imputation and scaling. Retain bounded heart-rate and motion history.

The offline preparation eligibility rule also checks `target >= 0`.
A live adapter cannot use ground-truth scored-label availability to choose epochs or reset history.
Measure sensor-only eligibility and isolate differences at unscored-label gaps before claiming end-to-end parity.
Resolve that contract before adding a service, Docker, or another architecture sweep.
