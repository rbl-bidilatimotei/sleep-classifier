# Availability-only control preparation

Status: preparation only, 2026-11-18. No control scores exist yet.

- All 876 tests pass; Ruff and whitespace checks pass
- Notebook 31 exports from a fresh kernel with participant training disabled
- Five MPS no-gradient checks pass: raw input 64 × 30 × 6, effective input 64 × 30 × 8, logits 64 × 4
- The reference and candidate both retain 2,852 nominal parameters
- All 130 copied feature tables match the full-HR reference byte-for-byte
- All five fitted normalization states exactly match their full-HR reference states
- Synthetic CPU/MPS checks confirm identical initial state tensors, zero deviation, unchanged availability, and finite outputs
- Unit tests cover gradients, future/padding invariance, legacy behavior, masked checkpoint restoration, and paired-policy confound rejection
- No participant-data model optimization or final-test observation reads occurred

[Fold preflights](no-gradient-checks.json) record actual fitted training-only state, mask definitions, and shapes.
[Synthetic checks](synthetic-device-checks.json) use generated tensors with zero optimizer steps.
The preflights fit preprocessing and perform forward passes. They do not measure classification performance.
Follow the [fixed protocol](../../hr-availability-control.md) for local training and interpretation.
