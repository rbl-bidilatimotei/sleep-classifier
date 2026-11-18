# Deviation-only control preparation

Status: preparation only, 2026-11-18. No deviation-only scores exist yet.

- All 919 tests, Ruff, and whitespace checks pass
- Notebook 32 exports from a fresh kernel with training disabled
- Five MPS checks pass: raw input 64 × 30 × 6, effective input 64 × 30 × 8, logits 64 × 4
- The candidate and reference retain 2,852 nominal parameters and identical initial tensors
- All 130 copied feature tables match the full-HR reference byte-for-byte
- All five fitted normalization states exactly match their corresponding full-HR reference states
- Synthetic CPU/MPS checks confirm zero availability, unchanged deviation, unchanged original channels, and finite logits
- Paired-report tests cover both single masks, accurate metadata, and mixed-mask rejection
- No participant-data model optimization or final-test observation reads occurred

[Fold checks](no-gradient-checks.json) preserve actual mask, state, and shape evidence.
[Synthetic checks](synthetic-device-checks.json) use generated tensors with zero optimizer steps.
The preflights fit preprocessing and perform forward passes. They do not measure classification performance.
Follow the [fixed protocol](../../hr-deviation-only.md) for local training and interpretation.
