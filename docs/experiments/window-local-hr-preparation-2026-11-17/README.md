# Window-local heart-rate preparation evidence

Status: preparation only, 2026-11-17. No candidate model scores exist.

- All 791 unit and integration tests pass
- Ruff and whitespace checks pass
- Notebook 29 exports from a fresh kernel with training disabled
- All five development-fold forward checks pass on MPS
- Raw input: 64 × 30 × 6; effective channels: 64 × 30 × 8; logits: 64 × 4
- Candidate parameters: 2,852, compared with the reference 2,756
- All 130 copied feature tables match the reference byte-for-byte
- Synthetic CPU and MPS checks confirm future-prefix invariance, padding invariance, and unavailable-value masking
- No participant-data model optimization or final-test observation reads occurred

[No-gradient fold checks](no-gradient-checks.json) record the actual fitted training-only normalization and shape checks.
[Synthetic causal checks](synthetic-causal-checks.json) use generated tensors, without participant observations or optimizer steps.
The forward checks fit preprocessing on training participants. They do not measure classification performance.

Follow the [fixed experiment protocol](../../window-local-heart-rate.md) for local training and interpretation.
