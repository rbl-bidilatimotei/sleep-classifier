# Twenty-epoch HR-reference preparation

Status: preparation only, 2026-11-18. No candidate classification scores exist yet.

- All 1,022 tests, Ruff, and whitespace checks pass
- Notebook 34 exports from a fresh kernel with training disabled
- Five MPS checks pass: raw input 64 × 30 × 6, effective input 64 × 30 × 8, logits 64 × 4
- Both models retain 2,852 parameters, the same initial trainable tensors, and both active HR channels
- All 130 copied feature tables match the ten-epoch reference byte-for-byte
- Original HR scales and available fitting-target counts match for all corresponding folds
- Deviation means and scales are fitted on training participants for the twenty-epoch definition and may differ intentionally
- Synthetic CPU/MPS checks confirm unchanged availability, original features, and causal prefix behavior with finite logits
- Unit tests cover normalization fitting, padding/future invariance, longer-horizon checkpoint restoration, and paired-policy confound rejection
- Both the standalone readiness guard and actual notebook command-binding tests pass without starting optimization
- No new participant-data model optimization or final-test observation reads occurred

[Fold preflights](no-gradient-checks.json) record each actual fitted state beside its ten-epoch reference state.
[Synthetic device checks](synthetic-device-checks.json) use generated observations and zero optimizer steps.
These checks validate preparation and execution contracts. They do not measure classification performance.
Follow the [fixed protocol](../../hr-history-horizon.md) for local training and interpretation.
