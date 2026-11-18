# Deviation-only seed replication preparation

Status: preparation only, 2026-11-18. Candidate seeds 43 and 44 have no measured scores.

- All four reused grouped studies pass metric-hash, prediction-score, and recipe validation
- New work is exactly ten candidate fits, capped at 400 epochs in total
- All 260 copied feature CSVs match their corresponding full-HR reference tables byte-for-byte
- All ten MPS no-gradient checks pass with six raw channels, eight internal channels, four logits, and 2,852 parameters
- Every fitted normalization state exactly matches its corresponding full-HR reference fold
- All 937 tests, Ruff, and whitespace checks pass
- Notebook 33 exports from a fresh kernel with training disabled and accessible figures
- The report shows the completed seed-42 pair; the three-seed aggregate remains pending
- Runtime tests verify each resolved CLI target and stop blocked runs before any command is launched
- No new participant-data model optimization or final-test observation reads occurred

[No-gradient fold checks](no-gradient-checks.json) record actual fitted state, mask, device, and shape evidence.
These checks fit training-only preprocessing and perform forward passes. They do not measure classification performance.
Follow the [fixed protocol](../../hr-deviation-only-seeds.md) for local training and interpretation.
