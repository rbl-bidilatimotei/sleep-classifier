# HR feature seed replication preparation

Status: preparation only, 2026-11-17. New candidate seeds 43 and 44 have no measured scores.

- All three original-P3 reference studies and the seed-42 HR candidate pass saved-artifact validation
- Only candidates 43 and 44 need training: ten fits, at most 400 epochs in total
- All 260 copied feature CSVs match their measured reference tables byte-for-byte
- All ten no-gradient MPS checks pass: raw input 64 × 30 × 6, effective input 64 × 30 × 8, logits 64 × 4
- Every candidate has 2,852 parameters and training-only deviation normalization
- All 799 tests pass; Ruff and whitespace checks pass
- Notebook 30 exports from a fresh kernel with training disabled and five accessible figures
- The report shows only the complete seed-42 pair; the full three-seed aggregate remains unavailable
- No new participant-data model optimization or final-test observation reads occurred

[No-gradient fold checks](no-gradient-checks.json) preserve normalization and shape evidence for each new fold.
These checks fit training-only preprocessing and perform forward passes. They do not measure classification performance.
Follow the [fixed protocol](../../window-local-hr-seeds.md) for local training and interpretation.
