# Ten MPS seed-replication checks pass without training

The two new motion candidates at seeds 43 and 44 each have five prepared folds.
All reuse checks pass for the three measured references and the seed-42 MPS motion candidate.
The study plan lists ten new fits, capped at 400 epochs, with no new reference fit.

All ten no-gradient preflights passed on MPS:

- Raw batch: 64 × 30 × 7
- Effective inputs: 64 × 30 × 9
- Output logits: 64 × 4
- Parameters: 2,900 in every candidate
- All seven scaler means match the corresponding seed-42 motion fold
- Fitted HR state matches the same-seed reference and seed-42 motion fold
- Motion source and imputation metadata match the seed-42 motion fold

[checks.json](checks.json) records each seed/fold check and target count.
No new candidate result directory, trained model, or grouped score summary exists.
Hash verification confirms all 936 files from the completed CPU and seed-42 MPS motion studies remain unchanged.
No raw sensor observation reread, participant-data optimization, or final-test observation read occurred.

The full locked suite passes 1,179 tests. Ruff and whitespace checks pass.
Notebook 38 exports from a fresh kernel with all execution switches false.
It displays the completed seed-42 pair while both new candidates and the three-seed aggregate remain pending.

Next: Timo enables RUN_TRAINING in notebook 38 for the two new candidate runs.
Keep RUN_PREPARATION false in this checkout because the ten folds are already prepared.
The seed report uses strict device matching and rejects source changes across folds or seeds.
