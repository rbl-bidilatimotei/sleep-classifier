# MPS repeat passes preflight without training

The five seed-42 MPS candidate folds are prepared in a fresh output directory.
The prepared manifest records MPS, and all participant assignments match the reused MPS reference.
No candidate result directory, trained checkpoint, or accuracy claim exists yet.

All five no-gradient checks passed on MPS:

- Raw batch shape: 64 × 30 × 7
- Effective model inputs: 64 × 30 × 9
- Output logits: four classes
- Parameters: 2,900, unchanged from the CPU motion candidate
- All seven scaler means match the corresponding CPU candidate checkpoint
- The fitted HR state matches both existing recipes exactly

[checks.json](checks.json) preserves target counts, source hashes, and the five preflight records.
Hash verification confirms all 466 files in the completed CPU study remain unchanged.
No raw sensor reread, participant-data optimization, or final-test observation read occurred.

Fifteen focused notebook/configuration/command-identity tests pass, including a blocked prepared-device mismatch.
Ruff and whitespace checks pass. Notebook 37 exports from a fresh kernel with all execution switches false.
Its candidate rows remain pending, and the measured reference remains available for inspection.

Next: Timo enables RUN_TRAINING in notebook 37 and executes the five new fits locally.
Keep RUN_PREPARATION false in this checkout. The paired report permits no device exception.
