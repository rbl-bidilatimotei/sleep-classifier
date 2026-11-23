# Five lower-rate motion checks pass without training

The seed-42 LR0.0003 motion candidate has five prepared MPS folds in a fresh output directory.
Its reused LR0.001 motion reference remains unchanged.
All five no-gradient checks confirm:

- Learning rate configured as 0.0003
- Raw batch shape 64 × 30 × 7 and effective input shape 64 × 30 × 9
- Four output logits and 2,900 parameters
- Scaler means, fitted HR state, and complete motion provenance equal the saved reference

[checks.json](checks.json) records the fold checks and target counts.
No new candidate result directory or trained checkpoint exists.
Hash verification confirms all 3,177 files across seven prior completed studies remain unchanged.
No raw sensor reread, participant-data optimization, or final-test observation read occurred.

The full locked suite passes 1,200 tests. Ruff and whitespace checks pass.
Notebook 39 exports from a fresh kernel with all execution switches false and the candidate marked incomplete.
Synthetic report tests enforce identical motion inputs and metadata under the learning-rate policy.
They reject source, imputation, HR state, capacity, schema, device, and unrelated recipe changes.

Next: Timo enables RUN_TRAINING in notebook 39 for the five new fits.
Keep RUN_PREPARATION false in this checkout because the folds are already prepared.
Review the fixed forty-epoch budget and selected-checkpoint rule without changing them during the run.
