# Fixed-lag motion candidate passes preflight, without training

Five folds are prepared from the existing development data. The final test remains reserved.
Notebook 36 ran from a fresh kernel with preparation and training disabled.
Its five no-gradient checks validate 64 × 30 × 7 raw batches, 64 × 30 × 9 internal inputs, and four output logits.
Each model has 2,900 parameters, exactly 48 more than the reference.
The resolved device is MPS, matching the measured reference.

Every fold preserves its original training/validation target counts and reports zero missing motion values.
The six original scaler means and the fitted HR state match the corresponding reference checkpoints exactly.
Synthetic tests also check original scaler scales and saved-state replay without refitting.
[checks.json](checks.json) records the five checks and feature provenance.
No candidate metrics.json or selected model checkpoint exists yet. No accuracy claim can be made.

Validation: 1,157 tests pass, Ruff and whitespace checks pass, and fresh-kernel notebook export passes.
The notebook's training-cell regression binds the actual command to the validated experiment configuration.
Source and derived hashes prevent training against changed cached sensor evidence.

Next: Timo runs the five candidate fits through notebook 36 with RUN_TRAINING enabled.
Keep RUN_PREPARATION false in this checkout because the folds already exist.
Review the complete paired result under the explicit motion_features policy.
