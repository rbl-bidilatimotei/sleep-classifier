# Eighty-epoch motion budget: preflight only

Prepared five fresh development folds on 2026-11-23. No participant-data optimization occurred.
The three final-test participants remain reserved. No final-test observations were read.

All five no-gradient MPS forward checks passed: seven raw inputs, nine effective channels, four logits, and 2,900 parameters.
The model-input schema, HR transform state, and motion provenance match the intact LR0.001 MPS working reference.
These checks validate construction and preprocessing. They do not estimate model quality or establish numerical training replication.

See [preflight.json](preflight.json) for fold counts, shapes, preprocessing, and prepared-manifest identity.
The planned cap is eighty epochs per fold. Patience remains twelve. The constant rate remains 0.0003.
Save every prefix-eligible checkpoint through epoch forty and predictions after every completed epoch.

The completed forty-epoch LR0.0003 weights and prediction rows were unavailable during inspection.
Its archived findings and historical HTML remain preserved. The missing-file cause is unknown.
The new comparison uses a shorter prefix from each new trajectory, not a restored historical run.

Notebook 40 and the modified notebook 39 passed fresh-kernel report-only execution.
Notebook 39's new pending report was exported separately under artifacts/notebooks/recovery-2026-11-23.
Its original completed HTML was verified unchanged by SHA-256.
The old output directory remained empty. The new output contains preparation files only.
No complete comparison JSON was written while training remained pending.
