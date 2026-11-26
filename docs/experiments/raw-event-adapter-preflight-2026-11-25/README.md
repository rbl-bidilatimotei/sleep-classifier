# Bounded raw-event adapter preflight

Date: 2026-11-25. Scope: development participant 1066528, first thirty completed epochs, origin zero and declared end 900 seconds.
This is a bounded preflight. The twenty-six-participant audit and all-five-fold raw replay remain unrun.

The [report](report.json) pins raw hashes, source identities, chunk diagnostics, and frozen model evidence.
Both chunk sizes preserve all base features, coverage values, fixed-lag support diagnostics, eligibility, segments, and explicit completion rules.
Maximum numerical difference from the batch reference is zero. All thirty epochs are eligible.

| Check | Chunk 128 | Chunk 1024 |
| --- | ---: | ---: |
| Completed epochs | 30 | 30 |
| Peak pending events | 1,633 | 2,521 |
| Queue limit | 8,192 | 8,192 |
| Feature mismatches | 0 | 0 |
| Accepted event conservation | Passed | Passed |

Canonical adapter inputs reproduced all thirty saved sensor-only MPS predictions exactly.
There were zero class disagreements, and history lengths and reset reasons matched.
Weights and fitted preprocessing stayed unchanged. Evaluation mode remained enabled, and no gradients were created.
No training or final-test observation access occurred.

The default full-audit output directory remains unused. Notebook 46 can run the full audit without an existing-results conflict.
The bound concerns adapter queues, not process RSS. The recorded-data audit loads full tables and retains outputs outside the adapter.
