# Notebook 46: full raw-event adapter audit

Date: 2026-11-26. Scope: twenty-six development participants, two chunk sizes, and five frozen held-out MPS checkpoints.
The [complete report](report.json) and [manifest](manifest.json) preserve the measured source and output identities.

## Acceptance result

The audit passed. Both chunkings matched all fresh batch fields, support diagnostics, eligibility decisions, and sensor segment IDs.
The saved notebook-44 sensor projection also matched exactly. There were zero numerical mismatches in either comparison.

| Check | Chunk 128 | Chunk 1024 |
| --- | ---: | ---: |
| Completed epochs | 24,738 | 24,738 |
| Sensor-eligible epochs | 23,674 | 23,674 |
| Input events within declared sessions | 36,133,288 | 36,133,288 |
| Exact duplicate events | 31,609 | 31,609 |
| Discarded tail events | 18,852 | 18,852 |
| Peak pending events | 2,131 | 3,026 |
| Declared queue cap | 8,192 | 8,192 |
| Feature mismatches | 0 | 0 |

Every session passed event conservation and closed with no pending events.
The tail counter measures events outside complete intervals at the declared endpoint. It is not a count of dropped eligible epochs.
The duplicate counter records ignored repeated observations, which do not alter features or consume additional queue capacity.

Chunk 1024 used a higher peak queue, as expected from less frequent watermark/drain calls.
Its measured chunk replay total was 167.5 seconds, versus 188.3 seconds for chunk 128.
These descriptive totals exclude parts of the full audit, including source reading and model replay. They are not a live latency benchmark.
The adapter's queue cap does not bound whole-process memory: the offline audit holds full sensor tables and emitted outputs.

## Frozen-model parity

The canonical chunk size was 128. Other chunkings had identical feature and state inputs and did not repeat model inference.
All 23,674 canonical predictions matched notebook 45 exactly across all five held-out models.
Probability difference was zero, with zero class disagreements and matched history lengths and reset reasons.
Weights and saved transforms stayed unchanged. Evaluation mode stayed enabled, and no gradients were created.
No training or final-test observation reads occurred.

This adds an executable raw-event preparation boundary. It does not create new accuracy or F1 results.
Notebook 45's sensor-only scores remain the relevant measured model results.

## What the result establishes

Ordered, checksum-verified recorded events can reproduce the measured batch and sensor-only inference behavior under explicit completion rules.
Changing chunk size changes pending queue use and observed replay time, while preserving the emitted data and predictions.
The audit supports the configured origin-zero, zero-lateness profile and conservative session-end policy.
Live source delivery, clock synchronization, concurrent access, and late-event grace periods remain outside this evidence.

## Next milestone

Connect this adapter to a version-checked evaluation bundle in a single-session runner and JSONL CLI.
Emit predictions for eligible epochs, explicit skip records for unsupported epochs, and a final session summary.
Preserve saved imputation and scaling. Keep operational logs separate from protocol output.
Bound input line size and outputs per event, validate the versioned event protocol, and fail explicitly on invalid input.
This is a smaller integration step before a network service, Docker, or further model tuning.
