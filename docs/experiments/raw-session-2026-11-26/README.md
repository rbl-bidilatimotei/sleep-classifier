# Notebook 47: operator export and synthetic session

Date: 2026-11-26. Scope: fixed seed-42 fold-01 MPS checkpoint and synthetic raw events.
The operator completed the notebook locally. This review retains the measured evidence without training or final-test reads.

## Export result

The [export verification](bundle-verification.json) passed all 4,587 existing fold-01 validation epochs.
Class disagreements were zero. Maximum probability difference was 7.450580596923828e-07, below the 2e-6 tolerance.
History lengths and reset reasons matched. Weights and fitted transforms stayed fixed, and no gradients were created.
The difference concerns numeric parity, not a change in model accuracy.

Bundle: `motion-seed42-fold01-raw-session-v1`.
Runtime revision: `7c28fc967450aebd2810643b6540e6ec02907e10`.
The sanitized checkpoint has its own file hash. It differs from the original checkpoint file hash because packaging removes unsupported fields.
That file-hash difference does not imply different weights; publication checks the restored model's predictions and integrity.

## Synthetic session result

The [saved responses](synthetic-responses.jsonl) contain five records from 76 input records.

| Output | Meaning |
| --- | --- |
| Session started | The configured bundle and checkpoint identify the session |
| Prediction at 0 seconds | [0,30) completed; history contains one epoch |
| Prediction at 30 seconds | [30,60) completed; history contains two epochs |
| Skip at 60 seconds | [60,90) completed without sensor coverage |
| Session finished | Three complete epochs were accounted for; model integrity passed |

Both synthetic predictions select Light. The example contains no stage targets, so this cannot measure accuracy or class bias.
The history length changes even though the model's selected class stays the same. Its probabilities also change.
The gap emits a coverage skip, rather than a fabricated stage prediction.
This example ends after the gap; the post-gap reset behavior is covered by synthetic runner tests, not demonstrated here.
Finish at 95 seconds discards two observations from the incomplete [90,120) interval.
Peak pending events were 33, below the 8,192 limit. This small example does not stress the queue limit.
The final summary confirms unchanged weights, evaluation mode, and no gradients.

## Packaging check

The review preserved a matching wheel before changing the checkout revision.
The [isolated wheel check](wheel-verification.json) reproduces all five response records exactly from the copied bundle and synthetic inputs.
The process cannot access the original artifact, data, module, or schema trees, or invoke Git.
This tests portability within the current Python/MPS environment. It does not validate another device or operating system.

The matching wheel is retained locally at:
`artifacts/raw-session-demo/runtime-wheel-7c28fc9/sleep_classifier-0.1.0-py3-none-any.whl`.
After a source commit or squash merge, use that matching wheel or export under a new bundle identifier.
Do not change the manifest revision or weaken the loader check.
The report's cached inspection verifies bundle file hashes, export evidence identity, and response bundle/checkpoint identifiers without restoring weights.

## Limits and next step

This combines the raw-event adapter and frozen bundle successfully. It adds no new quality scores or live sensor-delivery evidence.
The full recorded-development parity evidence remains notebook 46. The final test stays reserved.
Model selection, live clock synchronization, transport retries, concurrent ownership, and deployment validation remain separate concerns.

Next, add economical CI for lint, selected synthetic tests, and package contents. Exclude training and datasets.
Then add a local FastAPI interface around the existing runner, with session ownership, expiry, request limits, and explicit lifecycle errors.
Keep HTTP policy separate from feature preparation and model behavior. Add Docker after defining a validated container device profile.
