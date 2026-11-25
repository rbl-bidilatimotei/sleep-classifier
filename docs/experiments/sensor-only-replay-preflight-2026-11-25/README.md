# Sensor-only replay: one-fold MPS preflight

Date: 2026-11-25. Scope: the original fold-01 checkpoint and its six held-out development participants.
The full five-fold replay remains pending. No training or final-test observation access occurred.

The [measured report](fold-01.json) records control parity, fixed-state evidence, and paired scores.
The [source identity](source-identity.json) pins the input plan and replay implementation hashes.

| Check | Result |
| --- | ---: |
| Legacy-control epochs | 4,587 |
| Legacy maximum probability difference | 0 |
| Legacy class disagreements | 0 |
| Sensor-only predictions | 4,624 |
| Shared scored targets | 4,585 |
| Candidate-only predictions | 39 |
| Reference-only tail targets | 2 |
| Changed shared history lengths | 29 |
| Changed shared class predictions | 2 |

All legacy history lengths and reset reasons matched. Weights and fitted preprocessing stayed unchanged.
The model remained in evaluation mode. Inference created no gradients.
The candidate preserved extra unscored epochs in its history and disclosed the two tail exclusions.

Paired accuracy was 0.687241 and macro-F1 was 0.603508 for both policies.
Two changed class decisions happened to leave these aggregate scores unchanged.
Paired NLL changed from 0.853438 to 0.853743. The largest probability change was 0.240380.
These are single-fold observations, not evidence that the policy improves the model.
The preflight demonstrates that the replay can isolate a history-policy change with fixed weights.

The preflight did not create the default `artifacts/sensor-only-frozen-replay` directory.
Run notebook 45 to obtain all five folds. Do not generalize this preflight to the full development cohort.
