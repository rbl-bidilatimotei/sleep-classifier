# Notebook 19: cell-by-cell guide

The source notebook contains 26 physical cells, including Markdown and the retained empty scratch cell.
The notebook compares dropout 0.1 and 0.2 with seeds 42, 43, and 44.
It reuses all completed references and the seed-42 candidate. Only two new grouped runs required training.

## Cells and purpose

| Cell | What it does | Why it is needed |
| --- | --- | --- |
| 1 | States the question, fixed settings, reuse rules, and work budget | Defines the comparison before inspecting its scores |
| 2 | Imports basic tools, finds the repository, configures logging, and declares execution switches | Gives local and cloned checkouts the same paths; separates inspection from training |
| 3 | Explains optional environment setup | Identifies when a clone and dependency installation are necessary |
| 4 | Optionally clones the branch, installs uv from Astral, and installs locked dependency groups | Supports a fresh machine; existing local environments can skip it |
| 5 | Explains the no-gradient preflight | Separates compatibility checks from a measured training result |
| 6 | Runs neural_training --check for the seed-43 recipe | Checks prepared inputs, tensor shapes, model size, and device before optimization |
| 7 | Explains the paired experiment plan and reuse requirements | Prevents an unmatched or unnecessarily repeated comparison |
| 8 | Displays new runs and budget; verifies reused artifacts and identical participant-fold assignments | Detects damaged references, incompatible recipes, and changes to the comparison population |
| 9 | Explains preparation for the two new candidates | Distinguishes preparing fold inputs from fitting models |
| 10 | Optionally prepares new grouped runs | Writes unscaled fold inputs, provenance, and frozen configurations; it does not train or fit scaling |
| 11 | Explains explicit training opt-in and failure handling | Makes the costly operation visible and prevents accidental automatic recovery |
| 12 | Optionally trains seed 43 and then seed 44, with five fresh fits each | Produces the candidate results; scalers fit training participants only |
| 13 | Explains complete and pending pair reporting | Prevents a reused reference from being mistaken for a completed candidate |
| 14 | Checks run status and builds each complete paired report | Shows fold, participant, and class trade-offs; validates matching targets and saved metrics |
| 15 | Builds the aggregate only when every pair is complete | Reports the mean and sample standard deviation of the three seed effects without inventing missing scores |
| 16 | Plots mean fold macro-F1 and pooled REM recall by seed | Shows repeatability and whether REM sensitivity changes; recall still needs precision beside it |
| 17 | Plots pooled Wake, Deep, and REM F1 | Makes class trade-offs visible when one aggregate score improves |
| 18 | Explains the saved-probability diagnostic | Introduces information that hard-label scores cannot show |
| 19 | Validates saved probabilities, recomputes pooled NLL, and summarizes confidence on correct and wrong predictions | Checks probability loss independently of training logs; exposes possible confident errors |
| 20 | Explains the focused epoch-14 investigation and its limits | Separates a visual reference from a configured training event |
| 21 | Plots folds 01 and 04 across all seeds; reports minimum NLL, best macro-F1, largest upward loss step, and stored replay results | Tests whether the apparent jump is shared and whether loss and checkpoint selection tell different stories |
| 22 | Explains optional selected-checkpoint replay | Defines what this read-only check can establish and what it cannot reconstruct |
| 23 | Optionally restores seed-42 fold-01 and fold-04 checkpoints and evaluates each twice | Checks evaluation repeatability, saved prediction agreement, unchanged weights, and absent gradients |
| 24 | Explains HTML export | Separates a shareable executed report from the clean source notebook |
| 25 | Exports through a fresh IPC kernel with setup, preparation, training, replay, and recursive export skipped | Rebuilds the report without starting model optimization |
| 26 | Contains the retained empty scratch cell | Has no pipeline role; preserves the user's notebook structure |

The source switches default to false for setup, preparation, training, and checkpoint replay.
RUN_CHECK defaults to true. The preflight performs a forward pass without optimization.
Training and setup cells carry skip-execution tags for automated report export.
The replay cell uses MPS explicitly to match the measured local runs. Use another explicit device when necessary.
Cross-device probability agreement has a numerical tolerance and need not be bitwise identical.

The inspection cells are safeguards and scientific checks, not additional training stages.
They expose missing runs, invalid comparisons, participant sensitivity, and class trade-offs before drawing a conclusion.
The status, fold, and recipe checks use saved inputs and hashes rather than trusting a displayed table alone.

## What the metrics mean

Accuracy is the fraction of correct hard labels. It weights each scored target equally.
Light is common, so a high accuracy can coexist with weak Deep or REM detection.

Precision measures how often a predicted stage is correct.
Recall measures how many true examples of that stage the model detects.
F1 is their harmonic mean: 2 × precision × recall / (precision + recall).
Macro-F1 gives Wake, Light, Deep, and REM equal weight.

Mean fold macro-F1 averages the five fold scores equally.
Pooled macro-F1 first combines every held-out prediction, then computes the score.
These differ because F1 is nonlinear and fold populations differ.
Pooled accuracy and NLL weight targets equally, so larger folds contribute more.

The paired seed effect subtracts dropout-0.1 mean fold macro-F1 from dropout-0.2 mean fold macro-F1.
The reported sample standard deviation describes the three effects. It is not a confidence interval.
Repeated seeds and overlapping training folds do not create new independent participants.
The same validation folds also select checkpoints. The three final test participants remain unused.

## NLL and rising validation loss

NLL means negative log-likelihood. For N labeled targets:

    NLL = -(1 / N) × sum(log(probability assigned to the true class))

The logarithm is natural, so the units are nats. Lower is better.
A correct-class probability of 0.8 contributes 0.223; 0.2 contributes 1.609; 0.01 contributes 4.605.
The notebook clips at 1e-12 only to avoid taking log(0). This is a numerical guard, not learned calibration.

For this recipe, unweighted cross-entropy on logits equals softmax followed by NLL on the true class.
This equivalence assumes the current absence of class weights and label smoothing.
See [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html).

Accuracy and F1 discard probability confidence after choosing the largest probability.
A model can improve some hard labels while assigning smaller true-class probabilities to its remaining errors.
Its macro-F1 can rise while NLL rises. Lower NLL alone does not prove good calibration.
Confidence-on-error averages also compare changing error sets, not necessarily the same targets.

In cell 19, epochs means scored thirty-second targets, not optimization passes through the training data.
The same distinction applies to the checkpoint replay row count.

Training loss averages batches while parameters change and dropout is active.
Validation loss uses fixed end-of-epoch weights and evaluation mode, which disables dropout.
The two curves therefore do not measure an identical computation.
See [PyTorch Dropout](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html).

## What the focused audit establishes

The code uses fixed Adam learning rate 0.001, no scheduler, and no OPAL decisions in this experiment.
It switches between model.train() and model.eval() correctly. Validation does not shuffle its targets.
Checkpoint selection monitors validation macro-F1. It restores the best weights after the fit loop, not at epoch fourteen.
There is no configured epoch-fourteen change to the optimizer, sampler, dropout rate, or dataset.
An optimization epoch is unrelated to the time of night within a participant's recording.

The read-only replay reproduced the selected seed-42 fold-01 and fold-04 probabilities exactly on MPS.
Both passes agreed, weights stayed unchanged, and no gradients were created.
This checks the current selected checkpoint and saved-data path. It does not replay an unavailable epoch-fourteen checkpoint.

Per-epoch scalar losses cannot identify the participants or classes behind a historical spike.
Only the selected checkpoint and its target probabilities were retained.
A confidence shift, participant distribution shift, and stochastic optimizer variation remain plausible explanations.
A diagnostic repeat needs per-epoch class and participant summaries before choosing between those explanations.

See the [measured review](experiments/transformer-dropout-seeds-2026-11-11/README.md) for concrete seed and epoch values.
