# Context-thirty dropout regularization experiment

Status: prepared for Timo to run locally. No participant-data dropout-0.2 optimization has run.

## Question and controls

Does stronger residual/feedforward dropout improve the fixed context-thirty classifier?
The [replicated context results](experiments/transformer-context-seeds-2026-11-11/README.md) improved macro-F1 across all fifteen seed/fold cells.
Training loss still falls while validation loss often flattens or increases.
Those curves support a regularization question; they do not diagnose class imbalance as the cause.
Test one existing regularization setting before resampling or changing architecture.

| Setting | Measured reference | Candidate |
| --- | ---: | ---: |
| Residual/feedforward dropout | 0.1 | 0.2 |
| Attention-weight dropout | 0 | 0 |
| Context epochs | 30 | 30 |
| Representation / feedforward width | 16 / 32 | 16 / 32 |
| Layers / heads | 1 / 2 | 1 / 2 |
| Trainable parameters | 2,884 | 2,884 |
| Maximum epochs / patience | 40 / 12 | 40 / 12 |

Keep seed 42, batch 64, Adam learning rate 0.001, gradient clipping one, and validation macro-F1 checkpoint selection.
Keep the six features, eligibility, natural training sampling, unweighted cross-entropy, and the same five participant folds.
Use MPS for local comparison with the measured reference. OPAL remains disabled.
Attention-weight dropout stays zero under the existing MPS-compatible implementation.
Train five fresh candidates with training-only scalers. Do not resume or retrain the reference checkpoints.
Maximum new work is five fits and 200 epochs.

PyTorch [Dropout](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html) masks activations during training and becomes identity during evaluation.
Changing this probability does not change parameter shapes or the classification head.
The candidate's stochastic training path differs, so matching epoch histories are not expected.
The intervention may help or may underfit; no accuracy gain is assumed.

## Why sampling stays unchanged

Undersampling removes training examples; oversampling repeats examples and can repeatedly expose correlated minority windows.
Neither creates more independent participants or inherently fixes a generalization gap.
Class weighting is a third intervention: it changes loss contributions without adding or removing examples.
The earlier balanced logistic-regression result showed that recall gains can have substantial precision and majority-class costs.
A future sampling study should operate on complete training windows, fit weights on training participants only, and preserve natural validation frequencies.
This experiment changes none of those choices.

## Run locally

Open [notebook 18](../notebooks/18_transformer_dropout.ipynb).
Inspect configs/transformer-dropout-02.yaml and configs/transformer-dropout-grouped-validation.yaml.
Run the no-gradient preflight and inspect the grouped plan.
Enable RUN_PREPARATION once for new output directories.
Then enable RUN_TRAINING and execute the training cell interactively.
Leave preparation disabled if a candidate manifest already exists on this checkout.
Source notebook preparation and training flags default to false.
Report export skips setup, preparation, and optimization regardless of those flags.

Equivalent commands from the repository root:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.neural_training \
  --config configs/transformer-dropout-02.yaml --check
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-dropout-grouped-validation.yaml --plan
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/transformer-dropout-grouped-validation.yaml --prepare
# Start optimization only when ready
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/transformer-dropout-grouped-validation.yaml --train
```

Candidate outputs use artifacts/transformer-dropout-grouped-validation.
Existing results are refused. Partial artifacts remain after failure; automatic resume is unsupported.
Fresh clones need prepared development features and the measured context-thirty seed-42 artifacts to reproduce the paired report.

## Compare and decide

This uses the existing architecture-recipe comparison mode because dropout is part of model configuration.
No comparison policy or training loop is changed.
The preparation manifest freezes the complete recipe. The configuration review checks that only dropout and the output path changed.
The paired report still verifies fixed optimization, folds, coverage, provenance, actual device, and exact participant/time/target alignment.

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.paired_architecture_report \
  --reference artifacts/transformer-context-30-grouped-validation/summary.json \
  --candidate artifacts/transformer-dropout-grouped-validation/summary.json \
  --comparison architecture \
  --output artifacts/transformer-dropout-comparison/summary.json
uv run --locked --group gru --group notebook python scripts/execute_notebook.py \
  --notebook notebooks/18_transformer_dropout.ipynb
```

Use paired mean fold macro-F1 as the primary comparison.
Report pooled accuracy, every class precision/recall/F1, and participant gains and losses.
Inspect selected-checkpoint validation NLL, complete histories, selected epochs, and cap hits as secondary evidence.
Checkpoint selection still uses macro-F1; do not change it to NLL after viewing outcomes.
Training loss includes stochastic dropout, while validation loss uses evaluation mode.
Increasing dropout can raise training loss mechanically. A smaller gap alone is not success.
Better NLL with materially worse macro-F1 is a trade-off to discuss, not an automatic win.

One seed and validation-selected checkpoints limit interpretation. The same development people repeat across experiments.
The final three test participants remain reserved.
Review the measured result before replicating dropout, changing sampling, or implementing an adapted architecture.
Do not expand the dropout grid or tune individual folds during this experiment.

## Local preparation evidence

On 2026-11-11, all five candidate folds were prepared on MPS without optimization.
The complete grouped plan matches the measured context-thirty seed-42 reference.
All 130 copied feature CSV files match the reference byte for byte.
The model configuration differs only in residual/feedforward dropout and output location; both models retain 2,884 parameters.
The notebook executed from a fresh kernel and exported as HTML with the reference complete and candidate not_started.
Its base-model no-gradient MPS preflight passed with a 64 × 30 × 6 input batch.
Synthetic full padded MPS updates passed for dropout 0.1 and 0.2 at contexts ten and thirty.
An identical-weight evaluation test confirms that dropout changes neither parameter shapes nor evaluation-mode outputs.
All 311 tests and lint pass, including synthetic completed-notebook reporting of selected-checkpoint NLL.
On this checkout, leave RUN_PREPARATION disabled and enable RUN_TRAINING when ready.
