# Controlled GRU follow-up

The first GRU improved macro-F1 from 0.3045 to 0.4760 with almost unchanged accuracy.
Deep recall improved substantially. REM recall remains 14.38%.
The result shows learned class distinctions, but it is not sufficient for deployment.

Training continued for six epochs. Epoch 1 supplied the restored checkpoint.
Training loss kept falling while validation loss fluctuated.
This pattern does not justify assuming that more capacity or more epochs will help.

## Next experiment

Use notebook 08 or the suite CLI. Keep the same features, targets, split, seed, and device across trials.

| Trial | Change from reference | Question |
| --- | --- | --- |
| reference | None | What is the result on the selected device? |
| context1 | Context 10 → 1 epochs | Does temporal history help this architecture? |
| learning-rate-0003 | Learning rate 0.001 → 0.0003 | Does a smaller update improve validation performance? |
| patience12 | Patience 5 → 12 | Does validation performance recover with more training? |
| hidden64 | Hidden units 32 → 64 | Does more capacity help with fixed inputs? |

Each run has a maximum of 20 epochs. The suite has at most 100 training epochs.
The reference rerun avoids mixing CPU and MPS effects with configuration changes.
Early stopping still restores the best validation macro-F1 checkpoint.
The longer-patience trial directly tests the proposed stopping change without combining it with increased capacity.
These are exploratory comparisons on three validation participants, not five-fold validation.
Confirm promising settings across seeds and participant folds before drawing broad conclusions.
Repeated tuning can overfit this validation set. Keep the final test set untouched.

Inspect the plan without reading participant data or training:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.gru_suite --config configs/gru-followup.yaml --plan
```

Launch all five trials when ready:

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.gru_suite --config configs/gru-followup.yaml --train
```

The runner creates isolated trial folders and saves a summary after each trial.
It refuses an existing suite output directory. Choose a new directory for another suite.
It does not resume incomplete runs. Completed trial artifacts remain available after failure.
Each metrics file records preparation provenance, device, package versions, history, and class scores.
The summary records the metrics file hashes.

## Local execution and portable notebooks

Auto chooses CUDA, then MPS, then CPU. An explicit unavailable backend raises an error.
MPS passed synthetic forward, backward, Adam, and padding tests on the M4 Pro with PyTorch 2.15.0.
This verifies compatibility, not a speed advantage. Device arithmetic can change measured results.
Notebooks use the current checkout or SLEEP_CLASSIFIER_ROOT. They do not depend on a hosted provider.
The optional setup cell can clone the repository and install uv on another machine.

The HTML runner uses IPC when supported. This avoids the local unencrypted TCP kernel warning.
Unsupported platforms retain TCP transport. Interactive Jupyter servers control their own kernel transport.
A small HTML template adds descriptive plot alt text while preserving the existing Lab appearance.
No warning filter hides these messages.

## Reactive optimization

The [OPAL assessment](reactive-optimizer-assessment.md) records the source review and a concrete pilot path.
The optional smoke command tests the wrapper with synthetic data and no agent calls.
The offline keep backend cannot establish whether reactive optimizer edits improve this task.
A live pilot needs a chosen account and budget because generated optimizer code runs during training.
OPAL changes a run's optimization. The suite manages comparisons between runs.
