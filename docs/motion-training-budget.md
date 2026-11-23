# Motion training-budget comparison

## Question and controls

Does a longer budget help LR0.0003 after three folds reached the forty-epoch cap?
Notebook 40 permits eighty epochs and retains patience twelve.
It keeps seed 42, explicit MPS, five development folds, and every feature/model/optimizer setting fixed.
The three final-test participants remain reserved. Timo runs all training locally.

The recipe retains thirty epochs of context, P3 patches, width sixteen, dropout 0.2, and 2,900 parameters.
It retains both HR channels and fixed-lag motion, with seven raw inputs and nine effective channels.
It retains weighted cross-entropy, zero-decay Adam, batch size 64, and gradient clipping 1.0.
There is no scheduler, smoothing, agent, or checkpoint-rule change.

## Historical artifact availability

The local notebook-39 output directory was empty during inspection on 2026-11-23.
Its manifest, per-epoch predictions, and trained checkpoints were unavailable.
The cause is unknown. Setting RUN_TRAINING false does not remove artifacts or prepare folds.
The historical HTML and [archived findings](experiments/motion-learning-rate-2026-11-20/README.md) remain available.
The historical HTML also has a local backup under artifacts/recovery/motion-learning-rate-2026-11-23.
Do not overwrite these records or describe aggregate scores as recovered checkpoints.

Notebook 39 is the completed forty-epoch study. Notebook 40 is this new eighty-epoch study.
Training requires a prepared manifest for the intended study. Preparation and training have separate switches.
Changing RUN_TRAINING alone cannot create missing preparation files.
Preparation now checks device availability before creating its output directory.
Missing-manifest errors explain the required preparation command and preserve existing evidence.

## Within-trajectory control

Train five fresh eighty-epoch trajectories. Save validation predictions after every completed epoch.
Also save checkpoints for epochs one through forty. These writes do not change optimization or its random stream.

For each fold, select the first strict macro-F1 maximum through epoch forty as the prefix control.
Compare it with the first strict maximum from the full trajectory, including unchanged early stopping.
If stopping occurs before forty epochs, both selections must match.
The report verifies trajectory continuity, patience, target identity, prediction hashes, and selected checkpoint links.
It recomputes scores from selected prediction rows rather than trusting aggregate tables.

This control belongs to the new run. It does not restore the original notebook-39 weights.
Both budgets share one trajectory. MPS variability means a new prefix need not reproduce historical scores exactly.
More epochs also provide more validation-selection opportunities. A selected gain alone does not establish better generalization.
Mean paired fold macro-F1 remains primary. Inspect pooled NLL, stage trade-offs, participant effects, and completed epochs.
NLL measures the probability assigned to true labels; lower values indicate better probability predictions.
Keep the final test untouched. Replicate useful gains across seeds before adopting this recipe.

## Local execution

Open notebooks/40_motion_training_budget.ipynb. Restart the kernel and run cells in order.
Each code cell explains its purpose. All execution switches default to false.
Use the new configuration: configs/motion-training-budget-grouped.yaml.
Never point this run at the historical notebook-39 output directory.

Prepare once when the new directory is absent:

```bash
uv run --locked --group gru --group notebook python -m sleep_classifier.grouped_validation \
  --config configs/motion-training-budget-grouped.yaml --prepare
```

The five folds are already prepared in this checkout; all MPS no-gradient checks passed.
See [preflight evidence](experiments/motion-training-budget-preflight-2026-11-23/README.md).
Keep RUN_PREPARATION false here. Enable RUN_CHECK for optional no-gradient preflight.
Enable RUN_TRAINING when ready. Keep RUN_SETUP false in the existing environment.
Existing results block accidental retraining. Preserve partial evidence if a run fails.

```bash
uv run --locked --group gru --group notebook python -u -m sleep_classifier.grouped_validation \
  --config configs/motion-training-budget-grouped.yaml --train
```

After all folds finish, the notebook displays paired fold, participant, pooled, and stage scores.
It writes prefix-budget-comparison.json only after all verification checks pass.
Report-only HTML execution skips preparation, checks, training, and recursive export.
No candidate scores exist before Timo completes training.
