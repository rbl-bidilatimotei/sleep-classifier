# Codex-powered OPAL pilot

Status: prepared, not executed on participant data. No live agent calls have run.

## Purpose

Test adaptive optimization against an unchanged Adam control.
The completed seed comparison found only a small, seed-dependent patience benefit.
OPAL is not an established solution for overfitting or participant distribution differences.
Keep grouped participant evaluation on the roadmap regardless of this pilot's outcome.

## Fixed comparison

Use the original 32-unit GRU, ten-epoch context, six features, seed 42, and patience five.
Keep the twenty-epoch cap, frozen split, train-only scaler, and device fixed.
Run a fresh Adam control, then offline OPAL keep, then one live Codex pilot.
Each configuration has a separate output directory.
OPAL recipes refuse existing output directories. Choose a fresh directory before repeating a run.
Check that the resolved device matches across all three runs.
The keep result should match Adam within numerical tolerance before running a live backend.
The saved optimizer metadata records settings, package version, and update count.
OPAL programs and decision logs remain under the run's opal directory.
Inference checkpoints do not support resuming an OPAL conversation.

## Live settings and boundaries

The proposed backend uses gpt-6.1-sol with high effort and a 20,000-token soft cap.
At most two decisions are requested. Blocking execution, disabled summaries, and disabled source sharing are fixed.
Decision timeouts are bounded. Token usage can exceed the soft cap within an active turn.
The Codex backend does not enforce a dollar budget.
Review and accept these settings with the account owner before live execution.
The notebook's default RUN_LIVE flag is false.

The training CLI requires both --train and --allow-live-agent for Codex.
Preflight validates recipes and runs a no-gradient forward pass without importing OPAL or launching an agent.
It does not verify backend authentication, billing, or generated optimizer compatibility.
The optional OPAL dependency must be installed for keep or live training.

Follow the [upstream Codex setup](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#experimental-codex-backend)
for a separately authenticated OPAL Codex home. Do not copy existing authentication files.
Use a disposable environment with only prepared development inputs and required code accessible.
Keep raw data, final test observations, and unrelated secrets outside the accessible filesystem.
Generated optimizer Python executes with training-process permissions.
Disabled source sharing does not provide filesystem isolation.
See the [source assessment](reactive-optimizer-assessment.md) for the verified execution boundaries.

## Operator commands

Install the locked optional dependencies:

```bash
uv sync --locked --group gru --group notebook --group opal
```

Run preflight without training:

```bash
uv run --locked --group gru --group notebook --group opal python -m sleep_classifier.gru_training \
  --config configs/gru-opal-codex.yaml --check
```

Use notebook 10 for the three separately enabled training cells and saved-result inspection.
Select experiment/opal-codex-pilot before merge; use main afterwards.
Report export skips all setup and training cells.

## Acceptance and interpretation

Compare best-checkpoint accuracy, macro-F1, class scores, participant scores, elapsed time, and actual updates.
Inspect generated programs, applied decisions, token counts, errors, and wait times in OPAL's run records.
Check whether an apparent improvement came with a changed effective optimization budget.
The agent receives training losses, not validation measurements through this adapter.
Validation still selects the best checkpoint and early-stopping time.
Thus validation affects how long training continues, even though metrics are not sent to the agent.
No final test observations enter this pilot.
One live run is exploratory evidence. It cannot establish that reactive optimization is generally better.
Confirm promising settings across seeds and grouped development folds.
