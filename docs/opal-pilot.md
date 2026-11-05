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

## Codex CLI discovery and login

The desktop app does not guarantee a codex entry on the notebook process's PATH.
uv installs the Python environment; the Codex CLI is an external executable.
Discovery prefers optimizer_agent.binary, then CODEX_BINARY, then PATH.
On macOS, it also checks the CLI bundled in ChatGPT.app under /Applications or ~/Applications.
An invalid explicit executable raises an error instead of silently choosing another installation.

The local app-bundled executable was verified on 2026-11-05 as codex-cli 0.166.0-alpha.6.
No additional package installation was needed on this machine.
For another machine without that executable, install the official @openai/codex CLI through npm.
See [official installation guidance](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex).
Do not install an unrelated Python package named codex into the venv.

Run this no-inference readiness check before training:

```bash
uv run --locked --group gru --group notebook --group opal python -m sleep_classifier.gru_training \
  --config configs/gru-opal-codex.yaml --check-agent
```

If the isolated login is missing, the check prints the exact Terminal command.
Complete that interactive login yourself, then rerun the readiness check.
It uses OPAL's separate ~/.codex-opal home, not the current desktop login.
No credentials are copied or printed. No model requests run during this check.
Live training checks readiness before creating the output directory.
Normal --check remains a data/model preflight and does not check authentication.

The isolated home must exist before Codex can load its configuration.
Create it before the first login:

```bash
mkdir -p "$HOME/.codex-opal"
CODEX_HOME="$HOME/.codex-opal" \
  "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex" login --device-auth
```

A missing directory produces a configuration error before authentication starts.
An existing empty directory correctly reports Not logged in until authentication completes.
