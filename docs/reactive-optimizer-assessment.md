# Reactive optimizer assessment

Reviewed on 2026-11-03. Source inspection and offline synthetic checks completed. No real-data training or agent calls occurred.

The short link resolves to [Reactive Optimizers](https://xingjianbai.com/reactive-optimizers/).
Its implementation is [OPAL](https://github.com/xingjian-bai/opal), version 0.3.10 at
[commit e999c21](https://github.com/xingjian-bai/opal/tree/e999c21ee1a05891ea09d7d29bf2cd6597ca686b).
The findings below refer to that revision.

## Decision

OPAL is a feasible experimental addition to our PyTorch loop.
An optional, offline `backend="keep"` smoke experiment now tests the actual package API with our GRU.
Then prepare a separate, operator-approved reactive run with a recorded budget.
Keep Adam as the reference optimizer.

OPAL changes optimizer code during one training run. It does not schedule a grid of model experiments.
A small experiment runner still helps compare context length, learning rate, and random seeds.
These tools address different work: repeatable comparisons versus adaptive optimization within a run.

Our current GRU has 3,972 parameters and trains quickly.
The upstream examples report substantial agent overhead for short runs.
A paid pilot can test the idea, but faster training is not an established benefit for this project.
No published result establishes a benefit for wearable sleep staging.

## Interface and fit

The [wrapper API](https://github.com/xingjian-bai/opal/blob/e999c21ee1a05891ea09d7d29bf2cd6597ca686b/opal/wrapper.py)
accepts an existing PyTorch optimizer, the model, a step budget, and a goal sentence.
Our Adam optimizer, single-process loop, and float32 tensors fit this interface.
The project uses no scheduler, mixed precision, distributed training, or gradient accumulation.

The proposed integration has these explicit operations:

1. Construct Adam with the existing settings.
2. Wrap Adam with `opal.wrap(optimizer, model, total_steps=..., goal=..., backend="keep", summarize=False)`.
3. Keep gradient clipping in our loop for the first equivalence test.
4. Replace `optimizer.step()` with `optimizer.step(loss=loss)` in the OPAL path.
5. Call `optimizer.close()` in a `finally` block, including after early stopping.
6. Keep validation and checkpoint selection in our existing code.

Use `total_steps = epochs * len(train_loader)` as the maximum update budget.
Early stopping can end the run before that budget.
Record the actual update count and retain the existing best-validation-macro-F1 checkpoint rule.

Suggested goal: “Train a causal four-class wearable sleep-stage GRU; select the checkpoint by participant-held-out validation macro-F1.”
The final test participants remain outside this development experiment.
Do not send final test measurements through `log` or `report_validation`.

Keep clipping external initially so its behavior stays fixed.
OPAL also accepts `clip_grad_norm=1.0`, which lets generated code control clipping.
That transfer would be a separate experimental change.

`opt.log(validation_macro_f1=value)` records a measurement without exposing it to the agent by default.
`show_agent=True` changes that behavior.
The pilot should leave validation hidden from the agent and preserve our checkpoint selection.
This avoids adding another form of validation adaptation during the first comparison.

## Dependencies and devices

The [package metadata](https://github.com/xingjian-bai/opal/blob/e999c21ee1a05891ea09d7d29bf2cd6597ca686b/pyproject.toml)
requires Python 3.10+, PyTorch 2.5+, and NumPy 1.26.4+ for our Python version.
Our locked environment satisfies those version floors.
`opal-optimizer==0.3.10` is pinned in the optional `opal` dependency group. The uv lock records package hashes.
The unrelated package named `opal` uses the same import name and must not coexist.

CPU is supported upstream.
The wrapper explicitly synchronizes MPS when measuring updates, so it contains an MPS execution path.
This does not prove that every generated optimizer supports MPS.
Test our actual GRU forward pass, backward pass, Adam update, and keep wrapper on the Mac.

The [upstream guide](https://github.com/xingjian-bai/opal/blob/e999c21ee1a05891ea09d7d29bf2cd6597ca686b/docs/GUIDE.md#safety)
states that the agent's macOS shell cannot access Metal.
It can test generated code on CPU, while the training process can use MPS.
Record any device-specific failure rather than silently falling back to CPU.

## Agent access and cost

`backend="keep"` makes no model calls and needs no CLI login.
The default live backend requires Claude Code and an existing login on the training machine.
Its default model is `claude-opus-5-5`, with high effort.
Usage is charged to the configured account.
Do not authenticate, run `opal doctor --live`, or start a live backend as part of report generation.

Before a live pilot:

1. Run the offline checks and measure seconds per training update.
2. Run `opal estimate --total-steps N --sec-per-step S` with those measurements.
3. Agree on the account, model, and spending limit.
4. Set `max_cost_usd` and `summarize=False` explicitly.
5. Set decision and availability timeouts to bound waits.
6. Record the full wrapper settings and exact launch command.

The Claude budget check stops further decisions when the remaining estimate is insufficient.
Its CLI budget can overshoot by one model call; this is not a strict billing ceiling.
Automatic summaries use separate calls outside the reported optimizer budget, so disable them.
Use `backend_options` to set `timeout`, `unavailable_timeout`, and `retries` explicitly.
Keep the default decision sizing until the timing estimate is available.
Use `blocking=True` for the first pilot so application timing does not depend on agent response speed.

An experimental Codex backend exists, but `max_cost_usd` does not constrain it.
Its token budget is soft and an active turn can exceed it.
The default Claude path gives a clearer first cost-controlled comparison.
See the [cost controls](https://github.com/xingjian-bai/opal/blob/e999c21ee1a05891ea09d7d29bf2cd6597ca686b/docs/GUIDE.md#cost-and-limits).

## Information and execution boundaries

The agent receives the goal, parameter names and shapes, training losses, and analysis reports.
It also receives source copies unless `share_code=False`.
Generated analysis can inspect parameters and gradients.
Generated optimizer and analysis modules execute inside the training process through Python `exec`.
They have that process's access to files and objects.
The agent shell sandbox does not isolate those modules.
`share_code=False` prevents source copies, but does not enforce data isolation.
Upstream boundary checks report suspicious operations; they do not block them.

For a live pilot, use a disposable environment containing only prepared development inputs and required code.
Keep raw files and final test observations outside that environment's accessible filesystem.
Expose no project secrets beyond the required backend authentication mechanism.
Review saved programs and boundary events before accepting the result.
These requirements follow the actual code execution path, not just prompt instructions.

## Reproducibility and acceptance

Run 100 synthetic updates with plain Adam and 100 with the keep wrapper.
Use identical model states, batch order, seeds, and clipping.
Compare every loss and the final parameter tensors on CPU.
Repeat the comparison on MPS with an explicit numerical tolerance if exact equality fails.
Record the observed differences and backend versions.

For the live pilot, preserve OPAL's programs, decision records, settings, costs, and events alongside model provenance.
Save optimizer state if resume support is added; our inference checkpoint alone cannot resume an OPAL conversation.
`opal.replay` can reuse a final program, but that does not reproduce the original sequence of optimizer edits.
Blocking execution fixes application timing given the answers; new agent answers can still differ between runs.

Compare Adam and OPAL with the same features, split, seed, maximum updates, and checkpoint rule.
Report macro-F1, all class scores, participant scores, wall time, agent wait time, cost, and applied decisions.
Treat the first paid result as exploratory evidence.
Use further seeds only after the pilot proves operationally useful.

## Implemented offline check

The `sleep_classifier.opal_smoke` CLI uses synthetic sequences and the actual GRU class.
Its backend is fixed to `keep`; there is no live-backend option.
It preserves existing output directories and writes losses, parameter hashes, timings, and package versions.
The real-data trainer remains unchanged by this OPAL integration.

```bash
uv run --locked --group gru --group notebook --group opal python -m sleep_classifier.opal_smoke \
  --output-dir artifacts/opal-smoke-cpu-new --steps 100
uv run --locked --group gru --group notebook --group opal python -m sleep_classifier.opal_smoke \
  --output-dir artifacts/opal-smoke-mps-new --steps 100 --device mps
```

On this Mac, both 100-update checks passed with OPAL 0.3.10 and PyTorch 2.15.0.
Each backend produced identical per-update losses and matching final parameter hashes between plain Adam and the keep wrapper.
CPU and MPS hashes differ from each other, as expected from device numerical differences.
The synthetic workload uses eight hidden units and batches of eight five-epoch sequences.
Its timings do not predict full-dataset training throughput.
The recorded check covers integration equivalence, not reactive optimization quality or generated-code compatibility.
See [the compact verification record](results/opal-smoke-2026-11-03.json).

## Codex pilot for this project

Rechecked the upstream guide on 2026-11-04.
The installed 0.3.10 source already includes an experimental Codex backend.
No fork or replacement of Claude Code internals is required.
Select the backend when wrapping the optimizer:

```python
optimizer = opal.wrap(
    optimizer,
    model,
    total_steps=maximum_updates,
    goal="Improve training loss for the causal sleep-stage GRU",
    backend="codex",
    backend_options={"model": "gpt-6.1-sol", "effort": "high", "max_tokens": 20000},
    summarize=False,
    share_code=False,
    blocking=True,
)
```

This is an integration example, not an enabled project training option.
Use a separately authenticated CODEX_HOME as required by OPAL.
The token limit is soft; max_cost_usd does not constrain this backend.
Select the account and token budget before launching a live run.
Keep validation metrics hidden from the agent for the first comparison.
Retain the generated programs and decision logs with the run artifacts.
The current five-trial suite uses Adam and never calls this backend.
The separate keep smoke check made no language-model calls.
See the [upstream Codex instructions](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md#experimental-codex-backend).
