# Sleep classifier: project overview

Updated on 2026-11-12.

## Purpose

Build a reproducible Python system that classifies completed thirty-second sleep intervals from consumer wearable measurements.
The targets are Wake, Light, Deep, and REM. NREM contains Light and Deep; it is not an additional target.
The system uses current and earlier observations only. It does not forecast the next interval or use future-night data.

## Data and preparation

The PhysioNet sleep-accel dataset contains 31 participants, with heart rate, wrist acceleration, steps, and reference sleep labels.
Two participants were excluded because repeated heart-rate timestamps contained conflicting readings.
The frozen baseline split contains 23 training, three validation, and three reserved final-test participants.

We audited timestamps, exact duplicates, sensor gaps, label mappings, step coverage, and participant eligibility before training.
Exact duplicates are removed from derived data. Raw files remain unchanged and checksum-verified.
The label timestamp convention remains an explicit epoch-start assumption.

The first model recipe uses six features: heart-rate mean, standard deviation, minimum, and maximum; acceleration-magnitude mean and standard deviation.
Steps remain excluded because incomplete history must not silently become zero activity.
Eligibility requires sufficient heart-rate observations and motion coverage within each interval.
Rejected or missing intervals break temporal sequences. Scaling fits training participants only.
There are 26 development participants and three reserved test participants after exclusions.

## Experiments and findings

| Stage | Result | What we learned |
| --- | --- | --- |
| Majority reference | 54.46% accuracy, 0.1763 macro-F1 | Accuracy can hide complete minority-stage failure |
| Logistic regression | 56.08% accuracy, 0.3045 macro-F1 | The simple features contain useful signal |
| Class-weighted logistic regression | 0.2659 macro-F1 | Higher minority recall can sacrifice other stages and overall performance |
| First small GRU | 0.4760 macro-F1 | A nonlinear sequence model improved class-level scores on the same validation split |
| Context ablation | Context one: 0.4153; context ten: 0.4760 | Earlier epochs helped in this single-seed comparison |
| Capacity and learning-rate trials | Small single-seed gains | Neither change has a replicated advantage yet |
| Three-seed patience comparison | Mean gain 0.0080, entirely from seed 42 | Training longer did not consistently change the selected checkpoint |
| OPAL integration | First attempt failed; stable-CLI retry applied one optimizer change | Operational success must be checked separately from model scores |
| Successful OPAL retry | Macro-F1 0.4836 versus Adam 0.4760 | A small gain came with class trade-offs, more updates, and agent overhead |
| Five-fold grouped GRU validation | Mean fold macro-F1 0.4836 ± 0.0329; pooled accuracy 59.45% | Performance varies across participants, and minority-stage detection remains weak |

The standard deviation above describes five development fold scores; it is not a confidence interval.
Scores across different participant populations are not direct improvement measurements.
Every development participant was validated once in grouped validation, producing 22,708 eligible held-out epoch predictions.
Each fold also selected its checkpoint; these results are not an independent outer-test estimate.
The final test participants remain reserved.

## What worked and what did not

Causal preparation, participant splits, checksum validation, and reproducible scripts gave us a trustworthy comparison framework.
Notebooks explain and launch experiments; reusable modules perform preparation, training, and reporting.
YAML settings, uv.lock, tests, saved provenance, and curated figures preserve each experiment's evidence.
Source notebooks stay output-free. Executed reports are HTML artifacts.

The GRU improved class balance relative to logistic regression, but it did not solve minority-stage detection.
Higher training fit often coincided with rising validation loss.
That pattern suggests a generalization gap without establishing its cause.
A larger model, longer patience, and adaptive optimization have not established a reliable general advantage.

The first OPAL response failed a context audit and therefore left Adam unchanged.
The successful retry used a pinned stable CLI and an earlier decision.
It improved REM F1 but reduced Deep F1, performed 2.5 times as many updates, and spent most elapsed time waiting for the agent.
The soft token cap was exceeded during the active turn. OPAL remains available, but routine experiments use conventional training.

## Current engineering direction

The small causal Transformer improved mean fold macro-F1 from 0.4836 to 0.5045 on matched development folds.
Pooled accuracy rose from 59.45% to 60.86%; Deep recall improved more than REM recall.
Only half the participants gained macro-F1, and Wake F1 declined.
See the [measured results](experiments/transformer-grouped-2026-11-09/README.md).
The wider single-layer Transformer then lost macro-F1 in all five folds: mean 0.4828 versus 0.5045.
Width 16 remains the development reference. Increasing its budget to forty epochs and patience twelve improved mean fold macro-F1 to 0.5194.
REM recall rose from 31.73% to 40.27%; Deep F1 declined slightly.
The paired budget gain repeats with seeds 43 and 44: the mean seed effect is +0.01554 macro-F1.
REM precision and recall improve in all three seeds, while Deep F1 remains nearly flat or slightly lower.
The same validation folds select checkpoints, so the gain is not an independent generalization estimate.
The seed-42 comparison of fifteen versus five minutes then improves mean fold macro-F1 from 0.5194 to 0.5467.
Wake and Deep improve, while REM F1 declines slightly. All folds gain macro-F1, but nine participants lose it.
The context gain repeats with seeds 43 and 44: all fifteen seed/fold cells improve macro-F1.
The mean seed gain is 0.03491. Context thirty becomes the working development reference.
A seed-42 dropout increase from 0.1 to 0.2 then improves mean fold macro-F1 from 0.5467 to 0.5520.
Selected-checkpoint NLL falls, but two folds lose macro-F1 and Wake F1 declines.
The completed replication improves mean fold macro-F1 and pooled NLL for all three seeds, with mixed class trade-offs.
The mean macro-F1 gain is 0.00394. Seed 44 loses accuracy and REM F1.
The next proposal instruments two fixed-recipe folds to explain validation-loss changes before another architecture experiment.
See the [measured replication](experiments/transformer-dropout-seeds-2026-11-11/README.md).
This compares model recipes, including their capacity and regularization, rather than isolating attention as a single causal factor.
Real-data training remains operator-controlled.

A shared model factory is the next narrow abstraction; datasets, training, and evaluation should be reused across architectures.
Broader refactoring should follow observed duplication and stable interfaces.
A Python upgrade needs a separate compatibility check across PyTorch, MPS, notebooks, and target environments.
Docker, FastAPI inference, economical CI, and delivery automation remain later roadmap items.

See the [roadmap](roadmap.md), [grouped results](experiments/gru-grouped-2026-11-06/README.md),
and [learning review](learning-review.md) for details and remaining questions.

The [instrumented two-fold repeat](transformer-diagnostics.md) is complete in notebook 20.
Its evidence recorder and read-only inspector provide a reusable local evaluation foundation.
They preserve per-epoch probability changes and additive class/participant loss contributions.
A future experiment browser can use these interfaces without coupling its UI to training.

The repeat exactly reproduces the original MPS trajectories and supports overfitting in probability quality.
The [measured diagnostic review](experiments/transformer-diagnostics-2026-11-12/README.md) identifies matched-error, participant, and stage contributions.
The next proposal tests one lower constant learning rate with the remaining recipe fixed.

The [lower-learning-rate diagnostic](transformer-learning-rate.md) is prepared in notebook 21.
It changes only Adam's constant learning rate to 0.0003 and reuses the completed two-fold reference.

The completed [lower-rate diagnostic](experiments/transformer-learning-rate-2026-11-12/README.md) reveals a probability-loss versus stage-detection trade-off.
Keep constant 0.001 as the reference; test a single plateau scheduler before another architecture experiment.

The [plateau diagnostic](transformer-plateau.md) is prepared in notebook 22.
It tests dynamic update sizes while retaining the existing model and checkpoint objective.

The completed [plateau experiment](experiments/transformer-plateau-2026-11-12/README.md) does not improve the primary classification objective.
The next proposal tests a causal patch-based encoder across all five development folds.

The [causal patch model](causal-patch-transformer.md) is prepared for all five development folds.
It preserves thirty raw epochs while replacing per-epoch attention tokens with ten ordered three-epoch patches.
