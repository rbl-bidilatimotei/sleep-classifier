# Learning review before grouped validation

Updated on 2026-11-11. These conclusions describe development experiments, not final-test performance.

## What the experiments support

| Evidence | Supported conclusion | Limit |
| --- | --- | --- |
| Majority accuracy 54.46%, macro-F1 0.1763 | Overall accuracy hides minority-stage failure | Light dominates the inspected validation population |
| Logistic regression macro-F1 0.3045 | Six sensor summaries contain predictive signal | REM recall remains very low |
| Balanced logistic regression macro-F1 0.2659 | Weighting changes the class trade-off | Minority recall gains did not produce a better overall reference |
| GRU reference mean macro-F1 0.4916 across three seeds | The GRU improves class-level scores on the same validation split | The same three people were evaluated repeatedly |
| Context-one macro-F1 0.4153 versus context-ten 0.4760 | Temporal context helped in the seed-42 ablation | One seed and one participant split |
| Patience mean paired gain 0.0080 | Longer patience helped one seed | Two seeds selected the same checkpoint under both recipes |
| OPAL retry macro-F1 0.4836 versus Adam 0.4760 | The integration worked and changed training behavior | One seed, class trade-offs, more updates, and substantial agent overhead |
| Grouped GRU mean fold macro-F1 0.4836 ± 0.0329 | Participant-held-out results vary across the 26 development people | Checkpoints were selected on the same development validation folds |

## What remains uncertain

Falling training loss with rising validation loss is consistent with overfitting or participant distribution differences.
The curves do not identify the cause or prove that every class metric deteriorates each epoch.
Larger hidden size and lower learning rate have no replicated advantage yet.
The first six features may omit useful information, but that is a hypothesis to test rather than an established bottleneck.
Steps remain excluded because missing history needs explicit coverage treatment.
The epoch-start label convention remains a documented preparation assumption.

## Engineering lessons

A completed training process does not prove that an agent decision succeeded.
Separate operational status from model scores and retain decision records.
A checkpoint selected before an intervention cannot measure that intervention's effect.
A soft token cap can be exceeded within a single agent turn.
Keep dependency versions, source hashes, split membership, device, and model settings with results.
Preserve unsuccessful experiments instead of silently replacing them.

## What grouped validation established

There are 26 development participants, despite having many correlated epochs.
Grouped validation broadened the held-out population before further architecture tuning.
It does not create new independent people or replace an untouched final test.
The completed [grouped protocol](grouped-validation.md) explicitly states its checkpoint-selection limitation.
Selected epochs ranged from two to fifteen; early validation peaks were not universal.
The completed [Transformer comparison](experiments/transformer-grouped-2026-11-09/README.md) used identical held-out epochs.
Mean fold macro-F1 improved by 0.02087; pooled accuracy improved by 1.409 percentage points.
Deep recall improved by 7.278 points, while REM recall improved by only 0.453 points.
Wake F1 fell. Only half the participants gained macro-F1.
This supports a bounded capacity test, not a general claim about larger Transformers.
See the [project overview](project-overview.md) for a newcomer summary of the full experiment history.

## Capacity comparison

The [width-32 experiment](experiments/transformer-capacity-2026-11-09/README.md) lost macro-F1 in all five folds.
Mean fold macro-F1 fell from 0.50446 to 0.48277; only seven participants improved.
Increasing capacity alone did not improve Deep or REM scores under this recipe.
The width-16 reference reached the epoch cap in two folds, supporting a bounded training-budget test.
The wider recipe mostly selected early checkpoints, so longer training is not an established remedy for its losses.

## Training-budget comparison

The [longer width-16 budget](experiments/transformer-budget-2026-11-09/README.md) improved mean fold macro-F1 by 0.01497.
REM recall rose from 31.73% to 40.27%, with slightly higher precision. Deep F1 declined slightly.
Four folds improved and one was unchanged. The shared history prefixes match exactly in this local run.
Four selected checkpoints occurred after epoch twenty. More training opportunities helped this seed, but require replication before further tuning.

## Shared-seed budget replication

All three seeds gain development macro-F1 and REM precision/recall/F1 under the longer budget.
The mean paired seed macro-F1 gain is 0.01554 with sample deviation 0.00821.
Seed 44 has the largest paired macro-F1 gain; seed 42 has the largest absolute candidate scores.
Only seven of fifteen seed/fold cells improve; the others select the same score.
All shared histories match, so longer runs search a superset of checkpoints on the same validation folds.
This confirms a repeatable development selection gain, not independent generalization improvement.
Deep F1 remains essentially flat or slightly lower. Test more causal context before selecting another architecture family.
See the [measured seed report](experiments/transformer-budget-seeds-2026-11-10/README.md) for complete class and participant trade-offs.

## Longer causal context

Context thirty improves seed-42 macro-F1 in all five folds: mean 0.54665 versus 0.51943.
The largest gain is fold 04; the mean gain remains positive when that fold is excluded.
Wake and Deep F1 improve, while REM F1 declines. Nine of 26 participants lose macro-F1.
Post-hoc gains concentrate among targets with full thirty-epoch history; shorter-history bands lose macro-F1.
Different class mixtures and small Deep supports limit those band comparisons.
The larger position table and changed initialization remain confounders in this practical context recipe comparison.
Replicate with seeds 43 and 44 before selecting a default or moving to another architecture family.
See the [measured result](experiments/transformer-context-2026-11-10/README.md) for curves, class trade-offs, and limits.

## Replicated longer context

All fifteen seed/fold comparisons gain macro-F1. The mean seed gain is 0.03491 with sample deviation 0.00821.
Wake, Light, and Deep F1 improve for every seed; REM improves for two seeds and declines slightly for seed 42.
Context thirty is the working development reference, while independent generalization remains unmeasured.
The loss curves support a regularization question, not a diagnosis that class imbalance is the cause.
Selected-checkpoint pooled NLL improves for two seeds despite late-epoch loss increases.
Test one dropout change before resampling or another architectural change.
See the [measured context replication](experiments/transformer-context-seeds-2026-11-11/README.md).

## Dropout regularization

At seed 42, dropout 0.2 raises mean fold macro-F1 from 0.54665 to 0.55204 and lowers pooled selected NLL from 0.95569 to 0.92577.
Accuracy improves in all five folds, while macro-F1 improves in three. Fourteen participants improve and twelve decline.
REM F1 improves, Wake F1 declines, and Deep recall dips despite higher Deep F1.
Validation losses still rise after the selected checkpoint in every candidate fold.
This supports a partial regularization benefit, not elimination of overfitting or a universal metric improvement.
Repeat dropout 0.2 with seeds 43 and 44 before adopting it; reuse their measured dropout-0.1 references.
See the [complete dropout review](experiments/transformer-dropout-2026-11-11/README.md).

## Replicated dropout and the loss-curve investigation

Dropout 0.2 improves mean fold macro-F1 for seeds 42, 43, and 44 by 0.00539, 0.00488, and 0.00156.
Selected-checkpoint pooled NLL also falls for every seed. Only eight of fifteen fold cells gain macro-F1.
Deep F1 rises for every seed, but REM F1 falls for seed 44 and Wake F1 falls for two seeds.
The mean macro-F1 gain of 0.00394 is a modest development improvement, not a universal class benefit.

There is no common epoch-fourteen event. The optimizer, data, and evaluation settings do not switch then.
Minimum NLL and best macro-F1 frequently occur at different epochs.
Read-only MPS replay reproduces selected fold-01 and fold-04 predictions exactly without gradients or weight changes.
Scalar histories cannot identify the participants or confidence changes behind earlier loss spikes.
Recommend a two-fold instrumented fixed-recipe repeat before another grid or larger model.
See the [replication evidence](experiments/transformer-dropout-seeds-2026-11-11/README.md) and [complete notebook explanation](notebook-19-guide.md).

## Evidence collection before the diagnostic repeat

The next study repeats seed-42 dropout 0.2 on original folds 01 and 04 only.
Its model, split, optimizer, sampling, stopping, and selection settings remain fixed.
It adds per-epoch predictions, class/participant loss decompositions, gradient/LR records, and epochs thirteen through seventeen checkpoints.
Fixed-weight training evaluation permits a clearer training/validation comparison with dropout disabled.
Synthetic active-dropout tests preserve exact CPU histories, selected weights, predictions, and Torch random state with diagnostics enabled.
This is implementation evidence, not a new participant-data model result.
The local evidence inspector supports partial runs and verified matched-epoch comparisons without loading models.
See the [protocol and interpretation rules](transformer-diagnostics.md). Timo will run the real-data optimization locally.

## Measured diagnostic repeat

The two instrumented MPS runs exactly reproduce their original histories and selected weights.
They support probability overfitting: evaluation-mode training NLL falls while held-out NLL rises.
Matched persistent errors become more costly, while some correct labels become more confident.
The largest spikes differ: REM drives fold 01's epoch 11→12 rise; Light/Deep drive fold 04's epoch 24→25 rise.
Gradients remain finite, and the fixed learning rate stays 0.001. Late clipping is frequent, not proof of an exploding-gradient event.
Selected macro-F1 still improves after minimum NLL, so the checkpoint rule remains unchanged.
Recommend one constant-learning-rate comparison at 0.0003 before another architecture change.
See the [complete diagnostic evidence](experiments/transformer-diagnostics-2026-11-12/README.md).

## Measured lower learning rate

The lower constant rate improves selected NLL in both diagnostic folds, but mean fold macro-F1 falls by 0.01309.
Fold 01 gains 0.00076 macro-F1; fold 04 loses 0.02694. Only four of eleven participants improve.
Pooled REM recall falls from 47.09% to 36.11%, and REM NLL also worsens.
The smoother loss curves accompany slower training fit and do not establish an overall classification improvement.
Retain 0.001 as the reference. Next, test one validation-macro-F1 plateau schedule with all other settings fixed.
See the [complete measured comparison](experiments/transformer-learning-rate-2026-11-12/README.md).

## Measured plateau scheduling

The schedule activates after epoch ten in both diagnostic folds, with exact reference history prefixes through ten.
NLL and macro-F1 trajectories become smoother, but mean fold macro-F1 falls by 0.01374.
Selected pooled NLL improves to 0.92277 while REM and Deep recall decline.
Within-run smoothness is different from robustness across participants or seeds.
Both constant-lower-rate and plateau interventions trade some stage detection for probability loss.
Keep constant Adam 0.001 and test the planned patch representation on all five development folds next.
See the [measured scheduler report](experiments/transformer-plateau-2026-11-12/README.md).
