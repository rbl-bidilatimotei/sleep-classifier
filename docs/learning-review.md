# Learning review before grouped validation

Updated on 2026-11-10. These conclusions describe development experiments, not final-test performance.

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
