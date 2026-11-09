# Learning review before grouped validation

Updated on 2026-11-09. These conclusions describe development experiments, not final-test performance.

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
