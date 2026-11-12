# Architecture options for causal sleep staging

Assessment date: 2026-11-10. This note proposes experiments. It reports no new training.

## Historical decision: context comparison

Test longer history before replacing the current Transformer.
Change context from 10 to 30 completed epochs, which changes the available span from 5 to 15 minutes.
Keep the 40-epoch cap, patience 12, seed 42, and the existing five participant folds.
Replicate a useful result with seeds 43 and 44 before selecting a replacement default.

The [three-seed budget comparison](experiments/transformer-budget-seeds-2026-11-10/README.md)
supports retaining the larger budget as the next development reference.
Its checkpoints use validation macro-F1, so extra checkpoint opportunities can explain some gains.
The shared folds and repeated seeds do not provide independent evidence from new participants.

## Project constraints

The target is four-class classification of a completed 30-second epoch, using only information available at its end.
It is not numerical forecasting of future sensor values.
See the [prediction contract](first-experiment.md#prediction-contract).
The current inputs contain four heart-rate summaries and two motion summaries per epoch.
These are correlated derived features from two modalities, not six independent sensors.
See [feature definitions](../src/sleep_classifier/features.py).

The development cohort contains 26 participants; three final test participants remain reserved.
Overlapping epochs do not increase the number of independent participants.
The current model has one Transformer layer, width 16, and 2,564 trainable parameters.
Its ten temporal tokens already produce a small attention problem.
See the [reference configuration](../configs/transformer-budget-40-12.yaml)
and [model](../src/sleep_classifier/transformer_model.py).

## Paper evidence and project fit

The source column describes the cited paper. The fit column gives project-specific inference.
Forecasting benchmark rankings do not establish a universal leader or a sleep-stage classification ranking.

| Candidate | Primary-source evidence | Fit for this project |
| --- | --- | --- |
| PatchTST | Uses temporal patches and shared channel-independent Transformer weights. Patching reduces attention cost and supports longer histories. [Paper, 2022/2023](https://arxiv.org/abs/2211.14730). | A small patch encoder is a reasonable later candidate if longer history helps. Ten epochs provide little computational reason to patch. Coarse patches could hide stage transitions. |
| iTransformer | Embeds each variable's history as a token. Attention learns relationships between variables instead of temporal tokens. [Paper, 2023/2024](https://arxiv.org/abs/2310.06625). | Six feature tokens offer a possible cross-feature experiment. Four summarize the same heart-rate stream, so high-dimensional multivariate forecasting results transfer weakly. |
| TiDE | Uses an MLP encoder-decoder for long-term forecasting, including covariates and nonlinear dependencies. [Paper, 2023/2024](https://arxiv.org/abs/2304.08424). | A small MLP classifier is a useful separate reference. Removing the forecast decoder and adding a class head creates an adaptation, not unchanged TiDE. |
| N-HiTS | Combines multi-rate input sampling and hierarchical interpolation to construct forecasts at different scales. [Paper, 2022](https://arxiv.org/abs/2201.12886). | Multi-scale history may help later. Forecast-horizon interpolation has no direct four-class output equivalent, and pooling ten epochs may discard useful timing. |
| N-BEATS | Uses fully connected residual blocks with backcast and forecast outputs for univariate point forecasting. [Paper, 2019/2020](https://arxiv.org/abs/1905.10437). | Residual MLP ideas are reusable. Its original output task and univariate design require substantial changes for multivariate sleep classification. |
| EMTSF | The 2025 preprint combines forecasting experts, including xLSTM, enhanced Linear, PatchTST, and minGRU, through a Transformer gating network. [Paper](https://arxiv.org/abs/2510.23396). | Multiple experts and routing add selection, training, and engineering choices. The small participant cohort makes this a poor next milestone without stronger single-model evidence. |

The papers support their own forecasting claims under their reported datasets and protocols.
They do not show that these models outperform this project's causal classifier.
EMTSF's broad performance claim remains the authors' benchmark claim, not a verified current universal ranking.

## Adaptation requirements

Every candidate needs a four-class head and a classification objective.
Supply only the current completed epoch and earlier history; never supply future sensor values or future covariates.
Preserve padding masks, participant boundaries, and causal sequence construction.
Use causal attention masks when hidden states can otherwise access epochs later than their prediction target.
An encoder may attend throughout an already available history window when it predicts only that window's final epoch.
Fit scaling and other learned preprocessing on each fold's training participants only.
Do not normalize with statistics from a participant's complete future night.
Keep class weighting, checkpoint selection, and eligible evaluation epochs comparable.

A PatchTST-inspired classifier must explicitly define how it combines channels before the class head.
Channel-independent encoding does not prove that these six correlated summaries should remain separate.
Patch boundaries must exclude future epochs and preserve valid short-history behavior.
Label this model as an adaptation and compare it with the existing temporal encoder.

## Historical bounded comparison: context length

1. Reuse the measured context-10, budget-40/12, seed-42 reference.
2. Prepare one context-30 configuration with the same folds, features, optimizer, width, depth, and checkpoint rule.
3. Set both `context_epochs` and `transformer.max_context` to 30.
4. Have Timo start the five new fits through an explicit training command.
5. Compare the same eligible epoch IDs and report any short-history handling differences.
6. Report fold and participant macro-F1, class precision/recall/F1, accuracy, selected epochs, and measured runtime.
7. Review the trade-offs before authorizing seed replication or another architecture.

The learned positional table adds `(30 - 10) × 16 = 320` parameters, raising the total to 2,884.
This tests a practical longer-context configuration; it is not a strictly parameter-matched context ablation.
The attention map grows from 100 to 900 entries per head, but absolute cost remains small at this scale.
Measure total runtime rather than inferring it from attention dimensions alone.

If longer history helps, consider a small PatchTST-inspired patch classifier next.
If it does not, inspect feature limits and class errors before adding architectural complexity.
Keep the three final test participants untouched until model selection is complete.

## Current architecture milestone

The longer context has been replicated. Optimizer diagnostics did not improve the primary stage-classification objective.
The next [causal patch comparison](causal-patch-transformer.md) uses raw context thirty and three epochs per token.
It jointly embeds the six related summaries and reuses the causal Transformer backbone.
This is a PatchTST-inspired classifier, not the paper's channel-independent forecasting architecture.
The candidate remains small at 2,756 parameters and is prepared for five-fold seed-42 evaluation.
