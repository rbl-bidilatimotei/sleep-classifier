# Model card — causal sleep-stage classifier

**Status** · Research reference as of 2026-11-23. The five saved checkpoints are evaluation artifacts. A single trained deployment model remains pending. Evaluation-bundle support preserves the research-only checkpoint role.

## Intended use

Estimate the sleep stage of a completed 30-second epoch from wrist acceleration and heart-rate summaries. Data preparation assumes the source label timestamp `t` marks the start of `[t, t + 30 seconds)`. Physical sensor acquisition timing requires separate verification. The earliest prediction time is `t + 30 seconds`, once the interval summaries are available. The prediction describes that interval using current and earlier observations only.

| Output index | Stage | Source stage |
| ---: | --- | --- |
| 0 | Wake | Wake |
| 1 | Light | N1 and N2 |
| 2 | Deep | N3, including legacy N4 if present |
| 3 | REM | REM |

The intended use is research on wearable sleep staging. Future-epoch forecasting, clinical diagnosis, treatment decisions, and replacing polysomnography are outside the intended scope. [Target mapping and epoch convention](session-04.md)

## Working recipe

| Component | Setting |
| --- | --- |
| Model | Causal patch Transformer with three-epoch patches (P3) and 2,900 parameters |
| Context | Up to 30 contiguous completed epochs, including the target epoch |
| Encoder | Width 16, two attention heads, one layer, feed-forward width 32, dropout 0.2 |
| Training | Adam with constant learning rate 0.001, weighted cross-entropy, training seed 42, MPS, 40 epochs maximum, patience 12 |
| Prepared heart-rate inputs | Mean, standard deviation, minimum, and maximum |
| Prepared acceleration inputs | Magnitude mean, magnitude standard deviation, and fixed-lag motion change mean |
| Internal channels | Seven prepared inputs plus window-local heart-rate deviation and its availability flag, for nine effective channels |

Heart-rate deviation compares each epoch's mean with up to ten strictly earlier means in the same window. It requires at least four earlier means. Both channels use only the retained window and reset at history boundaries. The fixed-lag motion feature averages changes between supported neighboring one-second grid ticks within a completed epoch. Each fold fits its own motion imputation and normalization on training participants only. [Working configuration](../configs/fixed-lag-motion-mps.yaml) · [Motion feature and missing values](fixed-lag-motion-ablation.md) · [Heart-rate transform](window-local-heart-rate.md)

## Development data and evaluation

PhysioNet sleep-accel v1.0.0 contains 31 participants. The preparation process excluded two participants with unresolved conflicting heart-rate readings at repeated timestamps. The remaining 29 comprise 26 development participants and three reserved final-test participants. The reported experiments and replay used development observations only. [Dataset version](../README.md#data-attribution) · [Cohort and exclusions](project-overview.md#data-and-preparation)

Five participant-grouped folds produced 22,708 eligible validation predictions. Each fold uses separate participants for training and validation. Each fold selected the earliest checkpoint with the highest validation macro-F1. The same 26 participants informed repeated recipe choices. The scores describe adaptive development validation. [Grouped configuration](../configs/fixed-lag-motion-mps-grouped.yaml) · [Measured comparison](experiments/fixed-lag-motion-mps-2026-11-20/README.md) · [Selection procedure](experiments/motion-training-budget-2026-11-23/README.md)

### Seed 42 on MPS — 2026-11-20

| Metric | Value |
| --- | ---: |
| Mean fold macro-F1 (primary) | 0.58534268 |
| Pooled macro-F1 | 0.590389 |
| Pooled accuracy | 0.662982 |
| Selected pooled negative log likelihood (NLL) | 0.85845395 |

| Stage | Pooled precision | Pooled recall | Pooled F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Wake | 0.64743 | 0.43592 | 0.52103 | 1,904 |
| Light | 0.69799 | 0.78893 | 0.74068 | 12,825 |
| Deep | 0.61672 | 0.48958 | 0.54584 | 3,119 |
| REM | 0.57925 | 0.53086 | 0.55400 | 4,860 |

Mean fold macro-F1 weights each fold equally. Pooled metrics combine the five sets of validation predictions before scoring. Support counts true-stage epochs. [Seed-42 summary](experiments/fixed-lag-motion-mps-2026-11-20/summary.json) · [Selected NLL](experiments/motion-mps-seeds-2026-11-20/selected-nll.json)

The first logistic-regression baseline reached 56.08% accuracy on a three-participant validation split. The grouped causal Transformer later reached mean fold macro-F1 of 0.5045 versus 0.4836 for the grouped GRU. The baseline and grouped results use different validation populations. [Baseline](baseline-results.md) · [Grouped Transformer and GRU comparison](experiments/transformer-grouped-2026-11-09/README.md)

### Recipe replication

The motion feature improved mean fold macro-F1 over the matched no-motion recipe at all three MPS training seeds. The motion recipe scored 0.58016 at seed 43 and 0.58955 at seed 44. All seeds reuse the same development participants and folds. Seed 42 identifies the working checkpoints used in the chronological replay. [Motion replication results](experiments/motion-mps-seeds-2026-11-20/README.md) · [Seed summary](experiments/motion-mps-seeds-2026-11-20/summary.json)

## Chronological replay — 2026-11-23

Notebook 41 reproduced offline validation on MPS for all 22,708 eligible epochs using the saved seed-42 fold checkpoints. The largest absolute probability difference was `8.94e-7`, within the `2e-6` tolerance. All predicted classes agreed. Offline probabilities matched the saved validation probabilities exactly.

All history lengths matched offline preparation. The 168 history starts comprise 142 segment changes, 21 participant changes, and five fold initializations. Model parameters and buffers stayed unchanged. Replay created no gradients and performed no training. The audit starts from prepared epoch summaries and verifies the local MPS inference path. [Replay report](experiments/chronological-inference-2026-11-23/README.md) · [Replay summary](experiments/chronological-inference-2026-11-23/summary.json) · [Independent checks](experiments/chronological-inference-2026-11-23/verification.json)

## Limitations

- Deep and REM recall are below Light recall. The model classifies many Deep and REM epochs as Light. Wake recall is also low. Light accounts for over half of the targets and has the largest effect on pooled accuracy. [Per-stage evidence](experiments/fixed-lag-motion-mps-2026-11-20/summary.json)
- Short histories scored lower than full-context histories in the seed-42 post-hoc analysis. Missing or ineligible epochs break history and can return inference to short context. The reported scores cover eligible epochs only. [Temporal slices](experiments/fixed-lag-motion-mps-2026-11-20/temporal-slices.json) · [Preparation rules](session-04.md)
- Macro-F1 measures class decisions. Calibration requires separate evidence, and NLL alone cannot isolate calibration error. [Probability-quality results](experiments/motion-mps-seeds-2026-11-20/README.md#probability-quality-improves-at-two-seeds-while-late-overfitting-remains)
- The 26-person development cohort and repeated validation choices limit estimates of generalization. The three reserved participants remain unevaluated. Transfer to other wearables, populations, and backends requires separate validation. [Development scope](experiments/motion-training-budget-2026-11-23/README.md) · [Replay scope](experiments/chronological-inference-2026-11-23/README.md#timing-and-interpretation-limits)

The [label-free preparation audit](label-free-sensor-audit.md) identifies an additional inference-policy limitation.
Research segmentation resets after unscored epochs, even when sensor coverage is sufficient.
Sensor-only segmentation preserves those epochs and changes 566 shared development contexts.
Raw feature parity passed. The [full frozen replay](experiments/sensor-only-replay-2026-11-25/README.md) verifies sensor-only reset behavior.
On 22,705 shared development targets, accuracy changes from 66.298% to 66.426%, and macro-F1 from 0.590391 to 0.593919.
All legacy controls match exactly. The policy changes 177 class decisions within 566 changed histories; no weights or fitted transforms change.
Gains vary across participants and folds. REM F1 falls slightly. This is a separate input-policy result, not retraining or final-test evidence.
The frozen research scores and checkpoints remain unchanged.

## Requirements for a deployment model

Use all five fold checkpoints as evaluation evidence. Freeze a separate procedure to select and train one model before accessing the final-test set. The bundle contract specifies requirements for future export and loading. Recorded raw-event parity has passed. Live source validation and final-test evaluation remain pending. [Inference bundle contract](inference-bundle-contract.md)

## Evaluation-bundle implementation

The [exporter and loader](evaluation-bundles.md) package a selected evaluation fold with its fitted preprocessing and runtime identity.
Publication requires full selected-fold parity against saved development predictions. No training or model-quality improvement is implied.

The [bounded raw-event adapter](raw-event-adapter.md) now prepares epochs with explicit watermark, session-end, and late-event rules.
The [full adapter audit](experiments/raw-event-adapter-2026-11-26/README.md) preserves all 24,738 batch epochs and 23,674 sensor-only predictions exactly.
Both chunk sizes stay within the declared queue cap. This validates ordered recorded replay, not live device timing or process RSS.
This adds an input component, without changing the model, fitted preprocessing, or completed evaluation results.

The [single-session JSONL runner](raw-session-inference.md) connects the adapter to a verified evaluation bundle.
It reports coverage skips and session integrity explicitly. Synthetic integration evidence does not establish live clinical performance.
