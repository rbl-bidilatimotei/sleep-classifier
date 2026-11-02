# First measured baseline

Timo ran notebook 05 locally. This report was recorded on 2026-11-02.
The saved metrics snapshot is [baseline-validation-2026-11-02.json](results/baseline-validation-2026-11-02.json).
Its SHA-256 identifies the original metrics file. No model was retrained during this analysis.

| Model | Validation accuracy | Macro-F1 | REM precision | REM recall | REM F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority (light) | 54.46% | 0.1763 | 0% | 0% | 0 |
| Logistic regression | 56.08% | 0.3045 | 34.29% | 1.96% | 0.0371 |

Both scores use 2,780 retained epochs from 2,884 scored validation epochs (96.39% coverage).
Only three participants form this validation set. These are not final test results.
One participant has 89.30% coverage; the other two have 99.80% and 100%.
Pooled coverage does not describe every participant equally.

## Interpretation

The majority model always predicts light sleep. It establishes the class-frequency reference.
Logistic regression adds 45 correct predictions, an accuracy increase of 1.62 percentage points.
Macro-F1 increases by 0.1282. This indicates improved performance across classes, but the remaining imbalance is severe.

The model predicts light sleep for 2,592 of 2,780 retained epochs (93.24%).
True light-sleep labels account for 54.46% of those epochs.
It detects 40 of 188 wake epochs, 45 of 466 deep epochs, and 12 of 612 REM epochs.
Most missed REM and deep epochs become light predictions.

The saved optimizer used 75 iterations with a 300-iteration cap. It did not exhaust that cap.
A larger iteration limit alone is not supported as the next improvement.
This observation does not establish a globally optimal model or validate the input alignment.

## What this experiment establishes

The reproducible pipeline can fit training data and evaluate participant-held-out validation data.
The six features have some predictive signal for this linear classifier.
The current representation and model are inadequate for useful four-class staging.
The result does not establish the achievable accuracy of a recurrent model.

Possible limitations include class imbalance, limited features, and absence of temporal context.
The confusion matrix supports the prediction-bias finding. It does not identify which limitation is the dominant cause.
The epoch-start convention remains an assumption from the original classifier.
Final test observations remain reserved.

## Next controlled experiment

Keep this measured run as the reference. Compare class weighting using identical features and splits.
Measure macro-F1 and per-class precision/recall as well as accuracy.
Then compare a small causal GRU using participant-local sequence segments.
This separates the class-weighting effect from the temporal-model effect.
Timo will execute real-data training. No further training is authorized by this report.
