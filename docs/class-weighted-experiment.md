# Class-weighted baseline experiment

Date: 2026-11-02. Timo explicitly authorized this small comparison to run before opening the PR.
Only logistic-regression class weighting changed. The original run remains unchanged.

## Controlled comparison

Both runs use the same 19,928 training and 2,780 retained validation epochs.
There are six sensor features, the same training-only scaler procedure, seed 42, C=1, and a 300-iteration limit.
The final test participants remain unopened. The majority reference is unchanged.
Class weights are computed from training labels only. Validation labels do not determine the weights.
The weighted model completed in 63 iterations and did not exhaust the iteration limit.

| Validation metric | Original logistic regression | Balanced logistic regression |
| --- | ---: | ---: |
| Accuracy | 56.08% | 26.29% |
| Macro-F1 | 0.3045 | 0.2659 |
| Balanced accuracy | 32.36% | 41.52% |
| REM precision | 34.29% | 32.33% |
| REM recall | 1.96% | 38.56% |
| REM F1 | 0.0371 | 0.3517 |
| Deep recall | 9.66% | 85.19% |
| Light recall | 96.57% | 1.39% |

Coverage stays at 2780/2884 scored epochs (96.39%). Results cover three validation participants only.
Balanced accuracy is mean recall across the supported classes. It gives each supported class equal influence.
Overall accuracy weights each epoch equally and therefore reflects class prevalence.

## Interpretation

Weighting substantially improves REM and deep recall, but nearly eliminates correct light predictions.
The weighted model predicts deep sleep for 1,741 of 2,780 epochs (62.63%).
Its lower accuracy and macro-F1 mean it does not replace the unweighted baseline overall.
The higher balanced accuracy shows a different trade-off, not an unqualified improvement.

This comparison demonstrates that class weighting changes the prediction bias.
It does not establish class imbalance as the only cause of weak staging.
The six linear features may not separate stages adequately, and they contain no temporal context.
A small causal GRU is the next architecture comparison. Keep the same features and split first.

## Graphical evidence

Open the [tracked graphical report](experiments/class-weighted-baseline-2026-11-02/README.md).
It contains normalized confusion matrices with counts, per-class recall/F1, coefficient heatmaps, and prediction ribbons.
The coefficient heatmaps show standardized associations, not causal feature importance.
Each ribbon covers retained 30-second epochs. White gaps indicate omitted observations.
The observed lane contains PSG targets, not another model's predictions.

Full models, prediction CSVs, and HTML reports remain under ignored `artifacts/`.
The compact report, input metric hashes, configs, package versions, split, and figures are tracked in Git.
Prediction CSVs are intentionally not committed. They can be regenerated from the saved models.
Source notebooks remain output-free `.ipynb` files.

## Reproduce

Complete preparation from sessions 3 and 4 first.
The original saved baseline is required for the reference figures.
On a fresh checkout without its model files, fit that reference under separate explicit authorization.

```bash
uv sync --locked --group baseline --group notebook
uv run --locked --group baseline python -m sleep_classifier.baseline --config configs/baseline_balanced.yaml --train
uv run --locked --group baseline python -m sleep_classifier.baseline --config configs/baseline.yaml --export-predictions artifacts/baseline/validation_predictions.csv
uv run --locked --group baseline python -m sleep_classifier.experiment_report --reference artifacts/baseline/metrics.json --comparison artifacts/baseline-balanced/metrics.json --output-dir artifacts/comparisons/class-weighted-baseline
uv run --locked --group notebook --group baseline python scripts/execute_notebook.py --notebook notebooks/06_class_weighted_comparison.ipynb
```

Training overwrites the configured weighted output directory; preserve previous runs before repeating it.
Prediction export loads the saved original scaler and models without fitting them.
Report commands read existing results without fitting. The notebook's training cell is tagged to be skipped during report generation.
Use a new versioned report directory when publishing another measured experiment.
