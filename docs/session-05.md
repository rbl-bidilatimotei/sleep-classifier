# Session 5: operator-run baseline workflow

Date: 2026-10-30. No sleep-data model was trained during implementation.

## Experiment purpose

The majority classifier provides a reference score for the most common training class.
Logistic regression tests whether the six sensor summaries contain useful stage information.
It has no recurrent state. Compare the later GRU against this reference using the same participant split and sensor features.
These are preparation tools and an experiment protocol, not measured sleep-classification results.

## Design

`PreparedFeatureDataset` reads provenance-listed training and validation CSVs.
It selects eligible scored rows and the six explicit `FEATURE_COLUMNS`.
It checks finite features, target codes, split separation, participant identifiers, and eligible row counts.
The loader stops if a listed participant has no eligible rows, rather than hiding their missing coverage.
The feature loader never opens test tables. It does not yet provide a PyTorch dataset adapter.

`BaselineExperiment` orchestrates loading, model fitting, validation, and result storage through smaller methods.
The logistic-regression pipeline fits `StandardScaler` on training rows only.
It reuses that scaler for validation. The majority classifier learns the training majority class.

The baseline does not use steps or temporal context. Sensor-history and recurrent-model experiments follow later.
No clinical or deployment accuracy claim is made.

## Run locally

Complete the session 3 data inventory and session 4 feature preparation first.
Inspect `configs/baseline.yaml` before fitting.

```bash
uv sync --locked --group baseline --group notebook
uv run --locked --group baseline python -m sleep_classifier.baseline --config configs/baseline.yaml --train
```

The CLI refuses training unless `--train` is present.
The prepared dataset is not rebuilt automatically by this command.
Artifacts are saved under `artifacts/baseline/`. A new run overwrites the current baseline results there.
Copy results to a separate experiment directory before starting a comparison run.
Alternatively, change the configured output directory before running.

The artifacts include the fitted scaler/model pipeline, majority model, and metrics JSON.
The JSON records feature names, configuration, source preparation provenance, split, and package versions.
Metrics include validation accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, and participant results.
Coverage uses retained scored epochs divided by all scored epochs before filtering.
Model metrics describe retained observations, not rejected epochs.

## Run in Colab

Open `notebooks/05_baseline_experiment.ipynb` in a CPU runtime.
Select `experiment/baseline-workflow` before merge, or `main` after merge.
The setup cell clones the branch and installs uv through Astral when needed.
It runs project commands inside the locked uv environment.
Existing checkouts remain unchanged. Check the printed commit before running an experiment.

Enable setup and preparation for a fresh checkout. Run those sections first.
Enable `RUN_TRAINING` only when ready to fit the models.
The training cell runs the same module command as the local workflow.
Colab execution remains unverified in an actual Colab runtime.

## Report artifacts

Keep source notebooks as `.ipynb` files in Git.
Save executed reports as standalone HTML. The runner executes in memory and exports HTML without a second notebook artifact.
Existing EDA notebook artifacts were converted to HTML where necessary; duplicate executed notebook files were removed.

```bash
uv run --locked --group notebook --group baseline python scripts/execute_notebook.py --notebook notebooks/05_baseline_experiment.ipynb
```

The runner skips the tagged training cell. Report export cannot trigger that cell's model fit.
The default setup and preparation switches are false.
The report reads existing metrics, or states that no baseline results exist.
A source notebook opened interactively still permits the operator to run the training cell.

## Validation

Tests use synthetic feature tables and a tiny synthetic scaler/model fit.
They check partition isolation, explicit features, invalid values, provenance counts, coverage denominators, and train-only scaling.
The HTML runner is tested with synthetic notebooks, including a tagged cell that must not execute.
The report-only baseline notebook is executed with setup, preparation, and training disabled.
No validation accuracy from the real dataset has been generated yet.
