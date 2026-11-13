import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.paired_architecture_report import PairedArchitectureReport


def _write_pair(root: Path) -> tuple[Path, Path]:
    """Write two complete synthetic grouped experiments on matched participants.

    Args:
        root (Path): Parent directory for reference and candidate fixtures.
    """
    plan = {
        "folds": [{"name": "fold-01", "train": ["b"], "validation": ["a"]},
                  {"name": "fold-02", "train": ["a"], "validation": ["b"]}],
        "development": ["a", "b"], "reserved_test": ["untouched"],
        "split_seed": 42, "training_seed": 42, "maximum_epochs": 4,
        "checkpoint_selection": "best macro-F1 on development fold",
        "independent_outer_test": False,
    }
    recipe = {"seed": 42, "epochs": 2, "batch_size": 2, "context_epochs": 2,
              "learning_rate": 0.001, "patience": 1, "gradient_clip": 1, "device": "cpu"}
    paths = []
    for architecture in ("gru", "transformer"):
        directory = root / architecture
        directory.mkdir()
        results = []
        for fold in plan["folds"]:
            person = fold["validation"][0]
            output = directory / fold["name"] / "results"
            output.mkdir(parents=True)
            prediction = [0, 1, 2, 3] if architecture == "transformer" else [1, 1, 1, 1]
            frame = pd.DataFrame({"participant_id": [person] * 4, "time": [0, 30, 60, 90],
                                  "target": [0, 1, 2, 3],
                                  "prediction" if architecture == "transformer"
                                  else "gru_prediction": prediction})
            frame.to_csv(output / "validation_predictions.csv", index=False)
            metrics = {
                "accuracy": float(accuracy_score(frame.target, prediction)),
                "macro_f1": float(f1_score(frame.target, prediction, labels=[0, 1, 2, 3],
                                            average="macro", zero_division=0)),
                "best_epoch": 1, "features": list(FEATURE_COLUMNS), "device": "cpu",
                "participants": {person: {"eligible_epochs": 4}},
                "coverage": {"eligible_epochs": 4, "fraction": 1},
                "config": recipe | {"model_type": architecture},
                "provenance": {"split": fold, "rule": {"name": "moderate"},
                               "feature_columns": list(FEATURE_COLUMNS),
                               "test_observations_read": False, "steps_used": False,
                               "source_inventory": {"version": 1}, "epoch_start_assumed": True},
            }
            metric_path = output / "metrics.json"
            metric_path.write_text(json.dumps(metrics))
            results.append({"fold": fold["name"], "metrics_path": str(metric_path),
                            "metrics_sha256": hashlib.sha256(metric_path.read_bytes()).hexdigest(),
                            **{key: metrics[key] for key in
                               ("accuracy", "macro_f1", "best_epoch", "participants")}})
        path = directory / "summary.json"
        path.write_text(json.dumps({"status": "complete", "plan": plan, "results": results}))
        paths.append(path)

    return paths[0], paths[1]


def _rewrite_metrics(summary_path: Path, changes: dict[str, Any]) -> None:
    """Update a fixture artifact and its summary hash consistently.

    Args:
        summary_path (Path): Candidate summary locating the metrics fixture.
        changes (dict[str, Any]): Top-level metrics values to replace.
    """
    summary = json.loads(summary_path.read_text())
    result = summary["results"][0]
    path = Path(result["metrics_path"])
    metrics = json.loads(path.read_text()) | changes
    path.write_text(json.dumps(metrics))
    result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary))


def test_report_pairs_actual_predictions_and_accepts_both_column_schemas(tmp_path: Path) -> None:
    """Compute descriptive candidate gains on identical held-out epochs.

    Args:
        tmp_path (Path): Temporary experiment directory.
    """
    reference, candidate = _write_pair(tmp_path)
    report = PairedArchitectureReport(reference, candidate).build()

    assert report["eligible_epochs"] == 8
    assert report["validation_participants"] == 2
    assert report["reference"]["accuracy"] == 0.25
    assert report["candidate"]["accuracy"] == 1
    assert report["paired"]["mean_macro_f1_difference"] == pytest.approx(0.9)
    assert report["paired"]["sample_std_macro_f1_difference"] == 0
    assert report["candidate"]["classes"]["3"]["recall"] == 1
    assert report["independent_outer_test"] is False


def test_report_rejects_modified_metrics(tmp_path: Path) -> None:
    """Reject metrics whose contents no longer match the recorded hash.

    Args:
        tmp_path (Path): Temporary experiment directory.
    """
    reference, candidate = _write_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    path = Path(summary["results"][0]["metrics_path"])
    path.write_text(path.read_text() + "\n")

    with pytest.raises(ValueError, match="metrics changed"):
        PairedArchitectureReport(reference, candidate).build()


@pytest.mark.parametrize("mutation", ["participants", "seed", "test", "incomplete"])
def test_report_rejects_unmatched_plans(tmp_path: Path, mutation: str) -> None:
    """Reject unmatched validation membership, seeds, or incomplete results.

    Args:
        tmp_path (Path): Temporary experiment directory.
        mutation (str): Plan field or completion state to invalidate.
    """
    reference, candidate = _write_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    if mutation == "participants":
        summary["plan"]["folds"][0]["validation"] = ["different"]
    elif mutation == "seed":
        summary["plan"]["training_seed"] = 43
    elif mutation == "test":
        summary["plan"]["reserved_test"] = []
    else:
        summary["status"] = "running"
    candidate.write_text(json.dumps(summary))

    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate).build()


@pytest.mark.parametrize("mutation", ["time", "target", "duplicate", "missing", "nonfinite"])
def test_report_rejects_mismatched_epoch_populations(tmp_path: Path, mutation: str) -> None:
    """Reject shifted, relabeled, repeated, or invalid prediction rows.

    Args:
        tmp_path (Path): Temporary experiment directory.
        mutation (str): Prediction change that breaks the paired population.
    """
    reference, candidate = _write_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    path = Path(summary["results"][0]["metrics_path"]).parent / "validation_predictions.csv"
    frame = pd.read_csv(path)
    if mutation == "time":
        frame.loc[0, "time"] = 1
    elif mutation == "target":
        frame.loc[0, "target"] = 1
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "missing":
        frame.loc[0, "prediction"] = float("nan")
    else:
        frame["time"] = frame["time"].astype(float)
        frame.loc[0, "time"] = float("inf")
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate).build()


@pytest.mark.parametrize("key,value", [("device", "mps"), ("context_epochs", 10)])
def test_report_rejects_changed_shared_recipe(tmp_path: Path, key: str, value: Any) -> None:
    """Reject device or training changes mixed into the architecture comparison.

    Args:
        tmp_path (Path): Temporary experiment directory.
        key (str): Shared recipe field to change.
        value (Any): Candidate override that differs from the reference.
    """
    reference, candidate = _write_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    metrics = json.loads(Path(summary["results"][0]["metrics_path"]).read_text())
    _rewrite_metrics(candidate, {"config": metrics["config"] | {key: value}})

    with pytest.raises(ValueError, match="recipe differs"):
        PairedArchitectureReport(reference, candidate).build()


def test_notebook_marks_transformer_pending_when_only_gru_is_complete(tmp_path: Path) -> None:
    """Keep an existing reference distinct from an unstarted candidate.

    Args:
        tmp_path (Path): Isolated notebook report inputs.
    """
    from unittest.mock import Mock

    nbformat = pytest.importorskip("nbformat")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "transformer-grouped-validation.yaml").write_text(
        "output_dir: ../artifacts/candidate\n"
    )
    reference = tmp_path / "artifacts/gru-grouped-validation"
    reference.mkdir(parents=True)
    (reference / "summary.json").write_text(json.dumps({
        "status": "complete", "results": [{}] * 5,
        "aggregate": {"pooled_macro_f1": 0.49, "pooled_accuracy": 0.59},
    }))
    source = Path(__file__).resolve().parents[1] / "notebooks/12_transformer_comparison.ipynb"
    notebook = nbformat.read(source, as_version=4)
    cell = next(cell for cell in notebook.cells
                if cell.cell_type == "code" and "statuses = []" in cell.source)
    namespace = {"ROOT": tmp_path, "json": json, "logger": Mock(), "display": Mock()}
    exec(cell.source, namespace)
    assert namespace["statuses"][0]["status"] == "complete"
    assert namespace["statuses"][1]["status"] == "not_started"
    assert namespace["comparison"] is None
    assert not (tmp_path / "artifacts/transformer-comparison").exists()


def _write_budget_pair(root: Path) -> tuple[Path, Path]:
    """Write matched Transformer runs with different training budgets.

    Args:
        root (Path): Parent directory for synthetic experiment files.
    """
    reference, candidate = _write_pair(root)
    for summary_path, epochs, patience in ((reference, 2, 1), (candidate, 4, 2)):
        summary = json.loads(summary_path.read_text())
        summary["plan"]["maximum_epochs"] = 2 * epochs
        for result in summary["results"]:
            path = Path(result["metrics_path"])
            metrics = json.loads(path.read_text())
            metrics["config"].update({"model_type": "transformer", "epochs": epochs,
                                      "patience": patience, "transformer": {"d_model": 16}})
            metrics["model_config"] = {"d_model": 16}
            metrics["parameter_count"] = 2564
            path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary))

    return reference, candidate


def test_budget_report_requires_explicit_policy(tmp_path: Path) -> None:
    """Keep the default architecture policy strict when training budgets differ.

    Args:
        tmp_path (Path): Temporary synthetic artifact directory.
    """
    reference, candidate = _write_budget_pair(tmp_path)
    with pytest.raises(ValueError, match="maximum_epochs"):
        PairedArchitectureReport(reference, candidate).build()

    report = PairedArchitectureReport(reference, candidate, "training_budget").build()
    assert report["eligible_epochs"] == 8
    assert report["paired"]["mean_macro_f1_difference"] == pytest.approx(0.9)
    assert report["comparison"] == "training_budget"
    assert "selection" in report["selection_caveat"]


@pytest.mark.parametrize("mutation", [
    "width", "learning_rate", "model_config", "parameter_count", "missing_model",
    "epoch_cap", "invalid_patience", "reduced_patience", "unchanged_budget",
    "participant", "target", "metric_hash",
])
def test_budget_report_rejects_changes_outside_the_budget(tmp_path: Path, mutation: str) -> None:
    """Reject confounded recipes, invalid budgets, and damaged paired artifacts.

    Args:
        tmp_path (Path): Temporary synthetic artifact directory.
        mutation (str): Invalid change to apply before reporting.
    """
    reference, candidate = _write_budget_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    metrics_path = Path(summary["results"][0]["metrics_path"])
    metrics = json.loads(metrics_path.read_text())

    if mutation == "epoch_cap":
        summary["plan"]["maximum_epochs"] = 999
        candidate.write_text(json.dumps(summary))
    elif mutation == "participant":
        summary["plan"]["folds"][0]["validation"] = ["other"]
        candidate.write_text(json.dumps(summary))
    elif mutation == "target":
        path = metrics_path.parent / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "target"] = 1
        frame.to_csv(path, index=False)
    elif mutation == "metric_hash":
        metrics_path.write_text(metrics_path.read_text() + "\n")
    else:
        if mutation == "width":
            metrics["config"]["transformer"]["d_model"] = 32
        elif mutation == "learning_rate":
            metrics["config"]["learning_rate"] = 0.01
        elif mutation == "model_config":
            metrics["model_config"] = {"d_model": 32}
        elif mutation == "parameter_count":
            metrics["parameter_count"] = 9220
        elif mutation == "missing_model":
            metrics["model_config"] = None
        elif mutation == "invalid_patience":
            metrics["config"]["patience"] = True
        elif mutation == "reduced_patience":
            metrics["config"]["patience"] = 0
        elif mutation == "unchanged_budget":
            metrics["config"].update({"epochs": 2, "patience": 1})
        _rewrite_metrics(candidate, metrics)

    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "training_budget").build()


def test_budget_notebook_reports_completed_runs_and_training_histories(tmp_path: Path) -> None:
    """Exercise the completed notebook report without launching a trainer.

    Args:
        tmp_path (Path): Temporary synthetic notebook inputs and outputs.
    """
    from unittest.mock import Mock, patch

    nbformat = pytest.importorskip("nbformat")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    reference, candidate = _write_budget_pair(artifacts)
    for path in (reference, candidate):
        summary = json.loads(path.read_text())
        for result in summary["results"]:
            metrics_path = Path(result["metrics_path"])
            metrics = json.loads(metrics_path.read_text())
            metrics["history"] = [{"epoch": 1, "validation_macro_f1": metrics["macro_f1"]}]
            metrics_path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(summary))

    expected_reference = artifacts / "transformer-grouped-validation"
    expected_reference.mkdir()
    (expected_reference / "summary.json").write_text(reference.read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "transformer-budget-grouped-validation.yaml").write_text(
        "output_dir: ../artifacts/transformer\n"
    )
    source = (Path(__file__).resolve().parents[1]
              / "notebooks/14_transformer_training_budget.ipynb")
    notebook = nbformat.read(source, as_version=4)
    namespace = {"ROOT": tmp_path, "Path": Path, "json": json,
                 "logger": Mock(), "display": Mock()}
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code" and (
                "statuses = []" in cell.source or "figure, axes = plt.subplots" in cell.source
            ):
                with patch.object(plt, "show"):
                    exec(cell.source, namespace)

        assert namespace["comparison"]["comparison"] == "training_budget"
        assert len(namespace["history_rows"]) == 4
        output = artifacts / "transformer-budget-comparison/summary.json"
        assert json.loads(output.read_text())["eligible_epochs"] == 8
    finally:
        plt.close("all")


def _write_seed_study(root: Path) -> Path:
    """Write two synthetic shared-seed budget pairs and their study configuration.

    Args:
        root (Path): Temporary study directory.
    """
    import yaml

    pairs = []
    for seed in (42, 43):
        directory = root / f"seed-{seed}"
        directory.mkdir()
        reference, candidate = _write_budget_pair(directory)
        pair = {"seed": seed, "reuse": seed == 42}
        for role, path in (("reference", reference), ("candidate", candidate)):
            summary = json.loads(path.read_text())
            summary["plan"]["training_seed"] = seed
            for result in summary["results"]:
                metric_path = Path(result["metrics_path"])
                metrics = json.loads(metric_path.read_text())
                metrics["config"]["seed"] = seed
                metric_path.write_text(json.dumps(metrics))
                result["metrics_sha256"] = hashlib.sha256(metric_path.read_bytes()).hexdigest()
            path.write_text(json.dumps(summary))
            model_path = directory / f"{role}-model.yaml"
            model_path.write_text(yaml.safe_dump(metrics["config"]))
            grouped_path = directory / f"{role}-grouped.yaml"
            grouped_path.write_text(yaml.safe_dump({
                "base_model_config": str(model_path), "output_dir": str(path.parent), "folds": 2,
            }))
            pair[f"{role}_config"] = str(grouped_path)
        pairs.append(pair)
    config = root / "study.yaml"
    config.write_text(yaml.safe_dump({"pairs": pairs, "output_dir": "comparison"}))

    return config


def test_seed_study_reuses_existing_seed_and_summarizes_paired_effects(tmp_path: Path) -> None:
    """Aggregate seed effects without treating repeated epochs as new observations.

    Args:
        tmp_path (Path): Temporary synthetic study directory.
    """
    from sleep_classifier.budget_seed_report import BudgetSeedReport

    study = BudgetSeedReport(_write_seed_study(tmp_path))
    assert study.plan()["new_fits"] == 4
    assert study.plan()["maximum_new_epochs"] == 12
    assert all(row["status"] == "complete" for row in study.status())
    result = study.build()
    assert result["mean_seed_macro_f1_difference"] == pytest.approx(0.9)
    assert result["sample_std_seed_macro_f1_difference"] == 0
    assert [entry["seed"] for entry in result["seeds"]] == [42, 43]
    assert not study.output_dir.exists()


@pytest.mark.parametrize(
    "mutation", ["duplicate", "wrong_seed", "folds", "recipe", "model", "missing"],
)
def test_seed_study_rejects_unmatched_or_missing_results(tmp_path: Path, mutation: str) -> None:
    """Reject duplicate seeds, confounded recipes, changed folds, and incomplete pairs.

    Args:
        tmp_path (Path): Temporary synthetic study directory.
        mutation (str): Invalid study or recorded artifact change.
    """
    import yaml

    from sleep_classifier.budget_seed_report import BudgetSeedReport

    config = _write_seed_study(tmp_path)
    settings = yaml.safe_load(config.read_text())
    if mutation == "duplicate":
        settings["pairs"][1]["seed"] = 42
        config.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError, match="unique"):
            BudgetSeedReport(config)
        return

    study = BudgetSeedReport(config)
    case = study.cases[1]
    if mutation == "missing":
        Path(case["candidate_summary"]).unlink()
        assert study.status()[-1]["status"] == "not_started"
        with pytest.raises(FileNotFoundError):
            study.build()
        return

    for role in ("reference", "candidate"):
        path = Path(case[f"{role}_summary"])
        summary = json.loads(path.read_text())
        if mutation == "wrong_seed":
            summary["plan"]["training_seed"] = 999
        elif mutation == "folds":
            summary["plan"]["split_seed"] = 999
        elif mutation in ("recipe", "model"):
            for result in summary["results"]:
                metric_path = Path(result["metrics_path"])
                metrics = json.loads(metric_path.read_text())
                if mutation == "recipe":
                    metrics["config"]["learning_rate"] = 0.01
                else:
                    metrics["model_config"] = {"d_model": 32}
                    metrics["parameter_count"] = 9220
                metric_path.write_text(json.dumps(metrics))
                result["metrics_sha256"] = hashlib.sha256(metric_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="seed|recipe|Folds"):
        study.build()


def test_seed_notebook_reports_completed_pairs_without_training(tmp_path: Path) -> None:
    """Execute completed seed reporting and plot cells using synthetic saved predictions.

    Args:
        tmp_path (Path): Temporary synthetic study directory.
    """
    from unittest.mock import Mock, patch

    from sleep_classifier.budget_seed_report import BudgetSeedReport

    nbformat = pytest.importorskip("nbformat")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")
    study = BudgetSeedReport(_write_seed_study(tmp_path))
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/15_transformer_budget_seeds.ipynb",
        as_version=4,
    )
    namespace = {"study": study, "Path": Path, "json": json, "pd": pd,
                 "logger": Mock(), "display": Mock(),
                 "PairedArchitectureReport": PairedArchitectureReport}
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code" and (
                "statuses = study.status()" in cell.source
                or cell.source.startswith('if all(row["status"]')
                or cell.source.startswith("if seed_rows:")
            ):
                with patch.object(plt, "show"):
                    exec(cell.source, namespace)

        assert len(namespace["seed_rows"]) == 2
        report = json.loads((study.output_dir / "summary.json").read_text())
        assert len(report["seeds"]) == 2
        assert report["mean_seed_macro_f1_difference"] == pytest.approx(0.9)
    finally:
        plt.close("all")


def _write_context_pair(root: Path) -> tuple[Path, Path]:
    """Write matched fixed-budget Transformers with different position tables.

    Args:
        root (Path): Parent directory for synthetic experiment artifacts.
    """
    reference, candidate = _write_budget_pair(root)
    for summary_path, context in ((reference, 2), (candidate, 6)):
        summary = json.loads(summary_path.read_text())
        summary["plan"]["maximum_epochs"] = 8
        for result in summary["results"]:
            path = Path(result["metrics_path"])
            metrics = json.loads(path.read_text())
            metrics["config"].update({"epochs": 4, "patience": 2, "context_epochs": context,
                                      "transformer": {"d_model": 16, "max_context": context}})
            metrics["model_config"] = {"d_model": 16, "max_context": context}
            metrics["parameter_count"] = 2564 + (context - 2) * 16
            metrics["history"] = [{"epoch": 1, "validation_macro_f1": metrics["macro_f1"]}]
            path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary))

    return reference, candidate


def test_context_report_permits_only_longer_history_and_position_capacity(tmp_path: Path) -> None:
    """Compare longer context while preserving the held-out prediction population.

    Args:
        tmp_path (Path): Temporary synthetic experiment directory.
    """
    reference, candidate = _write_context_pair(tmp_path)
    for policy in ("architecture", "training_budget"):
        with pytest.raises(ValueError):
            PairedArchitectureReport(reference, candidate, policy).build()

    report = PairedArchitectureReport(reference, candidate, "context").build()
    assert report["comparison"] == "context"
    assert "positional" in report["capacity_caveat"]
    assert report["eligible_epochs"] == 8
    assert report["paired"]["mean_macro_f1_difference"] == pytest.approx(0.9)


@pytest.mark.parametrize("mutation", [
    "width", "dropout", "epochs", "patience", "learning_rate", "position_capacity",
    "parameter_count", "model_metadata", "unchanged_context", "shorter_context",
    "boolean_context", "coverage", "metric_hash", "target", "participant", "epoch_cap",
])
def test_context_report_rejects_confounded_or_invalid_pairs(tmp_path: Path, mutation: str) -> None:
    """Reject recipe changes and damaged evaluation inputs in context mode.

    Args:
        tmp_path (Path): Temporary synthetic experiment directory.
        mutation (str): Candidate metadata or prediction corruption to apply.
    """
    reference, candidate = _write_context_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    path = Path(summary["results"][0]["metrics_path"])
    metrics = json.loads(path.read_text())
    if mutation == "metric_hash":
        path.write_text(path.read_text() + "\n")
    elif mutation == "target":
        predictions = path.parent / "validation_predictions.csv"
        frame = pd.read_csv(predictions)
        frame.loc[0, "target"] = 1
        frame.to_csv(predictions, index=False)
    elif mutation in ("participant", "epoch_cap"):
        if mutation == "participant":
            summary["plan"]["folds"][0]["validation"] = ["other"]
        else:
            summary["plan"]["maximum_epochs"] = 999
        candidate.write_text(json.dumps(summary))
    else:
        if mutation in ("width", "dropout"):
            metrics["config"]["transformer"]["d_model" if mutation == "width" else "dropout"] = 32
        elif mutation in ("epochs", "patience", "learning_rate"):
            metrics["config"][mutation] = 100
        elif mutation == "position_capacity":
            metrics["config"]["transformer"]["max_context"] = 100
        elif mutation == "parameter_count":
            metrics["parameter_count"] += 1
        elif mutation == "model_metadata":
            metrics["model_config"]["d_model"] = 32
        elif mutation in ("unchanged_context", "shorter_context", "boolean_context"):
            values = {"unchanged_context": 2, "shorter_context": 1, "boolean_context": True}
            value = values[mutation]
            metrics["config"]["context_epochs"] = value
            metrics["config"]["transformer"]["max_context"] = value
            metrics["model_config"]["max_context"] = value
        elif mutation == "coverage":
            metrics["coverage"]["fraction"] = 0.5
        _rewrite_metrics(candidate, metrics)

    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "context").build()


def test_context_notebook_reports_completed_runs_without_optimization(tmp_path: Path) -> None:
    """Exercise completed context comparisons and plot cells with synthetic artifacts.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
    """
    from unittest.mock import Mock, patch

    nbformat = pytest.importorskip("nbformat")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    reference, candidate = _write_context_pair(artifacts)
    expected_reference = artifacts / "transformer-budget-grouped-validation"
    expected_reference.mkdir()
    (expected_reference / "summary.json").write_text(reference.read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "transformer-context-30-grouped-validation.yaml").write_text(
        "output_dir: ../artifacts/transformer\n"
    )
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/16_transformer_context.ipynb",
        as_version=4,
    )
    frame = pd.DataFrame({"participant_id": ["a"] * 4 + ["b"] * 4,
                          "time": [0, 30, 60, 90] * 2, "target": [0, 1, 2, 3] * 2})
    namespace = {"ROOT": tmp_path, "Path": Path, "json": json,
                 "frame": frame, "available": pd.Series([1, 2, 3, 4] * 2),
                 "logger": Mock(), "display": Mock()}
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code" and (
                "statuses = []" in cell.source or "figure, axes = plt.subplots" in cell.source
                or "history_strata = []" in cell.source
            ):
                with patch.object(plt, "show"):
                    exec(cell.source, namespace)

        assert namespace["comparison"]["comparison"] == "context"
        assert len(namespace["history_rows"]) == 4
        assert len(namespace["history_strata"]) == 6
        assert namespace["history_strata"][0]["epochs"] == 8
        assert namespace["history_strata"][1]["macro_f1"] is None
        output = artifacts / "transformer-context-comparison/summary.json"
        assert json.loads(output.read_text())["eligible_epochs"] == 8
    finally:
        plt.close("all")


def _write_context_seed_study(root: Path) -> Path:
    """Write context pairs with completed references and one planned new candidate.

    Args:
        root (Path): Temporary directory for synthetic grouped artifacts.
    """
    import yaml

    config = _write_seed_study(root)
    settings = yaml.safe_load(config.read_text())
    settings["comparison"] = "context"
    for pair in settings["pairs"]:
        pair.pop("reuse")
        pair.update(reuse_reference=True, reuse_candidate=pair["seed"] == 42)
        for role, context in (("reference", 2), ("candidate", 6)):
            grouped_path = Path(pair[f"{role}_config"])
            grouped = yaml.safe_load(grouped_path.read_text())
            summary_path = Path(grouped["output_dir"]) / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["plan"]["maximum_epochs"] = 8
            for result in summary["results"]:
                path = Path(result["metrics_path"])
                metrics = json.loads(path.read_text())
                metrics["config"].update({"epochs": 4, "patience": 2, "context_epochs": context,
                                          "transformer": {"d_model": 16, "max_context": context}})
                metrics["model_config"] = {"d_model": 16, "max_context": context}
                metrics["parameter_count"] = 2564 + (context - 2) * 16
                path.write_text(json.dumps(metrics))
                result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                predictions_path = path.parent / "validation_predictions.csv"
                predictions = pd.read_csv(predictions_path)
                column = "prediction" if "prediction" in predictions else "gru_prediction"
                for stage in range(4):
                    predictions[f"probability_{stage}"] = predictions[column].map(
                        {label: 0.7 if label == stage else 0.1 for label in range(4)}
                    )
                predictions.to_csv(predictions_path, index=False)
            summary_path.write_text(json.dumps(summary))
            Path(grouped["base_model_config"]).write_text(yaml.safe_dump(metrics["config"]))
    config.write_text(yaml.safe_dump(settings))

    return config


def test_context_seed_study_plans_only_new_candidates_and_preserves_budget_alias(
    tmp_path: Path,
) -> None:
    """Exclude all completed references while retaining the old budget-report API.

    Args:
        tmp_path (Path): Temporary synthetic seed-study directory.
    """
    from sleep_classifier.budget_seed_report import BudgetSeedReport
    from sleep_classifier.paired_seed_report import PairedSeedReport

    assert BudgetSeedReport is PairedSeedReport
    study = PairedSeedReport(_write_context_seed_study(tmp_path))
    assert [(run["seed"], run["recipe"]) for run in study.new_runs()] == [(43, "candidate")]
    assert study.plan()["new_fits"] == 2
    assert study.plan()["maximum_new_epochs"] == 8
    study.validate_reused_results()
    report = study.build()
    assert report["comparison"] == "context"
    assert report["mean_seed_macro_f1_difference"] == pytest.approx(0.9)
    assert all(entry["report"]["comparison"] == "context" for entry in report["seeds"])
    assert not study.output_dir.exists()


@pytest.mark.parametrize("mutation", ["missing", "hash", "seed", "flag", "comparison"])
def test_context_reuse_validation_rejects_unusable_references(
    tmp_path: Path, mutation: str,
) -> None:
    """Block new work when reused inputs or reuse declarations cannot be trusted.

    Args:
        tmp_path (Path): Temporary synthetic seed-study directory.
        mutation (str): Invalid reused artifact or declaration.
    """
    import yaml

    from sleep_classifier.paired_seed_report import PairedSeedReport

    config = _write_context_seed_study(tmp_path)
    settings = yaml.safe_load(config.read_text())
    if mutation in ("flag", "comparison"):
        if mutation == "flag":
            settings["pairs"][1]["reuse_reference"] = "true"
        else:
            settings["comparison"] = "unknown"
        config.write_text(yaml.safe_dump(settings))
        with pytest.raises(ValueError):
            PairedSeedReport(config)
        return

    study = PairedSeedReport(config)
    summary_path = Path(study.cases[1]["reference_summary"])
    summary = json.loads(summary_path.read_text())
    if mutation == "missing":
        summary_path.unlink()
    elif mutation == "hash":
        path = Path(summary["results"][0]["metrics_path"])
        path.write_text(path.read_text() + "\n")
    else:
        summary["plan"]["training_seed"] = 999
        summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        study.validate_reused_results()


@pytest.mark.parametrize("study_kind,notebook_name", [
    ("context", "17_transformer_context_seeds.ipynb"),
    ("architecture", "19_transformer_dropout_seeds.ipynb"),
])
def test_shared_seed_notebooks_report_completed_pairs_without_training(
    tmp_path: Path, study_kind: str, notebook_name: str,
) -> None:
    """Exercise context and dropout replication reports from synthetic saved inputs.

    Args:
        tmp_path (Path): Temporary synthetic study directory.
        study_kind (str): Explicit paired comparison policy.
        notebook_name (str): Source experiment notebook to execute.
    """
    from unittest.mock import Mock, patch

    from sleep_classifier.paired_seed_report import PairedSeedReport

    nbformat = pytest.importorskip("nbformat")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")
    config = (_write_context_seed_study(tmp_path) if study_kind == "context"
              else _write_dropout_seed_study(tmp_path))
    study = PairedSeedReport(config)
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks" / notebook_name,
        as_version=4,
    )
    namespace = {"study": study, "Path": Path, "json": json, "pd": pd,
                 "logger": Mock(), "display": Mock(),
                 "PairedArchitectureReport": PairedArchitectureReport}
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code" and (
                "statuses = study.status()" in cell.source
                or cell.source.startswith('if all(row["status"]')
                or cell.source.startswith("if seed_rows:")
                or cell.source.startswith("if class_rows:")
                or "probability_diagnostics = []" in cell.source
            ):
                with patch.object(plt, "show"):
                    exec(cell.source, namespace)

        assert len(namespace["seed_rows"]) == 2
        if study_kind == "architecture":
            audit = json.loads((study.output_dir / "epoch-audit.json").read_text())
            assert len(audit) == 4
            assert all(row["minimum_nll_epoch"] == 2 for row in audit)
            assert all(row["largest_upward_step_epoch"] == 3 for row in audit)
            assert all(row["best_macro_f1_epoch"] == 1 for row in audit)
        assert len(namespace["probability_diagnostics"]) == 4
        candidate = namespace["probability_diagnostics"][1]
        assert candidate["checkpoint_nll"] == pytest.approx(0.3566749439)
        assert candidate["mean_confidence_wrong"] is None
        result = json.loads((study.output_dir / "summary.json").read_text())
        assert result["comparison"] == study_kind
        assert result["mean_seed_macro_f1_difference"] == pytest.approx(0.9)
        assert [(run["seed"], run["recipe"]) for run in study.new_runs()] == [(43, "candidate")]
    finally:
        plt.close("all")


def test_dropout_notebook_reports_fixed_context_recipes_and_checkpoint_nll(tmp_path: Path) -> None:
    """Report a dropout recipe comparison and selected checkpoint loss without training.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
    """
    from unittest.mock import Mock, patch

    nbformat = pytest.importorskip("nbformat")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    reference, candidate = _write_context_pair(artifacts)
    for summary_path, dropout in ((reference, 0.1), (candidate, 0.2)):
        saved = json.loads(summary_path.read_text())
        for result in saved["results"]:
            path = Path(result["metrics_path"])
            metrics = json.loads(path.read_text())
            metrics["config"].update(context_epochs=6, transformer={
                "d_model": 16, "max_context": 6, "dropout": dropout,
            })
            metrics["model_config"] = {"d_model": 16, "max_context": 6, "dropout": dropout}
            metrics["parameter_count"] = 2628
            metrics["history"][0]["validation_loss"] = 1.0
            path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(saved))
    expected_reference = artifacts / "transformer-context-30-grouped-validation"
    expected_reference.mkdir()
    (expected_reference / "summary.json").write_text(reference.read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "transformer-dropout-grouped-validation.yaml").write_text(
        "output_dir: ../artifacts/transformer\n"
    )
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/18_transformer_dropout.ipynb",
        as_version=4,
    )
    namespace = {"ROOT": tmp_path, "Path": Path, "json": json,
                 "logger": Mock(), "display": Mock()}
    try:
        for cell in notebook.cells:
            if cell.cell_type == "code" and (
                "statuses = []" in cell.source or "figure, axes = plt.subplots" in cell.source
            ):
                with patch.object(plt, "show"):
                    exec(cell.source, namespace)
        assert namespace["comparison"]["eligible_epochs"] == 8
        assert namespace["comparison"]["paired"]["mean_macro_f1_difference"] == pytest.approx(0.9)
        assert len(namespace["history_rows"]) == 4
        assert [row["pooled_selected_nll"] for row in namespace["pooled_losses"]] == [1.0, 1.0]
        assert all(row["selected_validation_nll"] == 1.0 for row in namespace["history_rows"])
    finally:
        plt.close("all")


def _write_dropout_seed_study(root: Path) -> Path:
    """Write matched dropout recipes with fixed context and per-recipe reuse flags.

    Args:
        root (Path): Temporary directory for synthetic study artifacts.
    """
    import yaml

    config = _write_context_seed_study(root)
    settings = yaml.safe_load(config.read_text())
    settings["comparison"] = "architecture"
    for pair in settings["pairs"]:
        for role, dropout in (("reference", 0.1), ("candidate", 0.2)):
            grouped_path = Path(pair[f"{role}_config"])
            grouped = yaml.safe_load(grouped_path.read_text())
            summary_path = Path(grouped["output_dir"]) / "summary.json"
            saved = json.loads(summary_path.read_text())
            for result in saved["results"]:
                path = Path(result["metrics_path"])
                metrics = json.loads(path.read_text())
                metrics["config"].update(context_epochs=6, transformer={
                    "d_model": 16, "max_context": 6, "dropout": dropout,
                })
                metrics["model_config"] = {"d_model": 16, "max_context": 6, "dropout": dropout}
                metrics["parameter_count"] = 2628
                metrics["history"] = [
                    {"epoch": epoch, "validation_loss": loss,
                     "validation_macro_f1": metrics["macro_f1"]}
                    for epoch, loss in [(1, 1.0), (2, 0.7), (3, 0.9)]
                ]
                path.write_text(json.dumps(metrics))
                result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary_path.write_text(json.dumps(saved))
            Path(grouped["base_model_config"]).write_text(yaml.safe_dump(metrics["config"]))
    config.write_text(yaml.safe_dump(settings))

    return config


def _write_learning_rate_pair(root: Path) -> tuple[Path, Path]:
    """Write matched instrumented recipes differing only in their constant learning rates.

    Args:
        root (Path): Temporary directory for two synthetic grouped experiments.
    """
    paths = _write_pair(root)
    for summary_path, rate in zip(paths, (0.001, 0.0003), strict=True):
        summary = json.loads(summary_path.read_text())
        summary["plan"]["validation_scope"] = "selected development folds"
        for result in summary["results"]:
            path = Path(result["metrics_path"])
            metrics = json.loads(path.read_text())
            metrics["config"].update(
                model_type="transformer", learning_rate=rate,
                transformer={"d_model": 16, "max_context": 2, "dropout": 0.2},
                diagnostics={"enabled": True, "save_predictions": True},
                output_dir=str(path.parent), prepare_config="prepare.yaml",
            )
            metrics["model_config"] = {"d_model": 16, "max_context": 2, "dropout": 0.2}
            metrics["parameter_count"] = 2564
            path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary))
    return paths


def test_learning_rate_report_requires_explicit_policy_and_preserves_subset_caveat(
    tmp_path: Path,
) -> None:
    """Accept the isolated rate change and label the post-hoc diagnostic population.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
    """
    reference, candidate = _write_learning_rate_pair(tmp_path)
    with pytest.raises(ValueError, match="learning_rate"):
        PairedArchitectureReport(reference, candidate).build()
    report = PairedArchitectureReport(reference, candidate, "learning_rate").build()

    assert report["comparison"] == "learning_rate"
    assert report["validation_scope"] == "selected development folds"
    assert "not a full-cohort benchmark" in report["interpretation"]
    assert "asymptotic" in report["optimization_caveat"]
    assert report["paired"]["mean_macro_f1_difference"] == pytest.approx(0.9)
    assert report["eligible_epochs"] == 8
    assert not report["independent_outer_test"]


@pytest.mark.parametrize("mutation", [
    "dropout", "context", "patience", "seed", "diagnostics", "unknown_setting",
    "model_metadata", "parameter_count", "device", "population", "budget", "scope",
])
def test_learning_rate_report_rejects_confounded_recipes_or_populations(
    tmp_path: Path, mutation: str,
) -> None:
    """Reject simultaneous recipe changes, invalid model metadata, and unmatched targets.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
        mutation (str): Controlled inconsistency to introduce after creating a valid pair.
    """
    reference, candidate = _write_learning_rate_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    if mutation in {"budget", "scope"}:
        summary["plan"]["maximum_epochs" if mutation == "budget" else "validation_scope"] = (
            999 if mutation == "budget" else "all development folds"
        )
        candidate.write_text(json.dumps(summary))
    elif mutation == "population":
        path = Path(summary["results"][0]["metrics_path"]).parent / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "target"] = 1
        frame.to_csv(path, index=False)
    else:
        path = Path(summary["results"][0]["metrics_path"])
        metrics = json.loads(path.read_text())
        if mutation == "dropout":
            metrics["config"]["transformer"]["dropout"] = 0.3
        elif mutation == "context":
            metrics["config"]["context_epochs"] = 3
        elif mutation in {"patience", "seed"}:
            metrics["config"][mutation] += 1
        elif mutation == "diagnostics":
            metrics["config"]["diagnostics"]["save_predictions"] = False
        elif mutation == "unknown_setting":
            metrics["config"]["loss"] = "weighted_cross_entropy"
        elif mutation == "model_metadata":
            metrics["model_config"]["d_model"] = 32
        elif mutation == "parameter_count":
            metrics["parameter_count"] += 1
        elif mutation == "device":
            metrics["device"] = "mps"
        _rewrite_metrics(candidate, metrics)
    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "learning_rate").build()


@pytest.mark.parametrize("rate", [0.001, 0.0, -0.1, True, float("nan"), float("inf"), None])
def test_learning_rate_report_rejects_equal_or_invalid_learning_rates(
    tmp_path: Path, rate: float | bool | None,
) -> None:
    """Require a distinct positive finite numeric rate for the candidate recipe.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
        rate (float | bool | None): Invalid or unchanged candidate learning rate.
    """
    reference, candidate = _write_learning_rate_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    path = Path(summary["results"][0]["metrics_path"])
    metrics = json.loads(path.read_text())
    metrics["config"]["learning_rate"] = rate
    _rewrite_metrics(candidate, metrics)
    with pytest.raises(ValueError, match="learning rate|Learning rate"):
        PairedArchitectureReport(reference, candidate, "learning_rate").build()


def test_learning_rate_report_checks_fold_cap_even_when_both_plans_agree(tmp_path: Path) -> None:
    """Reject equally corrupted grouped budgets that disagree with the saved fold cap.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
    """
    reference, candidate = _write_learning_rate_pair(tmp_path)
    for path in (reference, candidate):
        summary = json.loads(path.read_text())
        summary["plan"]["maximum_epochs"] = 6
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="epoch budget"):
        PairedArchitectureReport(reference, candidate, "learning_rate").build()


def test_learning_rate_report_rejects_equal_but_invalid_parameter_count(tmp_path: Path) -> None:
    """Reject floating-point parameter metadata even when its numeric value matches.

    Args:
        tmp_path (Path): Temporary synthetic grouped experiment directory.
    """
    reference, candidate = _write_learning_rate_pair(tmp_path)
    _rewrite_metrics(candidate, {"parameter_count": 2564.0})
    with pytest.raises(ValueError, match="valid model metadata"):
        PairedArchitectureReport(reference, candidate, "learning_rate").build()


def test_learning_rate_notebook_reports_complete_pair_without_training(tmp_path: Path) -> None:
    """Exercise the populated notebook comparison against verified synthetic saved scores.

    Args:
        tmp_path (Path): Temporary paired metric and prediction artifacts.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    reference, candidate = _write_learning_rate_pair(tmp_path)
    notebook = json.loads((Path(__file__).resolve().parents[1]
                           / "notebooks/21_transformer_learning_rate.ipynb").read_text())
    namespace = {"experiment": SimpleNamespace(output_dir=candidate.parent),
                 "reference_summary": reference,
                 "PairedArchitectureReport": PairedArchitectureReport,
                 "pd": pd, "json": json, "logger": Mock(), "display": Mock()}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("comparison = None"):
            exec(source, namespace)
    report = json.loads((candidate.parent / "paired-comparison.json").read_text())
    assert report["comparison"] == "learning_rate"
    assert report["eligible_epochs"] == 8
    assert report["validation_scope"] == "selected development folds"
    assert len(namespace["class_rows"]) == 8
    assert namespace["display"].call_count == 3


def _write_scheduler_pair(root: Path) -> tuple[Path, Path]:
    """Write fixed base-rate recipes with a plateau scheduler added only to the candidate.

    Args:
        root (Path): Temporary directory for two synthetic grouped experiments.
    """
    pytest.importorskip("torch")
    reference, candidate = _write_learning_rate_pair(root)
    summary = json.loads(candidate.read_text())
    for result in summary["results"]:
        path = Path(result["metrics_path"])
        metrics = json.loads(path.read_text())
        metrics["config"]["learning_rate"] = 0.001
        metrics["config"]["lr_scheduler"] = {
            "kind": "reduce_on_plateau", "monitor": "validation_macro_f1", "mode": "max",
            "factor": 0.3, "patience": 3, "threshold": 0.0001, "threshold_mode": "abs",
            "cooldown": 0, "min_lr": 0.00003, "eps": 1e-8,
        }
        path.write_text(json.dumps(metrics))
        result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate.write_text(json.dumps(summary))
    return reference, candidate


def test_scheduler_report_requires_explicit_policy_and_preserves_subset_scope(
    tmp_path: Path,
) -> None:
    """Accept only the scheduler addition and retain development-feedback caveats.

    Args:
        tmp_path (Path): Temporary matched model metrics and prediction artifacts.
    """
    reference, candidate = _write_scheduler_pair(tmp_path)
    with pytest.raises(ValueError, match="scheduler"):
        PairedArchitectureReport(reference, candidate).build()
    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "learning_rate").build()
    report = PairedArchitectureReport(reference, candidate, "scheduler").build()
    assert report["comparison"] == "scheduler"
    assert report["validation_scope"] == "selected development folds"
    assert "not a full-cohort benchmark" in report["interpretation"]
    assert "validation feedback" in report["optimization_caveat"]
    assert report["paired"]["mean_macro_f1_difference"] == pytest.approx(0.9)
    assert report["eligible_epochs"] == 8
    assert not report["independent_outer_test"]


@pytest.mark.parametrize("mutation", [
    "base_rate", "dropout", "context", "diagnostics", "model_metadata", "parameter_count",
    "budget", "threshold", "monitor", "reference_scheduler", "missing_scheduler", "scope",
    "unknown_setting", "agent", "invalid_parameter_type", "device", "population", "min_lr",
])
def test_scheduler_report_rejects_confounded_controls_and_invalid_metadata(
    tmp_path: Path, mutation: str,
) -> None:
    """Reject scheduler comparisons with another changed control or invalid evidence.

    Args:
        tmp_path (Path): Temporary synthetic comparison artifacts.
        mutation (str): Inconsistency introduced after constructing a valid scheduler pair.
    """
    reference, candidate = _write_scheduler_pair(tmp_path)
    summary = json.loads(candidate.read_text())
    if mutation in {"budget", "scope"}:
        summary["plan"]["maximum_epochs" if mutation == "budget" else "validation_scope"] = (
            999 if mutation == "budget" else "all development folds"
        )
        candidate.write_text(json.dumps(summary))
    elif mutation == "population":
        path = Path(summary["results"][0]["metrics_path"]).parent / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "target"] = 1
        frame.to_csv(path, index=False)
    else:
        destination = reference if mutation == "reference_scheduler" else candidate
        saved = json.loads(destination.read_text())
        path = Path(saved["results"][0]["metrics_path"])
        metrics = json.loads(path.read_text())
        if mutation == "base_rate":
            metrics["config"]["learning_rate"] = 0.0003
        elif mutation == "dropout":
            metrics["config"]["transformer"]["dropout"] = 0.3
        elif mutation == "context":
            metrics["config"]["context_epochs"] = 3
        elif mutation == "diagnostics":
            metrics["config"]["diagnostics"]["save_predictions"] = False
        elif mutation == "model_metadata":
            metrics["model_config"]["d_model"] = 32
        elif mutation == "parameter_count":
            metrics["parameter_count"] += 1
        elif mutation == "invalid_parameter_type":
            metrics["parameter_count"] = float(metrics["parameter_count"])
        elif mutation == "threshold":
            metrics["config"]["lr_scheduler"]["threshold"] = -0.1
        elif mutation == "min_lr":
            metrics["config"]["lr_scheduler"]["min_lr"] = 0.1
        elif mutation == "monitor":
            metrics["config"]["lr_scheduler"].update(monitor="validation_loss", mode="min")
        elif mutation == "reference_scheduler":
            metrics["config"]["lr_scheduler"] = {"kind": "reduce_on_plateau"}
        elif mutation == "missing_scheduler":
            metrics["config"].pop("lr_scheduler")
        elif mutation == "unknown_setting":
            metrics["config"]["weight_decay"] = 0.01
        elif mutation == "agent":
            metrics["config"]["optimizer_agent"] = {"backend": "codex"}
        elif mutation == "device":
            metrics["device"] = "mps"
        _rewrite_metrics(destination, metrics)
    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "scheduler").build()


def test_scheduler_report_checks_fold_cap_when_both_plans_are_corrupted(tmp_path: Path) -> None:
    """Reject matching grouped caps that disagree with the fixed per-fold training budget.

    Args:
        tmp_path (Path): Temporary synthetic scheduler comparison artifacts.
    """
    reference, candidate = _write_scheduler_pair(tmp_path)
    for path in (reference, candidate):
        summary = json.loads(path.read_text())
        summary["plan"]["maximum_epochs"] = 6
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="epoch budget"):
        PairedArchitectureReport(reference, candidate, "scheduler").build()


@pytest.mark.parametrize("rate", [True, 0.0, float("nan"), None])
def test_scheduler_report_rejects_equal_invalid_initial_rates(
    tmp_path: Path, rate: float | bool | None,
) -> None:
    """Reject matching malformed initial rates before interpreting the scheduler control.

    Args:
        tmp_path (Path): Temporary synthetic scheduler comparison artifacts.
        rate (float | bool | None): Invalid base rate copied into both recorded recipes.
    """
    reference, candidate = _write_scheduler_pair(tmp_path)
    for summary_path in (reference, candidate):
        summary = json.loads(summary_path.read_text())
        path = Path(summary["results"][0]["metrics_path"])
        metrics = json.loads(path.read_text())
        metrics["config"]["learning_rate"] = rate
        _rewrite_metrics(summary_path, metrics)
    with pytest.raises(ValueError, match="valid initial rate"):
        PairedArchitectureReport(reference, candidate, "scheduler").build()


def test_plateau_notebook_reports_populated_scheduler_pair(tmp_path: Path) -> None:
    """Verify notebook tables use completed synthetic scheduler results without optimization.

    Args:
        tmp_path (Path): Temporary verified grouped metric artifacts.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    reference, candidate = _write_scheduler_pair(tmp_path)
    notebook = json.loads((Path(__file__).resolve().parents[1]
                           / "notebooks/22_transformer_plateau.ipynb").read_text())
    namespace = {"experiment": SimpleNamespace(output_dir=candidate.parent),
                 "reference_summary": reference,
                 "PairedArchitectureReport": PairedArchitectureReport,
                 "pd": pd, "json": json, "logger": Mock(), "display": Mock()}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("comparison = None"):
            exec(source, namespace)
    report = json.loads((candidate.parent / "paired-comparison.json").read_text())
    assert report["comparison"] == "scheduler"
    assert report["validation_scope"] == "selected development folds"
    assert len(namespace["class_rows"]) == 8
    assert namespace["display"].call_count == 3


def test_patch_notebook_renders_populated_architecture_scores_and_confusion(tmp_path: Path) -> None:
    """Render paired class scores and confusion matrices from synthetic saved predictions.

    Args:
        tmp_path (Path): Temporary matched architecture result artifacts.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    reference, candidate = _write_pair(tmp_path)
    notebook = json.loads((Path(__file__).resolve().parents[1]
                           / "notebooks/23_causal_patch_transformer.ipynb").read_text())
    namespace = {"experiment": SimpleNamespace(output_dir=candidate.parent),
                 "reference_summary": reference,
                 "PairedArchitectureReport": PairedArchitectureReport,
                 "pd": pd, "json": json, "logger": Mock(), "display": Mock(), "plt": plt}
    try:
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and (source.startswith("comparison = None")
                                                or "ConfusionMatrixDisplay" in source):
                exec(source, namespace)
        assert namespace["comparison"]["eligible_epochs"] == 8
        assert len(namespace["class_rows"]) == 8
        assert namespace["display"].call_count >= 4
        assert len(namespace["figure"].axes) == 2
    finally:
        plt.close("all")


def test_patch_seed_notebook_reports_completed_synthetic_pairs_without_training(
    tmp_path: Path,
) -> None:
    """Render complete paired seed scores, class charts, and probability diagnostics.

    Args:
        tmp_path (Path): Temporary verified synthetic seed-study artifacts.
    """
    from unittest.mock import Mock, patch

    import matplotlib

    from sleep_classifier.paired_seed_report import PairedSeedReport

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    study = PairedSeedReport(_write_dropout_seed_study(tmp_path))
    notebook = json.loads((Path(__file__).resolve().parents[1]
                           / "notebooks/24_patch_transformer_seeds.ipynb").read_text())
    namespace = {"study": study, "Path": Path, "json": json, "pd": pd,
                 "logger": Mock(), "display": Mock(),
                 "PairedArchitectureReport": PairedArchitectureReport}
    try:
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and any(marker in source for marker in (
                "statuses = study.status()", 'if all(row["status"]', "if seed_rows:",
                "if class_rows:", "probability_diagnostics = []",
            )):
                with patch.object(plt, "show"):
                    exec(source, namespace)
        assert len(namespace["seed_rows"]) == 2
        assert len(namespace["class_rows"]) == 16
        assert len(namespace["probability_diagnostics"]) == 4
        assert namespace["probability_diagnostics"][1]["checkpoint_nll"] == pytest.approx(
            0.3566749439,
        )
        saved = json.loads((study.output_dir / "summary.json").read_text())
        assert saved["mean_seed_macro_f1_difference"] == pytest.approx(0.9)
    finally:
        plt.close("all")


def test_patch_seed_probability_diagnostic_rejects_changed_argmax(tmp_path: Path) -> None:
    """Reject normalized probabilities that contradict the recorded predicted stage.

    Args:
        tmp_path (Path): Temporary synthetic seed-study artifacts to alter.
    """
    from sleep_classifier.paired_seed_report import PairedSeedReport

    study = PairedSeedReport(_write_dropout_seed_study(tmp_path))
    saved = json.loads(Path(study.cases[0]["reference_summary"]).read_text())
    path = Path(saved["results"][0]["metrics_path"]).parent / "validation_predictions.csv"
    predictions = pd.read_csv(path)
    column = "prediction" if "prediction" in predictions else "gru_prediction"
    different = (int(predictions.loc[0, column]) + 1) % 4
    predictions.loc[0, [f"probability_{stage}" for stage in range(4)]] = 0.1
    predictions.loc[0, f"probability_{different}"] = 0.7
    predictions.to_csv(path, index=False)
    notebook = json.loads((Path(__file__).resolve().parents[1]
                           / "notebooks/24_patch_transformer_seeds.ipynb").read_text())
    namespace = {"study": study, "statuses": study.status(), "Path": Path,
                 "json": json, "pd": pd}
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "probability_diagnostics = []" in "".join(cell["source"]))
    with pytest.raises(ValueError, match="probabilities disagree"):
        exec(source, namespace)


def _write_weight_decay_pair(root: Path) -> tuple[Path, Path]:
    """Write identical architecture recipes with one explicit decoupled-decay candidate.

    Args:
        root (Path): Temporary directory for two synthetic grouped experiments.
    """
    paths = _write_learning_rate_pair(root)
    for index, summary_path in enumerate(paths):
        summary = json.loads(summary_path.read_text())
        for result in summary["results"]:
            path = Path(result["metrics_path"])
            metrics = json.loads(path.read_text())
            metrics["config"]["learning_rate"] = 0.001
            metrics["optimizer"] = {"base_optimizer": "Adam", "updates": 2}
            if index == 1:
                metrics["config"]["optimizer"] = {
                    "name": "adamw", "weight_decay": 0.01, "exclude_bias_and_norm": True,
                }
                metrics["optimizer"] = {
                    "base_optimizer": "AdamW", "updates": 2,
                    "param_groups": [
                        {"name": "decay", "weight_decay": 0.01,
                         "parameter_names": ["projection.weight"], "parameter_count": 2500},
                        {"name": "no_decay", "weight_decay": 0.0,
                         "parameter_names": ["projection.bias"], "parameter_count": 64},
                    ],
                }
            path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary))
    return paths


def test_weight_decay_report_requires_explicit_policy(tmp_path: Path) -> None:
    """Accept only the declared decay intervention with effective optimizer evidence.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    reference, candidate = _write_weight_decay_pair(tmp_path)
    with pytest.raises(ValueError, match="explicit weight_decay"):
        PairedArchitectureReport(reference, candidate).build()

    report = PairedArchitectureReport(reference, candidate, "weight_decay").build()
    assert report["comparison"] == "weight_decay"
    assert report["eligible_epochs"] == 8
    assert not report["independent_outer_test"]
    assert "synthetic" in report["optimization_caveat"].lower()
    assert "reserved test" in report["selection_caveat"]


@pytest.mark.parametrize("mutation", [
    "model", "parameter_count", "rate", "architecture", "diagnostics", "extra_setting",
    "agent", "schedule", "zero_decay", "negative_decay", "boolean_decay", "nonfinite_decay",
    "no_exclusion", "unknown_optimizer_key", "reference_decay", "recorded_class", "missing_record",
    "recorded_decay", "recorded_count", "duplicate_parameters", "missing_group", "model_metadata",
])
def test_weight_decay_report_rejects_confounded_or_unverified_pairs(
    tmp_path: Path, mutation: str,
) -> None:
    """Reject recipe drift, invalid decay, or effective parameter-group mismatches.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
        mutation (str): Change that invalidates the controlled comparison.
    """
    reference, candidate = _write_weight_decay_pair(tmp_path)
    metrics_path = Path(json.loads(candidate.read_text())["results"][0]["metrics_path"])
    metrics = json.loads(metrics_path.read_text())
    if mutation == "model":
        metrics["model_config"]["d_model"] = 32
    elif mutation == "parameter_count":
        metrics["parameter_count"] += 1
    elif mutation == "rate":
        metrics["config"]["learning_rate"] = 0.0003
    elif mutation == "architecture":
        metrics["config"]["transformer"]["dropout"] = 0.3
    elif mutation == "diagnostics":
        metrics["config"]["diagnostics"]["save_predictions"] = False
    elif mutation == "extra_setting":
        metrics["config"]["label_smoothing"] = 0.1
    elif mutation == "agent":
        metrics["config"]["optimizer_agent"] = {"backend": "keep"}
    elif mutation == "schedule":
        metrics["config"]["lr_scheduler"] = {"kind": "reduce_on_plateau"}
    elif mutation in ("zero_decay", "negative_decay", "boolean_decay", "nonfinite_decay"):
        metrics["config"]["optimizer"]["weight_decay"] = {
            "zero_decay": 0.0, "negative_decay": -1.0,
            "boolean_decay": True, "nonfinite_decay": float("inf"),
        }[mutation]
    elif mutation == "no_exclusion":
        metrics["config"]["optimizer"]["exclude_bias_and_norm"] = False
    elif mutation == "unknown_optimizer_key":
        metrics["config"]["optimizer"]["unused"] = 1
    elif mutation == "reference_decay":
        reference_path = Path(json.loads(reference.read_text())["results"][0]["metrics_path"])
        control = json.loads(reference_path.read_text())
        control["config"]["optimizer"] = {"name": "adamw", "weight_decay": 0.01}
        _rewrite_metrics(reference, control)
    elif mutation == "recorded_class":
        metrics["optimizer"]["base_optimizer"] = "Adam"
    elif mutation == "missing_record":
        metrics["optimizer"] = None
    elif mutation == "recorded_decay":
        metrics["optimizer"]["param_groups"][1]["weight_decay"] = 0.01
    elif mutation == "recorded_count":
        metrics["optimizer"]["param_groups"][0]["parameter_count"] += 1
    elif mutation == "duplicate_parameters":
        metrics["optimizer"]["param_groups"][1]["parameter_names"] = ["projection.weight"]
    elif mutation == "missing_group":
        metrics["optimizer"]["param_groups"].pop()
    else:
        metrics["model_config"] = {}
    _rewrite_metrics(candidate, metrics)

    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "weight_decay").build()


def test_weight_decay_report_accepts_explicit_legacy_adam_control(tmp_path: Path) -> None:
    """Normalize a recorded zero-decay Adam setting to the historical implicit control.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    reference, candidate = _write_weight_decay_pair(tmp_path)
    summary = json.loads(reference.read_text())
    for result in summary["results"]:
        path = Path(result["metrics_path"])
        metrics = json.loads(path.read_text())
        metrics["config"]["optimizer"] = {"name": "adam", "weight_decay": 0.0}
        path.write_text(json.dumps(metrics))
        result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference.write_text(json.dumps(summary))

    assert PairedArchitectureReport(reference, candidate, "weight_decay").build()["comparison"] == (
        "weight_decay"
    )


def test_weight_decay_report_preserves_epoch_population_checks(tmp_path: Path) -> None:
    """Reject mismatched targets even when the optimizer recipes are valid.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    reference, candidate = _write_weight_decay_pair(tmp_path)
    path = Path(json.loads(candidate.read_text())["results"][0]["metrics_path"]).parent
    frame = pd.read_csv(path / "validation_predictions.csv")
    frame.loc[0, "time"] = 1
    frame.to_csv(path / "validation_predictions.csv", index=False)

    with pytest.raises(ValueError):
        PairedArchitectureReport(reference, candidate, "weight_decay").build()


def test_weight_decay_report_checks_grouped_epoch_cap(tmp_path: Path) -> None:
    """Require the saved grouped cap to agree with every recorded fold budget.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    reference, candidate = _write_weight_decay_pair(tmp_path)
    for summary_path in (reference, candidate):
        summary = json.loads(summary_path.read_text())
        summary["plan"]["maximum_epochs"] = 10
        summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="epoch budget"):
        PairedArchitectureReport(reference, candidate, "weight_decay").build()


def test_temporal_adapter_forwards_explicit_weight_decay_policy(tmp_path: Path) -> None:
    """Accept the paired optimizer policy before requiring temporal preparation metadata.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    from sleep_classifier.temporal_slice_report import TemporalSliceReport

    reference, candidate = _write_weight_decay_pair(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest"):
        TemporalSliceReport.from_grouped(reference, candidate, comparison="weight_decay")


def test_architecture_report_rejects_inconsistent_optimizer_provenance(tmp_path: Path) -> None:
    """Reject an effective optimizer that contradicts an otherwise unchanged recipe.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    reference, candidate = _write_pair(tmp_path)
    _rewrite_metrics(candidate, {"optimizer": {"base_optimizer": "AdamW", "updates": 2}})

    with pytest.raises(ValueError, match="Recorded optimizer"):
        PairedArchitectureReport(reference, candidate).build()
