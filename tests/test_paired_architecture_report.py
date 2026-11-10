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
