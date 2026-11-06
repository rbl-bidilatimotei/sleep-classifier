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
