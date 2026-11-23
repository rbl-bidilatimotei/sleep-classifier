import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
import torch  # noqa: E402

from sleep_classifier import prefix_budget_report as module  # noqa: E402
from sleep_classifier.epoch_diagnostics import (  # noqa: E402
    EpochDiagnostics,
    EpochDiagnosticsSettings,
)
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.motion_feature_input import MOTION_DEFINITION, MOTION_FEATURE  # noqa: E402
from sleep_classifier.prefix_budget_report import PrefixBudgetReport  # noqa: E402


class SyntheticStudy:
    """Write synthetic saved evidence without optimization or production sensor reads."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bind isolated output and replace only grouped setup and prepared-data loading.

        Args:
            root (Path): Temporary experiment directory.
            monkeypatch (pytest.MonkeyPatch): Scoped dependency replacement utility.
        """
        self.root = root
        self.settings = yaml.safe_load(Path("configs/motion-learning-rate.yaml").read_text())
        self.settings.update(epochs=4, patience=2)
        self.settings["diagnostics"]["checkpoint_epochs"] = [1, 2]
        self.plan = {"folds": [
            {"name": "fold-01", "train": ["p2"], "validation": ["p1"]},
            {"name": "fold-02", "train": ["p1"], "validation": ["p2"]},
        ]}
        self.manifest = {"device": "mps", "plan": self.plan}
        self.experiment = Mock(output_dir=root, model_settings=self.settings,
                               model_path=root / "model.yaml")
        self.experiment._validate_manifest.return_value = self.manifest
        monkeypatch.setattr(
            module, "GroupedValidationExperiment", Mock(return_value=self.experiment),
        )
        monkeypatch.setattr(module, "PreparedFeatureDataset", self.prepared_dataset)
        self.summary: dict[str, Any] = {"status": "complete", "plan": self.plan, "results": []}


    def prepared_dataset(self, path: Path) -> Mock:
        """Return validation identities for one synthetic fold.

        Args:
            path (Path): Requested fold feature directory.
        """
        index = int(path.parent.name[-2:])
        frame = self.frame(f"p{index}")
        return Mock(load=Mock(return_value={"partitions": {"validation": frame}}))


    @staticmethod
    def frame(participant: str) -> pd.DataFrame:
        """Create four distinct labeled validation epochs.

        Args:
            participant (str): Synthetic validation participant.
        """
        return pd.DataFrame({"participant_id": [participant] * 4,
                             "time": [0, 30, 60, 90], "target": [0, 1, 2, 3]})


    def prepare(self) -> None:
        """Write the frozen manifest without creating model results."""
        self.root.mkdir(exist_ok=True)
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))


    def complete(self, correct_counts: list[int]) -> None:
        """Record the same deterministic trajectory in both synthetic folds.

        Args:
            correct_counts (list[int]): Correct target counts for each completed epoch.
        """
        self.prepare()
        self.summary["results"] = [self.write_fold(fold, correct_counts)
                                   for fold in self.plan["folds"]]
        self.save_summary()


    def save_summary(self) -> None:
        """Persist the current synthetic completion registry."""
        (self.root / "summary.json").write_text(json.dumps(self.summary))


    def write_fold(self, fold: dict[str, Any], counts: list[int]) -> dict[str, Any]:
        """Create real diagnostic files with linked prediction and checkpoint hashes.

        Args:
            fold (dict[str, Any]): Synthetic participant membership.
            counts (list[int]): Number of correct targets in each epoch.
        """
        directory = self.root / fold["name"] / "results"
        recorder = EpochDiagnostics(EpochDiagnosticsSettings(self.settings["diagnostics"]),
                                    directory, {}, 1.0)
        frame = self.frame(fold["validation"][0])
        best_score, best_epoch = -1.0, 0
        selected = frame
        history: list[dict[str, Any]] = []
        for epoch, count in enumerate(counts, 1):
            predicted = np.arange(4)
            predicted[count:] = (predicted[count:] + 1) % 4
            probabilities = np.full((4, 4), 0.05)
            probabilities[np.arange(4), predicted] = 0.85
            table = recorder._prediction_table(frame, probabilities)
            score = PrefixBudgetReport._scores(table)["macro_f1"]
            recorder.begin_epoch()
            recorder.record_gradient(0.5)
            recorder.record_epoch(epoch, frame, probabilities, 0.7, [0.0003],
                                  torch.nn.Linear(1, 4), float(table.nll.mean()), 0.8, score)
            history.append({"epoch": epoch, "train_loss": 0.8,
                            "validation_loss": float(table.nll.mean()),
                            "validation_macro_f1": score,
                            "validation_accuracy": float(
                                (table.prediction == table.target).mean(),
                            )})
            if score > best_score:
                best_score, best_epoch, selected = score, epoch, table
        selected.to_csv(directory / "validation_predictions.csv", index=False)
        (directory / "model_checkpoint.pt").write_bytes(b"opaque final checkpoint")
        state = {"schema_version": 1, "heart_rate_scale": 1.0, "delta_mean": 0.0,
                 "delta_scale": 1.0, "available_training_targets": 4,
                 "scaling_fit": "training_targets_only"}
        raw_features = [*FEATURE_COLUMNS, MOTION_FEATURE]
        metrics = PrefixBudgetReport._scores(selected) | {
            "history": history,
            "config": copy.deepcopy(self.settings), "device": "mps", "best_epoch": best_epoch,
            "model_config": ModelFactory(self.settings).checkpoint_config(7, 4),
            "features": raw_features, "parameter_count": 2900,
            "model_features": ModelFactory(self.settings).model_features(raw_features),
            "window_features": {"settings": self.settings["window_features"], "scaling": state},
            "participants": {fold["validation"][0]: {"eligible_epochs": 4}},
            "provenance": {"participants": [{"partition": "train", "eligible_epochs": 4}]},
            "motion_features": {
                "name": "fixed_lag_motion", "definition": MOTION_DEFINITION,
                "audit_config_sha256": "a" * 64, "snapshot_manifest_sha256": "b" * 64,
                "scaling_fit": "training_targets_only", "missing_policy": "training_mean",
                "training_observed_targets": 4, "training_missing_targets": 0,
                "validation_observed_targets": 4, "validation_missing_targets": 0,
                "imputation_mean": 0.5,
            },
        }
        path = directory / "metrics.json"
        path.write_text(json.dumps(metrics))
        return {"fold": fold["name"], "metrics_path": str(path),
                "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **{key: metrics[key] for key in (
                    "accuracy", "macro_f1", "best_epoch", "participants",
                )}}


    def report(self, prefix: int = 2) -> PrefixBudgetReport:
        """Construct a report using the isolated grouped experiment.

        Args:
            prefix (int): Short checkpoint-selection budget.
        """
        return PrefixBudgetReport(self.root / "grouped.yaml", prefix)


def test_pending_report_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep absent and partial runs score-free and avoid creating report artifacts.

    Args:
        tmp_path (Path): Isolated temporary directory.
        monkeypatch (pytest.MonkeyPatch): Scoped setup replacement utility.
    """
    study = SyntheticStudy(tmp_path / "study", monkeypatch)
    assert study.report().build()["status"] == "not_prepared"
    assert not study.root.exists()
    study.prepare()
    report = study.report().build()
    assert report["status"] == "prepared"
    assert "reference" not in report
    assert not (study.root / "prefix-budget-comparison.json").exists()
    study.experiment._validate_manifest.assert_called_once()


def test_completed_budget_uses_real_prefix_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select saved prefix and final winners and recompute pooled stage scores.

    Args:
        tmp_path (Path): Isolated temporary directory.
        monkeypatch (pytest.MonkeyPatch): Scoped setup replacement utility.
    """
    study = SyntheticStudy(tmp_path, monkeypatch)
    study.complete([1, 2, 3, 4])
    report = study.report().build()
    assert report["status"] == "complete"
    assert report["candidate"]["macro_f1"] == 1.0
    assert report["reference"]["macro_f1"] < 1.0
    assert report["eligible_epochs"] == 8
    assert not report["original_notebook_39_restored"]
    for fold in report["folds"]:
        assert (fold["prefix_best_epoch"], fold["full_best_epoch"]) == (2, 4)
        checkpoint = fold["prefix_checkpoint"]
        assert PrefixBudgetReport._digest(Path(checkpoint["path"])) == checkpoint["sha256"]
    assert not (tmp_path / "prefix-budget-comparison.json").exists()


def test_early_stop_before_prefix_and_ties_keep_first_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the same earliest checkpoint when unchanged patience ends before the prefix cap.

    Args:
        tmp_path (Path): Isolated temporary directory.
        monkeypatch (pytest.MonkeyPatch): Scoped setup replacement utility.
    """
    study = SyntheticStudy(tmp_path, monkeypatch)
    study.settings["epochs"] = 8
    study.settings["diagnostics"]["checkpoint_epochs"] = list(range(1, 5))
    study.complete([3, 3, 3])
    result = study.report(4).build()
    assert result["reference"] == result["candidate"]
    assert all(row["prefix_best_epoch"] == row["full_best_epoch"] == 1 for row in result["folds"])


@pytest.mark.parametrize("mutation", ["sha", "epoch_gap", "truncated", "population", "schema",
                                      "selection", "continued", "probabilities", "source"])
def test_reject_inconsistent_saved_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    """Fail before producing complete comparisons when linked evidence contradicts the study.

    Args:
        tmp_path (Path): Isolated temporary directory.
        monkeypatch (pytest.MonkeyPatch): Scoped setup replacement utility.
        mutation (str): Contradiction introduced into otherwise valid saved evidence.
    """
    study = SyntheticStudy(tmp_path, monkeypatch)
    study.complete([1, 2, 3, 4] if mutation != "continued" else [3, 3, 3, 4])
    directory = tmp_path / "fold-01/results"
    if mutation == "sha":
        (directory / "metrics.json").write_text("{}")
    elif mutation == "epoch_gap":
        (directory / "diagnostics/epochs/epoch-002.json").unlink()
    elif mutation == "truncated":
        (directory / "diagnostics/epochs/epoch-004.json").unlink()
    elif mutation == "probabilities":
        path = directory / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame["probability_0"] += 0.01
        frame.to_csv(path, index=False)
    elif mutation in ("schema", "selection", "source"):
        path = directory / "metrics.json"
        metrics = json.loads(path.read_text())
        if mutation == "schema":
            metrics["model_features"].pop()
        elif mutation == "selection":
            metrics["best_epoch"] = 3
            study.summary["results"][0]["best_epoch"] = 3
        else:
            metrics["motion_features"]["snapshot_manifest_sha256"] = "c" * 64
        path.write_text(json.dumps(metrics))
        study.summary["results"][0]["metrics_sha256"] = PrefixBudgetReport._digest(path)
        study.save_summary()
    elif mutation == "population":
        study.plan["folds"][0]["validation"] = ["reserved-test"]
        study.prepare()
        study.save_summary()
    with pytest.raises(ValueError):
        study.report().build()


def test_missing_completion_registry_remains_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the full registry even when every fold has final metrics.

    Args:
        tmp_path (Path): Isolated temporary directory.
        monkeypatch (pytest.MonkeyPatch): Scoped setup replacement utility.
    """
    study = SyntheticStudy(tmp_path, monkeypatch)
    study.complete([1, 2, 3, 4])
    (tmp_path / "summary.json").unlink()
    result = study.report().build()
    assert result["status"] == "partial"
    assert "reference" not in result
