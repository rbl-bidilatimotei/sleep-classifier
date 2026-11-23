import argparse
import hashlib
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from sleep_classifier.baseline import PreparedFeatureDataset
from sleep_classifier.experiment_evidence import EpochEvidenceReport
from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.grouped_validation import GroupedValidationExperiment
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.motion_feature_input import MOTION_DEFINITION, MOTION_FEATURE
from sleep_classifier.paired_architecture_report import PairedArchitectureReport
from sleep_classifier.window_feature_settings import WindowFeatureSettings


class PrefixBudgetReport:
    """Compare selected checkpoints from two budgets on one saved training trajectory."""

    def __init__(self, config_path: Path, reference_epochs: int = 40) -> None:
        """Bind the grouped candidate and its shorter checkpoint-selection budget.

        Args:
            config_path (Path): Grouped experiment configuration for the longer run.
            reference_epochs (int): Maximum epoch included in the prefix control.
        """
        self.experiment = GroupedValidationExperiment(config_path)
        self.output_dir = self.experiment.output_dir
        self.settings = self.experiment.model_settings
        self.reference_epochs = reference_epochs
        self.candidate_epochs = self.settings["epochs"]
        if (type(reference_epochs) is not int or reference_epochs <= 0
                or reference_epochs >= self.candidate_epochs):
            raise ValueError("The prefix budget must be positive and below the candidate budget")
        diagnostics = self.settings.get("diagnostics", {})
        if (not diagnostics.get("enabled") or not diagnostics.get("save_predictions")
                or not set(range(1, reference_epochs + 1)) <= set(
                    diagnostics.get("checkpoint_epochs", [])
                )):
            raise ValueError("Prefix controls require predictions and every prefix checkpoint")
        if (self.settings.get("device") != "mps" or self.settings.get("learning_rate") != 0.0003
                or self.settings.get("model_type") != "patch_transformer"
                or self.settings.get("optimizer_agent") or self.settings.get("lr_scheduler")):
            raise ValueError("The prefix experiment requires the declared constant-rate MPS recipe")
        self.evidence = EpochEvidenceReport(self.output_dir)


    @staticmethod
    def _digest(path: Path) -> str:
        """Hash a saved artifact without changing it.

        Args:
            path (Path): Existing artifact file.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()


    def _base_report(self, status: str, folds: list[dict[str, Any]]) -> dict[str, Any]:
        """Describe budget identity and interpretation without fabricating pending scores.

        Args:
            status (str): Prepared, partial, complete, or not_prepared state.
            folds (list[dict[str, Any]]): Available fold-level status or score records.
        """
        return {
            "status": status, "comparison": "trajectory_prefix_budget",
            "reference_epochs": self.reference_epochs, "candidate_epochs": self.candidate_epochs,
            "control_origin": "same_new_trajectory_prefix",
            "original_notebook_39_restored": False, "model_training_performed_by_report": False,
            "independent_outer_test": False, "folds": folds,
            "selection_caveat": (
                "The longer budget offers more validation-checkpoint selection opportunities; "
                "a selected validation gain alone does not establish improved generalization"
            ),
            "control_caveat": (
                "The prefix is a checkpoint from this new trajectory, not a restored notebook39 run"
            ),
        }


    def build(self) -> dict[str, Any]:
        """Read verified evidence and compute scores only after every fold completes."""
        if not (self.output_dir / "manifest.json").is_file():
            return self._base_report("not_prepared", [])
        manifest = self.experiment._validate_manifest()
        if manifest["device"] != "mps":
            raise ValueError("Prepared prefix experiment must use MPS")
        summary_path = self.output_dir / "summary.json"
        evidence = self.evidence.build()
        if not summary_path.is_file():
            status = "prepared" if evidence["status"] == "prepared" else "partial"
            return self._base_report(status, evidence["folds"])
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete":
            return self._base_report("partial", evidence["folds"])
        if summary.get("plan") != manifest["plan"]:
            raise ValueError("Completed summary differs from the frozen fold plan")
        results = summary.get("results", [])
        names = [fold["name"] for fold in manifest["plan"]["folds"]]
        if (len(results) != len(names)
                or sorted(result["fold"] for result in results) != sorted(names)):
            raise ValueError("Completed summary must contain every planned fold exactly once")
        by_name = {result["fold"]: result for result in results}
        folds = []
        frames: dict[str, list[pd.DataFrame]] = {"reference": [], "candidate": []}
        source = None
        for fold in manifest["plan"]["folds"]:
            row, selected, current_source = self._fold_result(fold, by_name[fold["name"]])
            if source is not None and source != current_source:
                raise ValueError("Motion source changed across folds")
            source = current_source
            folds.append(row)
            for role in frames:
                frames[role].append(selected[role])
        return self._complete_report(folds, frames, source)


    def _load_metrics(self, fold: str, result: dict[str, Any]) -> dict[str, Any]:
        """Validate the completed fold record against its stored recipe and model metadata.

        Args:
            fold (str): Planned numbered fold.
            result (dict[str, Any]): Registered metrics path and content hash.
        """
        expected = self.output_dir / fold / "results/metrics.json"
        path = Path(result["metrics_path"])
        path = path if path.is_absolute() else self.output_dir / path
        if path.resolve() != expected.resolve() or self._digest(path) != result["metrics_sha256"]:
            raise ValueError("Completed fold metrics path or hash differs")
        metrics = json.loads(path.read_text())
        for key in ("accuracy", "macro_f1", "best_epoch", "participants"):
            if result.get(key) != metrics.get(key):
                raise ValueError(f"Completed summary differs from metrics at {key}")
        excluded = {"prepare_config", "output_dir"}
        actual = {key: value for key, value in metrics["config"].items() if key not in excluded}
        declared = {key: value for key, value in self.settings.items() if key not in excluded}
        for recipe in (actual, declared):
            motion = dict(recipe["motion_features"])
            motion["audit_config"] = str(
                (self.experiment.model_path.parent / motion["audit_config"]).resolve()
            )
            recipe["motion_features"] = motion
        if actual != declared or metrics.get("device") != "mps":
            raise ValueError("Completed fold differs from the configured MPS recipe")
        factory = ModelFactory(self.settings)
        if (metrics.get("model_config") != factory.checkpoint_config(7, 4)
                or metrics.get("features") != [*FEATURE_COLUMNS, MOTION_FEATURE]
                or metrics.get("parameter_count") != 2900):
            raise ValueError("Completed fold model or raw input schema differs")
        PairedArchitectureReport._validate_window_feature_record(
            metrics, WindowFeatureSettings(self.settings.get("window_features")),
        )
        PairedArchitectureReport._validate_motion_feature_record(metrics, MOTION_DEFINITION)
        return metrics


    @staticmethod
    def _validate_history(records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
        """Require registered training history to match each linked diagnostic epoch.

        Args:
            records (list[dict[str, Any]]): Saved per-epoch diagnostic records.
            metrics (dict[str, Any]): Registered final metrics including training history.
        """
        history = metrics.get("history", [])
        if len(history) != len(records):
            raise ValueError("Training history and diagnostic epoch counts differ")
        for item, record in zip(history, records, strict=True):
            expected = {
                "epoch": record["epoch"],
                "validation_macro_f1": record["validation_macro_f1"],
                "validation_accuracy": record["validation"]["accuracy"],
                "validation_loss": record["validation_criterion_nll"],
                "train_loss": record["training"]["stochastic_nll"],
            }
            if any(key not in item or not np.isclose(item[key], value, atol=1e-12, rtol=0)
                   for key, value in expected.items()):
                raise ValueError("Training history differs from diagnostic epoch metrics")


    def _select_epochs(
        self, records: list[dict[str, Any]], metrics: dict[str, Any],
    ) -> tuple[int, int]:
        """Replay strict-improvement selection and the unchanged early-stopping rule.

        Args:
            records (list[dict[str, Any]]): Contiguous validated completed-epoch records.
            metrics (dict[str, Any]): Final selected checkpoint metadata.
        """
        if (not records or [record["epoch"] for record in records]
                != list(range(1, len(records) + 1))):
            raise ValueError("The trajectory must contain every epoch starting at one")
        if len(records) > self.candidate_epochs:
            raise ValueError("Trajectory exceeds the configured candidate budget")
        best = -1.0
        full_epoch = 0
        prefix_epoch = 0
        stale = 0
        for record in records:
            epoch = record["epoch"]
            score = record["validation_macro_f1"]
            if (not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not 0 <= score <= 1):
                raise ValueError("Selection macro-F1 must be finite and between zero and one")
            if score > best:
                best, full_epoch, stale = score, epoch, 0
            else:
                stale += 1
            if epoch <= self.reference_epochs:
                prefix_epoch = full_epoch
            if stale >= self.settings["patience"] and epoch != len(records):
                raise ValueError("Trajectory continues after patience was exhausted")
            if record["optimizer"]["learning_rates"] != [self.settings["learning_rate"]]:
                raise ValueError("The supposedly constant learning rate changed")
        if len(records) < self.candidate_epochs and stale < self.settings["patience"]:
            raise ValueError("Completed trajectory ended before its cap or patience limit")
        if metrics["best_epoch"] != full_epoch or not np.isclose(metrics["macro_f1"], best):
            raise ValueError("Selected final checkpoint differs from the trajectory maximum")
        return prefix_epoch, full_epoch


    @staticmethod
    def _scores(frame: pd.DataFrame) -> dict[str, Any]:
        """Compute four-class and probability scores from verified target predictions.

        Args:
            frame (pd.DataFrame): Selected target identities, predictions, and probability losses.
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            frame.target, frame.prediction, labels=range(4), zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(frame.target, frame.prediction)),
            "macro_f1": float(f1.mean()), "nll": float(frame.nll.mean()),
            "confusion_matrix": confusion_matrix(frame.target, frame.prediction,
                                                   labels=range(4)).tolist(),
            "classes": {str(i): {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])}
                        for i in range(4)},
        }


    def _selected_frame(
        self, fold: str, epoch: int, record: dict[str, Any], expected: pd.DataFrame,
    ) -> pd.DataFrame:
        """Check a selected epoch's labels and scores against the frozen validation population.

        Args:
            fold (str): Planned numbered fold.
            epoch (int): Selected epoch within the new trajectory.
            record (dict[str, Any]): Epoch metrics with hashed linked artifacts.
            expected (pd.DataFrame): Eligible validation identities from the prepared fold.
        """
        frame = self.evidence._prediction_frame(fold, epoch)
        keys = ["participant_id", "time", "target"]
        expected = expected[keys].sort_values(keys, kind="stable").reset_index(drop=True)
        if not frame[keys].equals(expected):
            raise ValueError("Selected epoch differs from the frozen validation target identities")
        values = frame[[f"probability_{i}" for i in range(4)]].to_numpy(dtype=float)
        predictions = values.argmax(axis=1)
        if "prediction" in frame and not np.array_equal(frame.prediction, predictions):
            raise ValueError("Saved epoch predictions disagree with their probabilities")
        frame["prediction"] = predictions
        scores = self._scores(frame)
        if (not np.isclose(scores["macro_f1"], record["validation_macro_f1"], atol=1e-12, rtol=0)
                or not np.isclose(scores["accuracy"], record["validation"]["accuracy"],
                                  atol=1e-12, rtol=0)):
            raise ValueError("Selected epoch score differs from its linked predictions")
        return frame


    def _fold_result(
        self, fold: dict[str, Any], result: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
        """Build one paired prefix/full comparison from unchanged saved evidence.

        Args:
            fold (dict[str, Any]): Frozen participant membership and numbered name.
            result (dict[str, Any]): Registered completed-fold metrics reference.
        """
        name = fold["name"]
        metrics = self._load_metrics(name, result)
        records = self.evidence._read_epochs(name)
        self._validate_history(records, metrics)
        prefix_epoch, full_epoch = self._select_epochs(records, metrics)
        expected = PreparedFeatureDataset(self.output_dir / name / "features").load()[
            "partitions"
        ]["validation"]
        if set(expected.participant_id) != set(fold["validation"]):
            raise ValueError("Prepared validation membership differs from the fold plan")
        by_epoch = {record["epoch"]: record for record in records}
        selected = {role: self._selected_frame(name, epoch, by_epoch[epoch], expected)
                    for role, epoch in (("reference", prefix_epoch), ("candidate", full_epoch))}
        checkpoint = by_epoch[prefix_epoch]["files"].get("checkpoint")
        if checkpoint is None:
            raise ValueError("Selected prefix epoch has no saved checkpoint")
        checkpoint_path = self.evidence._linked_path(
            self.output_dir / name / "results/diagnostics", checkpoint,
        )
        final_path = self.output_dir / name / "results/model_checkpoint.pt"
        saved = pd.read_csv(final_path.parent / "validation_predictions.csv",
                            dtype={"participant_id": str})
        saved = saved.sort_values(["participant_id", "time", "target"], kind="stable")
        keys = ["participant_id", "time", "target", "prediction"]
        if not saved[keys].reset_index(drop=True).equals(selected["candidate"][keys]):
            raise ValueError("Final predictions differ from the selected full-trajectory epoch")
        probability_columns = [f"probability_{i}" for i in range(4)]
        if not np.allclose(saved[probability_columns].to_numpy(),
                           selected["candidate"][probability_columns].to_numpy(),
                           atol=1e-7, rtol=0):
            raise ValueError("Final probabilities differ from the selected full-trajectory epoch")
        scores = {role: self._scores(frame) for role, frame in selected.items()}
        for key in ("accuracy", "macro_f1", "confusion_matrix", "classes"):
            if scores["candidate"][key] != metrics[key]:
                raise ValueError(f"Final metrics differ from selected predictions at {key}")
        row = {"fold": name, "completed_epochs": len(records),
               "prefix_best_epoch": prefix_epoch, "full_best_epoch": full_epoch,
               "reference": scores["reference"], "candidate": scores["candidate"],
               "macro_f1_difference": (
                   scores["candidate"]["macro_f1"] - scores["reference"]["macro_f1"]
               ),
               "prefix_checkpoint": {
                   "path": str(checkpoint_path), "sha256": self._digest(checkpoint_path),
               },
               "candidate_checkpoint": {
                   "path": str(final_path), "sha256": self._digest(final_path),
               }}
        motion = metrics["motion_features"]
        source = {key: motion[key] for key in (
            "definition", "audit_config_sha256", "snapshot_manifest_sha256",
        )}
        return row, selected, source


    def _complete_report(
        self, folds: list[dict[str, Any]], frames: dict[str, list[pd.DataFrame]],
        source: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Summarize complete folds while retaining the common-trajectory limitation.

        Args:
            folds (list[dict[str, Any]]): Verified paired fold scores.
            frames (dict[str, list[pd.DataFrame]]): Selected prediction tables for each role.
            source (dict[str, Any] | None): Common immutable motion source identity.
        """
        pooled = {role: pd.concat(values, ignore_index=True) for role, values in frames.items()}
        for frame in pooled.values():
            if frame.duplicated(["participant_id", "time"]).any():
                raise ValueError("Validation participants or targets repeat across folds")
        report = self._base_report("complete", folds)
        report.update({role: self._scores(frame) for role, frame in pooled.items()})
        report["mean_fold"] = {role: mean(row[role]["macro_f1"] for row in folds)
                               for role in frames}
        report["paired"] = {"mean_macro_f1_difference": mean(
            row["macro_f1_difference"] for row in folds
        )}
        report["eligible_epochs"] = len(pooled["reference"])
        report["fixed_motion_inputs"] = source
        report["participants"] = [
            {"participant_id": participant,
             **{role: self._scores(frame.loc[frame.participant_id == participant])
                for role, frame in pooled.items()}}
            for participant in sorted(pooled["reference"].participant_id.unique())
        ]
        for participant in report["participants"]:
            participant["macro_f1_difference"] = (
                participant["candidate"]["macro_f1"] - participant["reference"]["macro_f1"]
            )
        return report


def main() -> None:
    """Write a complete prefix-budget report without running model optimization."""
    configure_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Compare checkpoint budgets on one saved trajectory",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-epochs", type=int, default=40)
    parser.add_argument("--report", action="store_true", required=True)
    args = parser.parse_args()
    experiment = PrefixBudgetReport(args.config, args.reference_epochs)
    report = experiment.build()
    if report["status"] == "complete":
        path = experiment.output_dir / "prefix-budget-comparison.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Prefix-budget report saved: %s", path)
    else:
        logger.info("Prefix-budget report pending: %s", report["status"])


if __name__ == "__main__":
    main()
