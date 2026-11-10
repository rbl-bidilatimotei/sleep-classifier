import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from sleep_classifier.features import FEATURE_COLUMNS

SHARED_RECIPE_KEYS = (
    "seed", "epochs", "batch_size", "context_epochs", "learning_rate", "patience",
    "gradient_clip", "device",
)
CLASS_LABELS = (0, 1, 2, 3)


class PairedArchitectureReport:
    """Compare completed grouped experiments with matched evaluation populations."""

    def __init__(
        self, reference_summary: Path, candidate_summary: Path,
        comparison: Literal["architecture", "training_budget", "context"] = "architecture",
    ) -> None:
        """Store the two grouped experiment summaries.

        Args:
            reference_summary (Path): Completed reference experiment summary.
            candidate_summary (Path): Completed candidate experiment summary.
            comparison (Literal["architecture", "training_budget", "context"]):
                Permitted recipe differences.
        """
        if comparison not in ("architecture", "training_budget", "context"):
            raise ValueError("Unknown paired comparison policy")

        self.reference_summary = reference_summary.resolve()
        self.candidate_summary = candidate_summary.resolve()
        self.comparison = comparison


    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read a JSON object used as recorded experiment metadata.

        Args:
            path (Path): Summary or metrics file.
        """
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")

        return value


    def _load_summary(self, path: Path) -> dict[str, Any]:
        """Require a complete grouped experiment with one result per fold.

        Args:
            path (Path): Recorded grouped summary.
        """
        summary = self._read_json(path)
        if summary.get("status") != "complete":
            raise ValueError("Both grouped experiments must be complete")

        folds = summary["plan"]["folds"]
        planned = [fold["name"] for fold in folds]
        recorded = [result["fold"] for result in summary["results"]]
        if len(planned) < 2 or len(set(planned)) != len(planned):
            raise ValueError("Grouped plans require at least two unique folds")
        if len(recorded) != len(set(recorded)) or set(planned) != set(recorded):
            raise ValueError("Completed results must match the planned folds exactly")

        return summary


    def _validate_plans(self, reference: dict[str, Any], candidate: dict[str, Any]) -> None:
        """Require matched participant assignments and evaluation seeds.

        Args:
            reference (dict[str, Any]): Reference grouped plan.
            candidate (dict[str, Any]): Candidate grouped plan.
        """
        for key in (
            "development", "reserved_test", "split_seed", "training_seed", "maximum_epochs",
            "checkpoint_selection", "independent_outer_test",
        ):
            if key == "maximum_epochs" and self.comparison == "training_budget":
                continue
            if key not in reference or key not in candidate or reference[key] != candidate[key]:
                raise ValueError(f"Grouped plans differ at {key}")

        reference_folds = {fold["name"]: fold for fold in reference["folds"]}
        candidate_folds = {fold["name"]: fold for fold in candidate["folds"]}
        if reference_folds != candidate_folds:
            raise ValueError("Grouped participant folds differ")


    def _load_fold(self, result: dict[str, Any], summary_path: Path) -> dict[str, Any]:
        """Verify a fold's metrics artifact before reading its predictions.

        Args:
            result (dict[str, Any]): Fold result containing a metrics path and hash.
            summary_path (Path): Parent summary used for relative artifact paths.
        """
        path = Path(result["metrics_path"])
        if not path.is_absolute():
            path = summary_path.parent / path
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != result["metrics_sha256"]:
            raise ValueError("Fold metrics changed after summary creation")

        metrics = self._read_json(path)
        if tuple(metrics.get("features", [])) != FEATURE_COLUMNS:
            raise ValueError("Fold features differ from the six-feature schema")
        for key in ("accuracy", "macro_f1", "best_epoch", "participants"):
            if result[key] != metrics[key]:
                raise ValueError(f"Fold summary differs from verified metrics at {key}")

        return {"metrics": metrics, "predictions": self._read_predictions(path.parent)}


    @staticmethod
    def _read_predictions(directory: Path) -> pd.DataFrame:
        """Read unique, finite validation predictions from a saved fold.

        Args:
            directory (Path): Directory containing validation_predictions.csv.
        """
        frame = pd.read_csv(directory / "validation_predictions.csv", dtype={"participant_id": str})
        prediction = "prediction" if "prediction" in frame else "gru_prediction"
        required = ["participant_id", "time", "target", prediction]
        if not set(required).issubset(frame.columns):
            raise ValueError("Validation predictions lack required columns")
        frame = frame.loc[:, required].rename(columns={prediction: "prediction"})
        if frame.empty or frame.isna().any().any():
            raise ValueError("Validation predictions must be nonempty and complete")
        numeric = frame[["time", "target", "prediction"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Validation prediction values must be finite")
        if (not frame.target.isin(CLASS_LABELS).all()
                or not frame.prediction.isin(CLASS_LABELS).all()):
            raise ValueError("Validation predictions must use stage codes 0..3")
        frame["time"] = frame["time"].astype(float)
        frame["target"] = frame["target"].astype(int)
        frame["prediction"] = frame["prediction"].astype(int)
        if frame.duplicated(["participant_id", "time"]).any():
            raise ValueError("Validation prediction epochs must be unique")

        return frame.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)


    def _validate_recipes(self, reference: dict[str, Any], candidate: dict[str, Any]) -> None:
        """Validate model and optimization settings under the selected comparison policy.

        Args:
            reference (dict[str, Any]): Verified reference fold metrics.
            candidate (dict[str, Any]): Verified candidate fold metrics.
        """
        if self.comparison == "training_budget":
            self._validate_budget_recipe(reference, candidate)
        elif self.comparison == "context":
            self._validate_context_recipe(reference, candidate)

        for key in SHARED_RECIPE_KEYS:
            if key in ("epochs", "patience") and self.comparison == "training_budget":
                continue
            if key == "context_epochs" and self.comparison == "context":
                continue
            if (key not in reference["config"] or key not in candidate["config"]
                    or reference["config"][key] != candidate["config"][key]):
                raise ValueError(f"Shared training recipe differs at {key}")
        if reference["config"].get("optimizer_agent") or candidate["config"].get("optimizer_agent"):
            raise ValueError("Paired comparisons require the plain Adam recipes")
        if reference["coverage"] != candidate["coverage"]:
            raise ValueError("Fold eligibility or coverage differs")
        if reference["device"] != candidate["device"]:
            raise ValueError("Resolved training devices differ")
        for key in (
            "split", "rule", "feature_columns", "test_observations_read", "steps_used",
            "source_inventory", "epoch_start_assumed",
        ):
            if reference["provenance"].get(key) != candidate["provenance"].get(key):
                raise ValueError(f"Fold preparation provenance differs at {key}")


    @staticmethod
    def _context_signature(metrics: dict[str, Any]) -> dict[str, Any]:
        """Validate context metadata and retain settings that must remain fixed.

        Args:
            metrics (dict[str, Any]): Verified Transformer fold metrics.
        """
        config = metrics["config"]
        context = config.get("context_epochs")
        model = metrics.get("model_config")
        if config.get("model_type") != "transformer" or not isinstance(model, dict):
            raise ValueError("Context comparison requires Transformer model metadata")
        if type(context) is not int or context < 1:
            raise ValueError("Context length must be a positive integer")
        if (config.get("transformer", {}).get("max_context") != context
                or model.get("max_context") != context):
            raise ValueError("Position table capacity must match the context length")
        if type(metrics.get("parameter_count")) is not int or metrics["parameter_count"] < 1:
            raise ValueError("Context comparison requires a positive parameter count")

        excluded = {"context_epochs", "prepare_config", "output_dir", "transformer"}
        return {
            "recipe": {key: value for key, value in config.items() if key not in excluded},
            "transformer": {key: value for key, value in config["transformer"].items()
                            if key != "max_context"},
            "model": {key: value for key, value in model.items() if key != "max_context"},
        }


    @classmethod
    def _validate_context_recipe(
        cls, reference: dict[str, Any], candidate: dict[str, Any],
    ) -> None:
        """Permit only longer context and its learned positional parameters.

        Args:
            reference (dict[str, Any]): Verified shorter-context fold metrics.
            candidate (dict[str, Any]): Verified longer-context fold metrics.
        """
        if cls._context_signature(reference) != cls._context_signature(candidate):
            raise ValueError("Context comparison requires the same model and training recipe")
        difference = candidate["config"]["context_epochs"] - reference["config"]["context_epochs"]
        if difference <= 0:
            raise ValueError("Context comparison requires a longer candidate history")
        width = reference["model_config"].get("d_model")
        if type(width) is not int or width < 1:
            raise ValueError("Context comparison requires a positive model width")
        if candidate["parameter_count"] - reference["parameter_count"] != difference * width:
            raise ValueError("Parameter difference must match the learned position table")


    @staticmethod
    def _validate_budget_recipe(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
        """Permit only a larger training budget for an unchanged model recipe.

        Args:
            reference (dict[str, Any]): Verified reference fold metrics.
            candidate (dict[str, Any]): Verified candidate fold metrics.
        """
        for key in ("model_config", "parameter_count"):
            if key not in reference or reference[key] != candidate.get(key):
                raise ValueError(f"Training-budget model differs at {key}")

        excluded = {"epochs", "patience", "prepare_config", "output_dir"}
        recipes = [{key: value for key, value in run["config"].items() if key not in excluded}
                   for run in (reference, candidate)]
        if recipes[0] != recipes[1]:
            raise ValueError("Training-budget comparison requires the same model and recipe")

        for key in ("epochs", "patience"):
            values = [run["config"].get(key) for run in (reference, candidate)]
            if any(type(value) is not int or value < 1 for value in values):
                raise ValueError(f"Training-budget {key} must be positive integers")
            if values[1] < values[0]:
                raise ValueError(f"Training-budget candidate must not reduce {key}")
        if all(reference["config"][key] == candidate["config"][key]
               for key in ("epochs", "patience")):
            raise ValueError("Training-budget comparison requires a budget increase")


    @staticmethod
    def _validate_budget_limit(plan: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Check that the grouped cap agrees with the recorded fold recipe.

        Args:
            plan (dict[str, Any]): Recorded grouped plan.
            metrics (dict[str, Any]): Verified fold metrics with its epoch cap.
        """
        maximum = plan.get("maximum_epochs")
        expected = len(plan["folds"]) * metrics["config"]["epochs"]
        if type(maximum) is not int or maximum != expected:
            raise ValueError("Grouped epoch budget differs from the fold recipe")


    @staticmethod
    def _validate_population(
        fold: dict[str, Any], reference: pd.DataFrame, candidate: pd.DataFrame,
        metrics: dict[str, Any], candidate_metrics: dict[str, Any],
    ) -> None:
        """Check exact held-out epoch alignment and recorded eligible counts.

        Args:
            fold (dict[str, Any]): Frozen training and validation membership.
            reference (pd.DataFrame): Sorted reference predictions.
            candidate (pd.DataFrame): Sorted candidate predictions.
            metrics (dict[str, Any]): Verified reference metrics and participant counts.
            candidate_metrics (dict[str, Any]): Verified candidate participant counts.
        """
        columns = ["participant_id", "time", "target"]
        if not reference[columns].equals(candidate[columns]):
            raise ValueError("Prediction populations or targets differ between architectures")
        if set(reference.participant_id) != set(fold["validation"]):
            raise ValueError("Predictions differ from the planned validation participants")
        for recorded in (metrics, candidate_metrics):
            if set(recorded["participants"]) != set(fold["validation"]):
                raise ValueError("Participant metrics differ from planned validation membership")
            for participant, metadata in recorded["participants"].items():
                count = int((reference.participant_id == participant).sum())
                if count != metadata["eligible_epochs"]:
                    raise ValueError("Prediction counts differ from recorded eligible coverage")


    @staticmethod
    def _score(frame: pd.DataFrame) -> dict[str, Any]:
        """Compute pooled stage metrics from the aligned saved predictions.

        Args:
            frame (pd.DataFrame): Predictions with target and prediction stage codes.
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            frame.target, frame.prediction, labels=CLASS_LABELS, zero_division=0,
        )

        return {
            "accuracy": float(accuracy_score(frame.target, frame.prediction)),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": confusion_matrix(
                frame.target, frame.prediction, labels=CLASS_LABELS,
            ).tolist(),
            "classes": {str(label): {"precision": float(precision[index]),
                                     "recall": float(recall[index]), "f1": float(f1[index]),
                                     "support": int(support[index])}
                        for index, label in enumerate(CLASS_LABELS)},
        }


    def build(self) -> dict[str, Any]:
        """Return descriptive paired results without training or altering artifacts."""
        reference = self._load_summary(self.reference_summary)
        candidate = self._load_summary(self.candidate_summary)
        self._validate_plans(reference["plan"], candidate["plan"])
        reference_results = {result["fold"]: result for result in reference["results"]}
        candidate_results = {result["fold"]: result for result in candidate["results"]}
        paired = []
        participants = []
        reference_frames = []
        candidate_frames = []

        for fold in reference["plan"]["folds"]:
            name = fold["name"]
            left = self._load_fold(reference_results[name], self.reference_summary)
            right = self._load_fold(candidate_results[name], self.candidate_summary)
            self._validate_recipes(left["metrics"], right["metrics"])
            if self.comparison in ("training_budget", "context"):
                self._validate_budget_limit(reference["plan"], left["metrics"])
                self._validate_budget_limit(candidate["plan"], right["metrics"])
            self._validate_population(
                fold, left["predictions"], right["predictions"],
                left["metrics"], right["metrics"],
            )
            row = self._compare_fold(name, left, right)
            paired.append(row)
            participants.extend(self._compare_participants(name, left, right))
            reference_frames.append(left["predictions"])
            candidate_frames.append(right["predictions"])

        report = self._summarize(reference_frames, candidate_frames, paired, participants)
        if self.comparison == "training_budget":
            report["comparison"] = "training_budget"
            report["selection_caveat"] = (
                "A larger budget offers more validation checkpoint selection opportunities"
            )
        elif self.comparison == "context":
            report["comparison"] = "context"
            report["capacity_caveat"] = (
                "Longer context also enlarges the learned positional table"
            )

        return report


    def _compare_fold(
        self, name: str, reference: dict[str, Any], candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare verified checkpoint scores with recomputed prediction scores.

        Args:
            name (str): Frozen fold identifier.
            reference (dict[str, Any]): Verified reference metrics and predictions.
            candidate (dict[str, Any]): Verified candidate metrics and predictions.
        """
        scores = [self._score(run["predictions"]) for run in (reference, candidate)]
        for run, score in zip((reference, candidate), scores):
            for metric in ("macro_f1", "accuracy"):
                if not math.isclose(score[metric], run["metrics"][metric], abs_tol=1e-10):
                    raise ValueError("Metrics do not match the saved validation predictions")

        return {"fold": name, "reference_macro_f1": scores[0]["macro_f1"],
                "candidate_macro_f1": scores[1]["macro_f1"],
                "macro_f1_difference": scores[1]["macro_f1"] - scores[0]["macro_f1"],
                "reference_accuracy": scores[0]["accuracy"],
                "candidate_accuracy": scores[1]["accuracy"],
                "accuracy_difference": scores[1]["accuracy"] - scores[0]["accuracy"]}


    def _compare_participants(
        self, fold: str, reference: dict[str, Any], candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Compare each participant using its aligned saved predictions.

        Args:
            fold (str): Fold containing the held-out participant.
            reference (dict[str, Any]): Verified reference predictions and metrics.
            candidate (dict[str, Any]): Verified candidate predictions and metrics.
        """
        rows = []
        for participant, frame in reference["predictions"].groupby("participant_id"):
            other = candidate["predictions"].loc[
                candidate["predictions"].participant_id == participant
            ]
            left, right = self._score(frame), self._score(other)
            rows.append({"participant_id": participant, "fold": fold,
                         "reference": left, "candidate": right,
                         "macro_f1_difference": right["macro_f1"] - left["macro_f1"],
                         "accuracy_difference": right["accuracy"] - left["accuracy"]})

        return rows


    def _summarize(
        self, reference_frames: list[pd.DataFrame], candidate_frames: list[pd.DataFrame],
        folds: list[dict[str, Any]], participants: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Pool aligned predictions and describe paired fold variation.

        Args:
            reference_frames (list[pd.DataFrame]): Held-out predictions per reference fold.
            candidate_frames (list[pd.DataFrame]): Held-out predictions per candidate fold.
            folds (list[dict[str, Any]]): Paired selected-checkpoint fold scores.
            participants (list[dict[str, Any]]): Paired participant scores.
        """
        reference = pd.concat(reference_frames, ignore_index=True)
        candidate = pd.concat(candidate_frames, ignore_index=True)
        if reference.duplicated(["participant_id", "time"]).any():
            raise ValueError("Validation epochs repeat across folds")
        differences = [fold["macro_f1_difference"] for fold in folds]

        return {"folds": folds, "participants": participants,
                "reference": self._score(reference), "candidate": self._score(candidate),
                "paired": {"mean_macro_f1_difference": mean(differences),
                           "sample_std_macro_f1_difference": stdev(differences)},
                "eligible_epochs": len(reference), "validation_participants": len(participants),
                "independent_outer_test": False,
                "interpretation": "Descriptive development comparison; no significance claim"}


def main() -> None:
    """Write a validated paired comparison from completed grouped experiments."""
    import argparse
    import logging

    from sleep_classifier.logging_utils import configure_logging

    parser = argparse.ArgumentParser(description="Compare matched grouped model experiments")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", choices=("architecture", "training_budget", "context"),
                        default="architecture")
    args = parser.parse_args()
    if args.output.resolve() in (args.reference.resolve(), args.candidate.resolve()):
        parser.error("Comparison output must not replace either source summary")

    configure_logging()
    report = PairedArchitectureReport(args.reference, args.candidate, args.comparison).build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    logging.getLogger(__name__).info("Paired comparison saved: %s; differences=%s",
                                     args.output, report["paired"])


if __name__ == "__main__":
    main()
