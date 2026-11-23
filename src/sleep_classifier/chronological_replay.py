import argparse
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from sleep_classifier.checkpoint_audit import CheckpointReplayAudit
from sleep_classifier.grouped_validation import GroupedValidationExperiment
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.neural_training import TrainingConfig
from sleep_classifier.streaming_inference import StreamingEpochPredictor

LOGGER = logging.getLogger(__name__)


class ChronologicalReplayAudit:
    """Compare bounded chronological inference with offline and saved validation predictions."""

    def __init__(self, fold_dir: Path, device: str, absolute_tolerance: float) -> None:
        """Bind a saved fold and an explicit numerical acceptance tolerance.

        Args:
            fold_dir (Path): Prepared development fold containing the selected checkpoint.
            device (str): Explicit inference backend without automatic fallback.
            absolute_tolerance (float): Maximum accepted absolute probability difference.
        """
        self.fold_dir = fold_dir.resolve()
        self.device = device
        self.absolute_tolerance = absolute_tolerance
        self.predictions = pd.DataFrame()


    @staticmethod
    def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
        """Measure probability discrepancies and changes in the predicted class.

        Args:
            left (np.ndarray): First ordered four-class probability matrix.
            right (np.ndarray): Second probability matrix on identical target identities.
        """
        return {
            "max_abs_probability_difference": float(np.abs(left - right).max()),
            "prediction_disagreements": int((left.argmax(1) != right.argmax(1)).sum()),
        }


    def _stream(self, frame: pd.DataFrame, predictor: StreamingEpochPredictor) -> np.ndarray:
        """Feed only the current epoch summary to the bounded inference interface.

        Args:
            frame (pd.DataFrame): Ordered validation summaries with evaluation metadata.
            predictor (StreamingEpochPredictor): Restored model with an initially empty history.
        """
        rows = []
        for index, row in enumerate(frame.to_dict("records")):
            start = perf_counter()
            result = predictor.predict_epoch(
                str(row["participant_id"]), int(row["segment_id"]), float(row["time"]),
                {name: float(row[name]) for name in predictor.feature_columns},
            )
            rows.append(result | {"target": int(row["target"]),
                                  "elapsed_seconds": perf_counter() - start})
            if (index + 1) % 1000 == 0:
                LOGGER.info("Chronological replay %s: %d/%d epochs", self.fold_dir.name,
                            index + 1, len(frame))
        self.predictions = pd.DataFrame(rows)
        return np.asarray(self.predictions.probabilities.tolist())


    def build(self) -> dict[str, Any]:
        """Replay one fold without fitting transforms, calculating gradients, or writing files."""
        replay = CheckpointReplayAudit(self.fold_dir, self.device)
        model, data, frame = replay._load()
        predictor = StreamingEpochPredictor(
            self.fold_dir / "results/model_checkpoint.pt", self.device,
        )
        before = {name: value.detach().cpu().clone()
                  for name, value in predictor.model.state_dict().items()}
        offline = replay._predict(model, data)
        saved = self._saved_probabilities(frame)
        streamed = self._stream(frame, predictor)
        expected_lengths = np.minimum(
            frame.groupby(["participant_id", "segment_id"], sort=False).cumcount() + 1,
            predictor.context_epochs,
        )
        comparisons = {"stream_vs_offline": self._difference(streamed, offline),
                       "offline_vs_saved": self._difference(offline, saved)}
        unchanged = all(torch.equal(before[name], value.detach().cpu())
                        for name, value in predictor.model.state_dict().items())
        history_matches = bool(np.array_equal(self.predictions.history_epochs, expected_lengths))
        gradients = any(parameter.grad is not None for parameter in predictor.model.parameters())
        passed = (all(row["max_abs_probability_difference"] <= self.absolute_tolerance
                      and row["prediction_disagreements"] == 0 for row in comparisons.values())
                  and unchanged and not gradients and history_matches)
        return self._report(frame, predictor, comparisons, passed, unchanged,
                            gradients, history_matches)


    def _saved_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Require saved predictions to identify exactly the replayed validation targets.

        Args:
            frame (pd.DataFrame): Ordered validation summaries from the prepared fold.
        """
        saved = pd.read_csv(self.fold_dir / "results/validation_predictions.csv",
                            dtype={"participant_id": str})
        saved = saved.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)
        if (len(saved) != len(frame)
                or not np.array_equal(saved.participant_id.to_numpy(dtype=str),
                                      frame.participant_id.to_numpy(dtype=str))
                or not np.array_equal(saved[["time", "target"]].to_numpy(),
                                      frame[["time", "target"]].to_numpy())):
            raise ValueError("Saved predictions differ from the replay validation identities")
        values = saved[[f"probability_{stage}" for stage in range(4)]].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(1), 1, atol=1e-6, rtol=0)
                or not np.array_equal(values.argmax(1), saved.prediction)):
            raise ValueError("Saved probabilities or class predictions are invalid")
        return values


    def _report(
        self, frame: pd.DataFrame, predictor: StreamingEpochPredictor,
        comparisons: dict[str, dict[str, float | int]], passed: bool, unchanged: bool,
        gradients: bool, history_matches: bool,
    ) -> dict[str, Any]:
        """Summarize numerical parity, history boundaries, and approximate replay timing.

        Args:
            frame (pd.DataFrame): Frozen validation target population.
            predictor (StreamingEpochPredictor): Model and checkpoint identity after replay.
            comparisons (dict[str, dict[str, float | int]]): Observed parity discrepancies.
            passed (bool): Whether every predeclared acceptance condition holds.
            unchanged (bool): Whether every model parameter and buffer remained unchanged.
            gradients (bool): Whether inference created parameter gradients.
            history_matches (bool): Whether every history length matches offline segmentation.
        """
        return {
            "fold": self.fold_dir.name, "status": "passed" if passed else "failed",
            "device": self.device, "epochs": len(frame),
            "participants": int(frame.participant_id.nunique()),
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "context_epochs": predictor.context_epochs, "features": list(predictor.feature_columns),
            "absolute_tolerance": self.absolute_tolerance, **comparisons,
            "history_lengths_match": history_matches, "weights_unchanged": unchanged,
            "gradients_created": gradients, "training_performed": False,
            "history_start_epochs": int((self.predictions.history_epochs == 1).sum()),
            "full_context_epochs": int((self.predictions.history_epochs
                                         == predictor.context_epochs).sum()),
            "reset_counts": self.predictions.reset_reason.dropna().value_counts().to_dict(),
            "observed_p50_epoch_seconds": float(self.predictions.elapsed_seconds.median()),
            "observed_p95_epoch_seconds": float(self.predictions.elapsed_seconds.quantile(.95)),
            "timing_caveat": "Python replay timings are descriptive, not a deployment benchmark",
        }


class ChronologicalReplayStudy:
    """Audit the frozen development checkpoints without another training experiment."""

    def __init__(self, config_path: Path) -> None:
        """Resolve the reference study, explicit backend, and fresh audit output.

        Args:
            config_path (Path): YAML file declaring the replay study.
        """
        self.config_path = config_path.resolve()
        self.settings = yaml.safe_load(self.config_path.read_text())
        required = {"reference_grouped_config", "output_dir", "device", "absolute_tolerance"}
        if not isinstance(self.settings, dict) or set(self.settings) != required:
            raise ValueError("Replay config must declare exactly the supported audit settings")
        tolerance = self.settings["absolute_tolerance"]
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not np.isfinite(tolerance) or not 0 < tolerance <= 1e-4):
            raise ValueError("Replay probability tolerance must be positive and at most 1e-4")
        if self.settings["device"] not in ("cpu", "mps", "cuda"):
            raise ValueError("Replay requires an explicit backend")
        self.reference = GroupedValidationExperiment(
            self.config_path.parent / self.settings["reference_grouped_config"],
        )
        self.output_dir = (self.config_path.parent / self.settings["output_dir"]).resolve()
        if (self.output_dir.is_relative_to(self.reference.output_dir)
                or self.reference.output_dir.is_relative_to(self.output_dir)):
            raise ValueError("Replay outputs must be separate from the saved model artifacts")


    @staticmethod
    def _hash(path: Path) -> str:
        """Hash an existing input or output without changing its bytes.

        Args:
            path (Path): Existing local artifact.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()


    def plan(self) -> dict[str, Any]:
        """Verify the completed reference without inference or final-test observations."""
        manifest = self.reference._validate_manifest()
        path = self.reference.output_dir / "summary.json"
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete" or summary.get("plan") != manifest["plan"]:
            raise ValueError("Replay requires a completed reference with the frozen fold plan")
        if manifest["device"] != self.settings["device"]:
            raise ValueError("Replay backend must match the measured reference backend")
        folds = manifest["plan"]["folds"]
        results = {row["fold"]: row for row in summary["results"]}
        if len(summary["results"]) != len(folds) or set(results) != {f["name"] for f in folds}:
            raise ValueError("Every frozen fold must have exactly one completed result")
        sources = {}
        for fold in folds:
            directory = self.reference.output_dir / fold["name"] / "results"
            metrics = directory / "metrics.json"
            registered = Path(results[fold["name"]]["metrics_path"])
            if registered.resolve() != metrics.resolve():
                raise ValueError("Registered metrics must belong to the frozen reference fold")
            if self._hash(metrics) != results[fold["name"]]["metrics_sha256"]:
                raise ValueError("Registered reference metrics hash differs")
            sources[fold["name"]] = {name: self._hash(directory / name) for name in (
                "metrics.json", "model_checkpoint.pt", "validation_predictions.csv",
            )}
        return {"schema_version": 1, "config_sha256": self._hash(self.config_path),
                "reference_manifest_sha256": self._hash(
                    self.reference.output_dir / "manifest.json",
                ),
                "reference_summary_sha256": self._hash(path), "sources": sources,
                "device": self.settings["device"], "folds": [fold["name"] for fold in folds],
                "development_participants": len(manifest["plan"]["development"]),
                "reserved_test_participants": len(manifest["plan"]["reserved_test"])}


    def validate_run_start(self) -> dict[str, Any]:
        """Check the source, device, and fresh destination before any output creation."""
        plan = self.plan()
        if self.output_dir.exists():
            raise FileExistsError("Replay snapshot exists; inspect it instead of overwriting")
        TrainingConfig(self.reference.model_settings).device()
        return plan


    def run(self) -> Path:
        """Write a fresh replay snapshot while preserving every source artifact."""
        plan = self.validate_run_start()
        self.output_dir.mkdir(parents=True)
        rows = []
        for fold in plan["folds"]:
            LOGGER.info("Replaying frozen checkpoint: %s device=%s", fold, plan["device"])
            audit = ChronologicalReplayAudit(self.reference.output_dir / fold, plan["device"],
                                              self.settings["absolute_tolerance"])
            row = audit.build()
            destination = self.output_dir / f"{fold}.csv"
            predictions = audit.predictions.drop(columns="probabilities").copy()
            values = np.asarray(audit.predictions.probabilities.tolist())
            for stage in range(4):
                predictions[f"probability_{stage}"] = values[:, stage]
            predictions.to_csv(destination, index=False)
            rows.append(row | {"predictions_sha256": self._hash(destination)})
        if self.plan() != plan:
            raise ValueError("Reference artifacts changed during the replay")
        status = "passed" if all(row["status"] == "passed" for row in rows) else "failed"
        report = {"status": status,
                  "plan": plan, "folds": rows, "training_performed": False,
                  "test_observations_read": False,
                  "scope": "Prepared epoch summaries only; raw sensor ingestion is not audited"}
        destination = self.output_dir / "report.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        return destination


    def read_report(self) -> dict[str, Any]:
        """Validate cached evidence against the current reference and saved prediction hashes."""
        path = self.output_dir / "report.json"
        if not path.exists():
            return {"status": "partial" if self.output_dir.exists() else "pending", "folds": []}
        report = json.loads(path.read_text())
        if report["plan"] != self.plan():
            raise ValueError("Cached replay references changed inputs or settings")
        names = [row["fold"] for row in report["folds"]]
        if len(names) != len(report["plan"]["folds"]) or set(names) != set(report["plan"]["folds"]):
            raise ValueError("Cached replay must contain every planned fold exactly once")
        expected_status = (
            "passed" if all(row["status"] == "passed" for row in report["folds"]) else "failed"
        )
        if report["status"] != expected_status:
            raise ValueError("Cached replay status differs from its fold outcomes")
        for row in report["folds"]:
            expected = report["plan"]["sources"][row["fold"]]["model_checkpoint.pt"]
            if (row["checkpoint_sha256"] != expected
                    or row["device"] != self.settings["device"]
                    or row["absolute_tolerance"] != self.settings["absolute_tolerance"]):
                raise ValueError("Cached replay checkpoint, backend, or tolerance differs")
            if self._hash(self.output_dir / f"{row['fold']}.csv") != row["predictions_sha256"]:
                raise ValueError("Cached replay predictions changed")
        return report


def main() -> None:
    """Run the explicitly requested inference audit without any training command."""
    parser = argparse.ArgumentParser(description="Audit chronological checkpoint inference parity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    configure_logging()
    study = ChronologicalReplayStudy(args.config)
    LOGGER.info("Chronological replay report: %s", study.run())


if __name__ == "__main__":
    main()
