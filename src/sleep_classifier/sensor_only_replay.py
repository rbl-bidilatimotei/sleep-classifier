import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
import yaml

from sleep_classifier.bundle_export_source import EvaluationExportSource
from sleep_classifier.bundle_publication import publish_directory
from sleep_classifier.chronological_replay import ChronologicalReplayStudy
from sleep_classifier.data import sha256
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.sensor_audit_source import DevelopmentSensorSource
from sleep_classifier.sensor_preparation_audit import LabelFreeSensorAudit
from sleep_classifier.sensor_replay_inference import FrozenSensorStream
from sleep_classifier.sensor_replay_inputs import SensorReplayInputs
from sleep_classifier.sensor_replay_report import SensorReplayReport
from sleep_classifier.streaming_inference import StreamingEpochPredictor

LOGGER = logging.getLogger(__name__)


class SensorOnlyReplayStudy:
    """Measure sensor-only resets with frozen held-out models and paired development targets."""

    def __init__(self, config_path: Path) -> None:
        """Resolve immutable audit inputs and an explicit validated inference backend.

        Args:
            config_path (Path): YAML file for the frozen sensor-only replay study.
        """
        self.config_path = config_path.resolve()
        data = self.config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(data).hexdigest()
        settings = yaml.safe_load(data)
        if (not isinstance(settings, dict) or set(settings) != {
                "sensor_audit_config", "chronological_config", "output_dir", "device"}):
            raise ValueError("Sensor replay requires exactly the documented settings")
        self.settings = settings
        for name in ("sensor_audit_config", "chronological_config", "output_dir"):
            if not isinstance(settings[name], str) or not settings[name].strip():
                raise ValueError("Replay paths must be nonempty strings")
        self.audit = LabelFreeSensorAudit(self.config_path.parent / settings["sensor_audit_config"])
        self.chronological = ChronologicalReplayStudy(
            self.config_path.parent / settings["chronological_config"],
        )
        if settings["device"] != self.chronological.settings["device"]:
            raise ValueError("Sensor replay must retain the frozen chronological backend")
        self.device = settings["device"]
        self.output_dir = (self.config_path.parent / settings["output_dir"]).resolve()
        protected = EvaluationExportSource(
            self.chronological.config_path, "fold-01",
        ).protected_roots()
        for directory in (*protected, self.audit.output_dir, self.audit.prepared_dir,
                          self.audit.raw_data_dir, self.config_path.parent,
                          Path(__file__).resolve().parent):
            if (self.output_dir.is_relative_to(directory)
                    or directory.is_relative_to(self.output_dir)):
                raise ValueError("Replay output must be separate from immutable sources and code")


    def _sensor_snapshot(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Verify preparation output identities without parsing epoch observations."""
        root = self.audit.output_dir
        manifest_path = DevelopmentSensorSource._inside(root / "manifest.json", root)
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema_version") != 1
                or manifest["config_sha256"] != sha256(self.audit.config_path)):
            raise ValueError("Sensor snapshot version or configuration differs")
        hashes = {"manifest.json": sha256(manifest_path)}
        for name in ("summary.json", "rows.csv"):
            path = DevelopmentSensorSource._inside(root / name, root)
            hashes[name] = sha256(path)
            if hashes[name] != manifest["outputs"][name]:
                raise ValueError("Sensor preparation snapshot checksum differs")
        summary = json.loads((root / "summary.json").read_text())
        if summary["status"] != "passed" or any(summary["plan"][name] for name in (
            "training_performed", "test_observations_read", "raw_labels_read", "model_loaded",
        )):
            raise ValueError("Replay requires a passed development-only sensor preparation audit")
        return summary, hashes


    def plan(self) -> dict[str, Any]:
        """Verify source hashes and fold membership without restoring any model weights."""
        if sha256(self.config_path) != self.config_sha256:
            raise ValueError("Replay configuration changed after initialization")
        report = self.chronological.read_report()
        if report["status"] != "passed":
            raise ValueError("Sensor replay requires the passed full chronological control")
        sensor, hashes = self._sensor_snapshot()
        folds = self.chronological.reference.plan()
        if (set(sensor["plan"]["development_participants"]) != set(folds["development"])
                or set(sensor["plan"]["reserved_test_participants"])
                != set(folds["reserved_test"])):
            raise ValueError("Sensor and frozen-model development membership differ")
        return {"schema_version": 1, "config_sha256": self.config_sha256,
                "sensor_snapshot": hashes,
                "chronological_report_sha256": sha256(
                    self.chronological.output_dir / "report.json"),
                "chronological_plan": report["plan"], "folds": folds["folds"],
                "device": self.device, "training_performed": False,
                "test_observations_read": False,
                "absolute_tolerance": self.chronological.settings["absolute_tolerance"]}


    def validate_run_start(self) -> dict[str, Any]:
        """Require verified inputs, the explicit backend, and an unused output path."""
        if self.output_dir.exists():
            raise FileExistsError("Frozen replay exists; inspect it or select a fresh output path")
        plan = self.plan()
        StreamingEpochPredictor._resolve_device(self.device)
        return plan


    def _run_fold(
        self, fold: dict[str, Any], audit_rows: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Predict candidate rows before opening labeled control observations.

        Args:
            fold (dict[str, Any]): Frozen fold name and disjoint participant membership.
            audit_rows (pd.DataFrame): Verified cached raw summaries and sensor-only segment fields.
        """
        source = EvaluationExportSource(self.chronological.config_path, fold["name"])
        snapshot = source.snapshot()
        checkpoint, _ = source.checkpoint(snapshot)
        metadata = self._metadata_digest(checkpoint)
        candidate = SensorReplayInputs.candidate(
            audit_rows, fold["validation"], checkpoint["motion_features"]["imputation_mean"],
        )
        digest = snapshot["fold"]["checkpoint_sha256"]
        stream = FrozenSensorStream(checkpoint, digest, self.device)
        predictions = stream.predict(candidate, f"{fold['name']} sensor-only")
        reference = SensorReplayInputs.reference(source.fold_dir, fold["validation"], checkpoint)
        saved_path = self.chronological.output_dir / f"{fold['name']}.csv"
        saved = pd.read_csv(saved_path, dtype={"participant_id": str}).sort_values(
            ["participant_id", "time"], kind="stable",
        ).reset_index(drop=True)
        control = stream.predict(reference, f"{fold['name']} legacy control")
        parity = stream.control_parity(control, saved, snapshot["fold"]["absolute_tolerance"])
        integrity = stream.integrity()
        if self._metadata_digest(checkpoint) != metadata:
            raise ValueError("Frozen preprocessing metadata changed during replay")
        integrity["saved_transforms_unchanged"] = True
        integrity["preprocessing_sha256"] = metadata
        report = {"fold": fold["name"], "checkpoint_sha256": digest,
                  "control": parity, "integrity": integrity,
                  "comparison": SensorReplayReport.build(predictions, saved)}
        return predictions, saved, report


    @staticmethod
    def _metadata_digest(checkpoint: dict[str, Any]) -> str:
        """Identify fitted transforms and constructor settings independently of model weights.

        Args:
            checkpoint (dict[str, Any]): Frozen weights and JSON-compatible fitted metadata.
        """
        values = {key: value for key, value in checkpoint.items() if key != "state_dict"}
        data = json.dumps(values, sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()


    def run(self) -> Path:
        """Publish a verified replay snapshot without fitting or changing any source artifact."""
        plan = self.validate_run_start()
        rows, _ = self.audit.load_report()
        development = {person for fold in plan["folds"] for person in fold["validation"]}
        if set(rows.participant_id) != development:
            raise ValueError("Sensor rows must contain exactly the frozen development cohort")
        candidates, references, results = [], [], []
        for fold in plan["folds"]:
            LOGGER.info("Starting frozen replay fold=%s device=%s", fold["name"], self.device)
            candidate, reference, result = self._run_fold(fold, rows)
            candidates.append(candidate.assign(fold=fold["name"]))
            references.append(reference.assign(fold=fold["name"]))
            results.append(result)
        candidate = pd.concat(candidates, ignore_index=True)
        reference = pd.concat(references, ignore_index=True)
        report = {"status": "passed", "plan": plan, "folds": results,
                  "pooled": SensorReplayReport.build(candidate, reference),
                  "training_performed": False, "test_observations_read": False,
                  "scope": "Cached raw-sensor summaries; sensor-only eligibility and resets"}
        self._validate_report(report)
        if self.plan() != plan:
            raise ValueError("Replay sources or settings changed during inference")
        self._publish(candidate, reference, report)
        return self.output_dir


    def _publish(
        self, candidate: pd.DataFrame, reference: pd.DataFrame, report: dict[str, Any],
    ) -> None:
        """Publish predictions and evidence with no replacement of existing snapshots.

        Args:
            candidate (pd.DataFrame): Label-free predictions, including unscored epochs.
            reference (pd.DataFrame): Separate frozen scored control predictions.
            report (dict[str, Any]): Paired scores, control parity, and model-state evidence.
        """
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".sensor-replay-", dir=self.output_dir.parent) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            candidate.to_csv(staging / "candidate-predictions.csv", index=False)
            reference.to_csv(staging / "reference-predictions.csv", index=False)
            (staging / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False)+"\n",
            )
            names = ("candidate-predictions.csv", "reference-predictions.csv", "report.json")
            module_dir = Path(__file__).resolve().parent
            manifest = {"schema_version": 1,
                        "outputs": {name: sha256(staging/name) for name in names},
                        "implementation_hashes": {name: sha256(module_dir/name) for name in (
                            "sensor_only_replay.py", "sensor_replay_inputs.py",
                            "sensor_replay_inference.py", "sensor_replay_report.py",
                            "streaming_inference.py", "model_factory.py",
                        )}}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            publish_directory(staging, self.output_dir)


    def read_report(self) -> dict[str, Any]:
        """Verify cached evidence without another inference pass or training operation."""
        if not self.output_dir.exists():
            return {"status": "pending", "folds": []}
        root = self.output_dir
        path = DevelopmentSensorSource._inside(root / "manifest.json", root)
        if not path.exists():
            return {"status": "partial", "folds": []}
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported replay snapshot version")
        for name in ("candidate-predictions.csv", "reference-predictions.csv", "report.json"):
            file = DevelopmentSensorSource._inside(root / name, root)
            if sha256(file) != manifest["outputs"][name]:
                raise ValueError("Cached frozen replay checksum differs")
        report = json.loads((root / "report.json").read_text())
        if report["plan"] != self.plan():
            raise ValueError("Cached frozen replay source identities changed")
        self._validate_report(report)
        return report


    @staticmethod
    def _validate_report(report: dict[str, Any]) -> None:
        """Require complete control and model-state evidence without an accuracy target.

        Args:
            report (dict[str, Any]): Measured or cached full-study replay report.
        """
        if (report.get("status") != "passed" or report.get("training_performed") is not False
                or report.get("test_observations_read") is not False):
            raise ValueError("Replay evidence must exclude training and final-test observations")
        plan = report["plan"]
        expected = {fold["name"] for fold in plan["folds"]}
        names = [fold["fold"] for fold in report["folds"]]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Replay evidence requires every frozen fold exactly once")
        for fold in report["folds"]:
            SensorOnlyReplayStudy._validate_fold_evidence(fold, plan)
        for field in ("candidate_epochs", "reference_epochs", "shared_scored_epochs",
                      "candidate_only_epochs", "reference_only_epochs"):
            total = sum(fold["comparison"]["scopes"][field] for fold in report["folds"])
            if report["pooled"]["scopes"][field] != total:
                raise ValueError("Pooled replay scope differs from the frozen fold totals")


    @staticmethod
    def _validate_fold_evidence(fold: dict[str, Any], plan: dict[str, Any]) -> None:
        """Check the source identity, legacy parity, and frozen-state acceptance conditions.

        Args:
            fold (dict[str, Any]): One fold's measured control and integrity evidence.
            plan (dict[str, Any]): Verified frozen study plan.
        """
        source = plan["chronological_plan"]["sources"][fold["fold"]]["model_checkpoint.pt"]
        control, integrity = fold["control"], fold["integrity"]
        difference = control.get("maximum_probability_difference")
        if (fold["checkpoint_sha256"] != source or control.get("status") != "passed"
                or isinstance(difference, bool) or not isinstance(difference, (int, float))
                or not math.isfinite(difference)
                or not 0 <= difference <= plan["absolute_tolerance"]
                or control.get("class_disagreements") != 0
                or control.get("history_lengths_match") is not True
                or control.get("reset_reasons_match") is not True):
            raise ValueError("Replay control identity or numerical parity failed")
        if (any(integrity.get(key) is not True for key in (
                "weights_unchanged", "evaluation_mode", "saved_transforms_unchanged"))
                or integrity.get("gradients_created") is not False
                or integrity.get("training_performed") is not False):
            raise ValueError("Replay model or fitted-transform integrity failed")
        scopes = fold["comparison"]["scopes"]
        if control["epochs"] != scopes["reference_epochs"] or scopes["shared_scored_epochs"] <= 0:
            raise ValueError("Replay scored scope must match the verified control")


def main() -> None:
    """Inspect the replay plan or explicitly run fixed-model inference."""
    parser = argparse.ArgumentParser(description="Replay frozen models with sensor-only history")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    study = SensorOnlyReplayStudy(args.config)
    LOGGER.info("Frozen sensor replay: %s", study.run() if args.run else study.plan())


if __name__ == "__main__":
    main()
