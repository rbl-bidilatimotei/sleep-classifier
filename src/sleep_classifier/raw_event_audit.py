import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sleep_classifier.bundle_export_source import EvaluationExportSource
from sleep_classifier.bundle_publication import publish_directory
from sleep_classifier.data import sha256
from sleep_classifier.features import EpochFeatureBuilder
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.motion_grid import FixedLagMotionFeatures
from sleep_classifier.recorded_event_replay import RecordedEventReplay
from sleep_classifier.sensor_audit_source import DevelopmentSensorSource
from sleep_classifier.sensor_epochs import SessionEpochClock
from sleep_classifier.sensor_only_replay import SensorOnlyReplayStudy
from sleep_classifier.sensor_replay_inference import FrozenSensorStream
from sleep_classifier.sensor_replay_inputs import SENSOR_COLUMNS, SensorReplayInputs
from sleep_classifier.sensor_replay_report import SensorReplayReport
from sleep_classifier.streaming_inference import StreamingEpochPredictor

LOGGER = logging.getLogger(__name__)


class RawEventAudit:
    """Compare bounded raw-event preparation with verified batch features and fixed predictions."""

    def __init__(self, config_path: Path) -> None:
        """Resolve the fixed replay source, chunking experiments, and immutable destination.

        Args:
            config_path (Path): YAML file for the development-only raw adapter audit.
        """
        self.config_path = config_path.resolve()
        data = self.config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(data).hexdigest()
        values = yaml.safe_load(data)
        if not isinstance(values, dict) or set(values) != {
            "replay_config", "output_dir", "chunk_sizes", "max_buffered_events",
        }:
            raise ValueError("Raw adapter audit requires exactly the documented settings")
        for key in ("replay_config", "output_dir"):
            if not isinstance(values[key], str) or not values[key].strip():
                raise ValueError("Raw audit paths must be nonempty strings")
        capacity, chunks = values["max_buffered_events"], values["chunk_sizes"]
        if (type(capacity) is not int or capacity <= 0 or not isinstance(chunks, list)
                or len(chunks) < 2
                or any(type(size) is not int or not 0 < size <= capacity for size in chunks)
                or len(chunks) != len(set(chunks))):
            raise ValueError("At least two unique positive chunk sizes must fit the buffer limit")
        self.capacity, self.chunk_sizes = capacity, chunks
        self.replay = SensorOnlyReplayStudy(self.config_path.parent / values["replay_config"])
        self.output_dir = (self.config_path.parent / values["output_dir"]).resolve()
        export_source = EvaluationExportSource(self.replay.chronological.config_path, "fold-01")
        roots = export_source.protected_roots()
        for root in (*roots, self.replay.output_dir, self.replay.audit.output_dir,
                     self.replay.audit.raw_data_dir, self.replay.audit.prepared_dir,
                     self.config_path.parent, Path(__file__).resolve().parent):
            if self.output_dir.is_relative_to(root) or root.is_relative_to(self.output_dir):
                raise ValueError("Raw audit output must be separate from immutable inputs and code")


    def _source(self) -> DevelopmentSensorSource:
        """Build the frozen development allowlist without opening any sensor observations."""
        audit = self.replay.audit
        return DevelopmentSensorSource(audit.prepared_dir, audit.raw_data_dir, audit.split_path)


    def plan(self) -> dict[str, Any]:
        """Verify frozen sources and declare exact clock, lateness, completion, and buffer rules."""
        if sha256(self.config_path) != self.config_sha256:
            raise ValueError("Raw audit configuration changed after initialization")
        report = self.replay.read_report()
        if report["status"] != "passed":
            raise ValueError("Raw adapter audit requires the passed sensor-only frozen replay")
        source = self._source()
        development = {person for fold in report["plan"]["folds"] for person in fold["validation"]}
        frozen = self.replay.chronological.reference.plan()
        if (set(source.partitions) != development
                or set(source.reserved) != set(frozen["reserved_test"])):
            raise ValueError("Raw sensor membership differs from the frozen replay cohort")
        return {"schema_version": 1, "config_sha256": self.config_sha256,
                "replay_plan": report["plan"],
                "replay_manifest_sha256": sha256(self.replay.output_dir / "manifest.json"),
                "participants": sorted(source.partitions), "reserved_test": source.reserved,
                "chunk_sizes": self.chunk_sizes, "max_buffered_events": self.capacity,
                "session_start_seconds": self.replay.audit.clock.start_seconds,
                "session_end_policy": "earlier_final_sensor_timestamp",
                "watermark_policy": "no_future_events_strictly_before_watermark",
                "late_event_policy": "reject_without_state_change",
                "tail_policy": "discard_incomplete_epoch_and_count_events",
                "canonical_prediction_chunk_size": self.chunk_sizes[0],
                "expected_prediction_epochs": report["pooled"]["scopes"]["candidate_epochs"],
                "device": self.replay.device, "training_performed": False,
                "test_observations_read": False}


    def validate_run_start(self) -> dict[str, Any]:
        """Require verified sources, the measured backend, and an unused destination."""
        if self.output_dir.exists():
            raise FileExistsError("Raw audit exists; inspect it or select a fresh output path")
        plan = self.plan()
        StreamingEpochPredictor._resolve_device(self.replay.device)
        return plan


    def _batch(
        self, heart: pd.DataFrame, motion: pd.DataFrame, rule: dict[str, float],
    ) -> pd.DataFrame:
        """Build the unchanged full-session numerical reference without labels.

        Args:
            heart (pd.DataFrame): Verified development heart-rate observations.
            motion (pd.DataFrame): Verified development acceleration observations.
            rule (dict[str, float]): Frozen coverage thresholds.
        """
        clock = SessionEpochClock(self.replay.audit.clock.start_seconds)
        epochs = clock.completed_epochs(heart, motion)
        batch = EpochFeatureBuilder(rule).build_sensor_epochs(heart, motion, epochs)
        grid = FixedLagMotionFeatures().motion_summaries(motion, epochs)
        return batch.merge(grid, on="time", validate="one_to_one")


    @staticmethod
    def _feature_parity(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
        """Require identical schedules, coverage decisions, segments, and numeric summaries.

        Args:
            actual (pd.DataFrame): Completed outputs from one raw-event chunking run.
            expected (pd.DataFrame): Full-session batch reference or historical sensor projection.
        """
        if len(actual) != len(expected) or not np.array_equal(actual.time, expected.time):
            raise ValueError("Raw adapter epoch schedule differs from batch preparation")
        if not {"available_at", "watermark_at_emission"} <= set(actual):
            raise ValueError("Raw adapter output lacks explicit completion and emission times")
        available = actual.available_at.to_numpy(float)
        watermarks = actual.watermark_at_emission.to_numpy(float)
        if (not np.isfinite(available).all() or not np.isfinite(watermarks).all()
                or not np.isclose(available, actual.time.to_numpy(float)+30,
                                  rtol=0, atol=1e-8).all()
                or (watermarks < available).any()):
            raise ValueError("Raw adapter emitted an epoch before its completion watermark")
        maximum, mismatches = 0.0, 0
        for column in set(expected)-{"eligible", "segment_id", "participant_id"}:
            left, right = actual[column].to_numpy(float), expected[column].to_numpy(float)
            matches = np.isclose(left, right, rtol=0, atol=1e-8, equal_nan=True)
            mismatches += int((~matches).sum())
            finite = np.isfinite(left) & np.isfinite(right)
            maximum = max(maximum, float(np.max(np.abs(left[finite]-right[finite]), initial=0)))
        eligibility = np.array_equal(actual.eligible, expected.eligible)
        segments = np.array_equal(actual.segment_id, expected.segment_id)
        if mismatches or not eligibility or not segments:
            raise ValueError("Raw adapter summaries, eligibility, or resets differ from batch")
        return {"feature_maximum_difference": maximum, "feature_mismatches": mismatches,
                "eligibility_matches": bool(eligibility), "segments_match": bool(segments)}


    def _participant(
        self, person: str, source: DevelopmentSensorSource, historical: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Replay two chunkings and compare every epoch before restoring a model.

        Args:
            person (str): Frozen development participant identifier.
            source (DevelopmentSensorSource): Guarded raw sensor reader.
            historical (pd.DataFrame): Verified notebook-44 sensor summaries with unused labels.
        """
        heart, motion = source.load_raw(person, "heart_rate"), source.load_raw(person, "motion")
        end = float(min(heart.time.max(), motion.time.max()))
        records, canonical = [], pd.DataFrame()
        batch = None
        for chunk in self.chunk_sizes:
            started = perf_counter()
            driver = RecordedEventReplay(source.rule, self.replay.audit.clock.start_seconds,
                                         self.capacity)
            actual, status = driver.run(person, heart, motion, chunk, end)
            if batch is None:
                batch = self._batch(heart, motion, source.rule)
                saved = historical.loc[historical.participant_id.eq(person)].set_index("time")
                historical_columns = [name for name in batch if name in saved.columns]
                saved = saved.loc[batch.time, historical_columns].reset_index()
            parity = self._feature_parity(actual, batch)
            historical_parity = self._feature_parity(actual, saved)
            consumed = int(actual.heart_sample_count.sum()+actual.motion_sample_count.sum())
            conserved = status["input_events"] == (consumed+status["duplicate_events"]
                                                    +status["discarded_tail_events"])
            if not conserved:
                raise ValueError("Raw adapter accepted event conservation failed")
            if status["buffered_events"] or status["peak_buffered_events"] > self.capacity:
                raise ValueError("Raw adapter buffer did not stay bounded or close cleanly")
            records.append({"participant_id": person, "chunk_size": chunk,
                            "epochs": len(actual), "eligible_epochs": int(actual.eligible.sum()),
                            **parity, **status, "elapsed_seconds": perf_counter()-started,
                            "historical_feature_maximum_difference": historical_parity[
                                "feature_maximum_difference"],
                            "historical_feature_mismatches": historical_parity[
                                "feature_mismatches"],
                            "event_conservation_matches": conserved})
            if chunk == self.chunk_sizes[0]:
                canonical = actual
            LOGGER.info("Raw adapter participant=%s chunk=%d epochs=%d peak_buffer=%d",
                        person, chunk, len(actual), status["peak_buffered_events"])
        return canonical, records


    def _predictions(self, rows: pd.DataFrame) -> list[dict[str, Any]]:
        """Compare canonical raw-event predictions with the frozen held-out predictions.

        Args:
            rows (pd.DataFrame): Label-free canonical adapter outputs for every development session.
        """
        saved = pd.read_csv(self.replay.output_dir / "candidate-predictions.csv",
                            dtype={"participant_id": str})
        report = self.replay.read_report()
        records = []
        for fold in report["plan"]["folds"]:
            source = EvaluationExportSource(self.replay.chronological.config_path, fold["name"])
            snapshot = source.snapshot()
            checkpoint, _ = source.checkpoint(snapshot)
            metadata = self.replay._metadata_digest(checkpoint)
            candidate = SensorReplayInputs.candidate(
                rows, fold["validation"], checkpoint["motion_features"]["imputation_mean"],
            )
            digest = snapshot["fold"]["checkpoint_sha256"]
            stream = FrozenSensorStream(checkpoint, digest, self.replay.device)
            actual = stream.predict(candidate, f"{fold['name']} raw adapter")
            expected = saved.loc[saved.participant_id.isin(fold["validation"])].sort_values(
                ["participant_id", "time"], kind="stable",
            ).reset_index(drop=True)
            parity = self._prediction_parity(actual, expected)
            integrity = stream.integrity()
            if self.replay._metadata_digest(checkpoint) != metadata:
                raise ValueError("Saved fitted metadata changed during raw adapter replay")
            records.append({"fold": fold["name"], "checkpoint_sha256": digest, **parity,
                            **integrity, "saved_transforms_unchanged": True})
        return records


    @staticmethod
    def _prediction_parity(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
        """Compare label-free probabilities and state boundaries on every predicted epoch.

        Args:
            actual (pd.DataFrame): Fresh canonical adapter predictions.
            expected (pd.DataFrame): Verified notebook-45 sensor-only predictions without labels.
        """
        SensorReplayReport._validated(actual, "candidate")
        SensorReplayReport._validated(expected, "candidate")
        keys = ["participant_id", "time"]
        if not np.array_equal(actual[keys].to_numpy(), expected[keys].to_numpy()):
            raise ValueError("Raw adapter prediction identities differ from the fixed replay")
        columns = [f"probability_{stage}" for stage in range(4)]
        values, reference = actual[columns].to_numpy(float), expected[columns].to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(reference).all():
            raise ValueError("Raw adapter prediction probabilities must be finite")
        difference = float(np.abs(values-reference).max())
        disagreements = int(actual.prediction.ne(expected.prediction).sum())
        lengths = np.array_equal(actual.history_epochs, expected.history_epochs)
        resets = np.array_equal(actual.reset_reason.fillna("").to_numpy(str),
                                expected.reset_reason.fillna("").to_numpy(str))
        if difference > 2e-6 or disagreements or not lengths or not resets:
            raise ValueError("Raw adapter predictions or resets differ from the fixed replay")
        return {"epochs": len(actual), "maximum_probability_difference": difference,
                "class_disagreements": disagreements, "history_lengths_match": bool(lengths),
                "reset_reasons_match": bool(resets)}


    def run(self) -> Path:
        """Publish a development audit only after every chunking and frozen fold passes."""
        plan = self.validate_run_start()
        source = self._source()
        historical = pd.read_csv(
            self.replay.audit.output_dir/"rows.csv", usecols=list(SENSOR_COLUMNS)+[
                f"{sensor}_{field}" for sensor in ("heart", "motion") for field in (
                    "sample_count", "occupied_bin_fraction", "largest_observation_gap")],
            dtype={"participant_id": str},
        )
        frames, records = [], []
        for index, person in enumerate(plan["participants"], start=1):
            LOGGER.info("Starting raw adapter session %d/%d id=%s", index,
                        len(plan["participants"]), person)
            frame, participant_records = self._participant(person, source, historical)
            frames.append(frame)
            records.extend(participant_records)
        predictions = self._predictions(pd.concat(frames, ignore_index=True))
        report = {"status": "passed", "plan": plan, "participants": records,
                  "prediction_parity": predictions, "raw_input_hashes": source.input_hashes,
                  "training_performed": False, "test_observations_read": False,
                  "scope": "Adapter state is bounded; the offline audit consumer retains outputs"}
        if self.plan() != plan:
            raise ValueError("Raw audit sources or configuration changed during replay")
        self._validate_report(report)
        self._publish(report)
        return self.output_dir


    def _publish(self, report: dict[str, Any]) -> None:
        """Publish immutable evidence with source, output, and implementation identities.

        Args:
            report (dict[str, Any]): Full measured adapter audit and fixed-model parity evidence.
        """
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".raw-adapter-", dir=self.output_dir.parent) as temporary:
            staging = Path(temporary)/"snapshot"
            staging.mkdir()
            (staging/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            module = Path(__file__).resolve().parent
            manifest = {"schema_version": 1, "report_sha256": sha256(staging/"report.json"),
                        "implementation_hashes": {name: sha256(module/name) for name in (
                            "raw_event_adapter.py", "recorded_event_replay.py",
                            "raw_event_audit.py",
                        )}}
            (staging/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            publish_directory(staging, self.output_dir)


    def read_report(self) -> dict[str, Any]:
        """Verify cached audit evidence without rereading raw data or restoring any model."""
        if not self.output_dir.exists():
            return {"status": "pending", "participants": [], "prediction_parity": []}
        root = self.output_dir
        manifest_path = DevelopmentSensorSource._inside(root/"manifest.json", root)
        if not manifest_path.exists():
            return {"status": "partial", "participants": [], "prediction_parity": []}
        manifest = json.loads(manifest_path.read_text())
        path = DevelopmentSensorSource._inside(root/"report.json", root)
        if manifest.get("schema_version") != 1 or sha256(path) != manifest["report_sha256"]:
            raise ValueError("Raw adapter evidence version or checksum differs")
        report = json.loads(path.read_text())
        if report["plan"] != self.plan() or report["status"] != "passed":
            raise ValueError("Cached raw adapter source identities or status differ")
        self._validate_report(report)
        return report


    @staticmethod
    def _validate_report(report: dict[str, Any]) -> None:
        """Require every participant/chunk and fixed fold to satisfy the declared acceptance rules.

        Args:
            report (dict[str, Any]): Measured or cached complete raw-event audit report.
        """
        if (report.get("status") != "passed" or report.get("training_performed") is not False
                or report.get("test_observations_read") is not False):
            raise ValueError("Raw audit evidence must exclude training and final-test observations")
        plan = report["plan"]
        expected = {(person, chunk) for person in plan["participants"]
                    for chunk in plan["chunk_sizes"]}
        records = report["participants"]
        names = [(row["participant_id"], row["chunk_size"]) for row in records]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Raw audit must cover every participant and chunk exactly once")
        for row in records:
            maximum = row.get("feature_maximum_difference")
            historical = row.get("historical_feature_maximum_difference")
            if (not isinstance(maximum, (int, float)) or isinstance(maximum, bool)
                    or not math.isfinite(maximum) or not 0 <= maximum <= 1e-8
                    or not isinstance(historical, (int, float)) or isinstance(historical, bool)
                    or not math.isfinite(historical) or not 0 <= historical <= 1e-8
                    or row.get("historical_feature_mismatches") != 0
                    or row.get("event_conservation_matches") is not True
                    or row.get("feature_mismatches") != 0
                    or row.get("eligibility_matches") is not True
                    or row.get("segments_match") is not True
                    or row.get("closed") is not True or row.get("buffered_events") != 0
                    or not 0 <= row["peak_buffered_events"] <= plan["max_buffered_events"]
                    or row["epochs"] <= 0 or row["emitted_epochs"] != row["epochs"]):
                raise ValueError("Raw audit feature, reset, closure, or buffer conditions failed")
        RawEventAudit._validate_predictions(report)


    @staticmethod
    def _validate_predictions(report: dict[str, Any]) -> None:
        """Check canonical prediction coverage and all unchanged-model parity conditions.

        Args:
            report (dict[str, Any]): Complete raw-event audit with its verified replay plan.
        """
        plan, rows = report["plan"], report["prediction_parity"]
        folds = {fold["name"] for fold in plan["replay_plan"]["folds"]}
        if len(rows) != len(folds) or {row["fold"] for row in rows} != folds:
            raise ValueError("Raw audit must replay every frozen fold exactly once")
        total = sum(row["epochs"] for row in rows)
        if total != plan["expected_prediction_epochs"]:
            raise ValueError("Raw audit predicted population differs from the fixed replay")
        for row in rows:
            source = plan["replay_plan"]["chronological_plan"]["sources"][row["fold"]]
            difference = row.get("maximum_probability_difference")
            if (row["checkpoint_sha256"] != source["model_checkpoint.pt"]
                    or not isinstance(difference, (int, float)) or isinstance(difference, bool)
                    or not math.isfinite(difference) or not 0 <= difference <= 2e-6
                    or row.get("class_disagreements") != 0
                    or any(row.get(key) is not True for key in (
                        "history_lengths_match", "reset_reasons_match", "weights_unchanged",
                        "saved_transforms_unchanged", "evaluation_mode"))
                    or row.get("gradients_created") is not False
                    or row.get("training_performed") is not False):
                raise ValueError("Raw audit frozen-model identity or parity conditions failed")


def main() -> None:
    """Inspect the raw adapter audit plan or run the explicitly requested development replay."""
    parser = argparse.ArgumentParser(description="Audit bounded raw-event preparation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    audit = RawEventAudit(args.config)
    LOGGER.info("Raw event audit: %s", audit.run() if args.run else audit.plan())


if __name__ == "__main__":
    main()
