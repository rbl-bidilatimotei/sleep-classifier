import argparse
import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sleep_classifier.bundle_publication import publish_directory
from sleep_classifier.data import sha256
from sleep_classifier.features import FEATURE_COLUMNS, EpochFeatureBuilder
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.motion_grid import FixedLagMotionFeatures
from sleep_classifier.sensor_audit_source import DevelopmentSensorSource
from sleep_classifier.sensor_epochs import SessionEpochClock

LOGGER = logging.getLogger(__name__)
COVERAGE_FIELDS = ("sample_count", "occupied_bin_fraction", "largest_observation_gap")
SUMMARY_COLUMNS = (*FEATURE_COLUMNS, *(f"{sensor}_{field}"
                   for sensor in ("heart", "motion") for field in COVERAGE_FIELDS))


class LabelFreeSensorAudit:
    """Audit raw development preparation without loading or modifying any prediction model."""

    def __init__(self, config_path: Path) -> None:
        """Resolve explicit inputs, session origin, and immutable output settings.

        Args:
            config_path (Path): YAML file for the sensor-only preparation audit.
        """
        self.config_path = config_path.resolve()
        config_bytes = self.config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        values = yaml.safe_load(config_bytes)
        paths = {"prepared_dir", "raw_data_dir", "split_path", "motion_snapshot_dir", "output_dir"}
        if not isinstance(values, dict) or set(values) != paths | {"session_start_seconds"}:
            raise ValueError("Sensor audit requires exactly the documented settings")
        for key in paths:
            if not isinstance(values[key], str) or not values[key].strip():
                raise ValueError("Sensor audit paths must be nonempty strings")
            setattr(self, key, (self.config_path.parent / values[key]).resolve())
        self.clock = SessionEpochClock(values["session_start_seconds"])
        for source in (self.prepared_dir, self.raw_data_dir, self.motion_snapshot_dir,
                       self.split_path, Path(__file__).resolve().parent):
            if self.output_dir.is_relative_to(source) or source.is_relative_to(self.output_dir):
                raise ValueError("Audit output must not contain or modify source directories")
        if self.config_path.is_relative_to(self.output_dir):
            raise ValueError("Audit output must not contain its configuration")


    def _source(self) -> DevelopmentSensorSource:
        """Construct a validated development-only source without opening observations."""
        return DevelopmentSensorSource(self.prepared_dir, self.raw_data_dir, self.split_path)


    def plan(self) -> dict[str, Any]:
        """Report the label-free schedule and allowed reads from metadata only."""
        source = self._source()
        return {"development_participants": sorted(source.partitions),
                "reserved_test_participants": source.reserved,
                "session_start_seconds": self.clock.start_seconds, "epoch_seconds": 30,
                "end_policy": "earlier_sensor_final_timestamp",
                "context_epochs": 30, "absolute_tolerance": 1e-8,
                "model_loaded": False, "training_performed": False,
                "test_observations_read": False, "raw_labels_read": False,
                "output_dir": str(self.output_dir)}


    @staticmethod
    def _history(rows: pd.DataFrame) -> pd.Series:
        """Measure bounded consecutive context from the supplied eligibility and segments.

        Args:
            rows (pd.DataFrame): Chronological epochs with eligible and segment_id columns.
        """
        counts = rows.groupby("segment_id", sort=False).cumcount().add(1).clip(upper=30)
        return counts.where(rows.eligible, 0).astype(int)


    def _prepare(self, source: DevelopmentSensorSource, person: str) -> pd.DataFrame:
        """Build all seven raw channels without opening a target-bearing reference table.

        Args:
            source (DevelopmentSensorSource): Checked development-only observation source.
            person (str): Allowed development participant identifier.
        """
        heart = source.load_raw(person, "heart_rate")
        motion = source.load_raw(person, "motion")
        epochs = self.clock.completed_epochs(heart, motion)
        rows = EpochFeatureBuilder(source.rule).build_sensor_epochs(heart, motion, epochs)
        grid = FixedLagMotionFeatures().motion_summaries(motion, epochs)
        rows["motion_grid_change_mean"] = grid.motion_grid_change_mean
        rows["history_epochs"] = self._history(rows)
        rows["participant_id"] = person
        return rows


    def _compare(
        self, candidate: pd.DataFrame, reference: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Separate numerical differences from label filtering and session-boundary differences.

        Args:
            candidate (pd.DataFrame): Completed sensor-only rows prepared before reference access.
            reference (pd.DataFrame): Saved label-dependent research preparation.
        """
        reference = reference.sort_values("time", kind="stable").reset_index(drop=True)
        reference["history_epochs"] = self._history(reference)
        merged = candidate.merge(reference, on="time", how="outer", suffixes=("", "_reference"),
                                 validate="one_to_one", indicator=True, sort=True)
        common = merged._merge.eq("both")
        if not common.any():
            raise ValueError("Clock and reference require overlapping epoch starts")
        eligible = merged.eligible.eq(True)
        old_eligible = merged.eligible_reference.eq(True)
        scored = merged.target.ge(0)
        merged["epoch_scope"] = merged._merge.astype(str)
        merged["label_only_exclusion"] = common & eligible & ~scored
        merged["context_changed"] = (common & eligible & old_eligible
                                     & merged.history_epochs.ne(merged.history_epochs_reference))
        mismatches, differences = self._feature_parity(merged.loc[common])
        record = {"participant_id": str(candidate.participant_id.iloc[0]),
                  "clock_epochs": len(candidate), "reference_epochs": len(reference),
                  "common_epochs": int(common.sum()),
                  "sensor_eligible_epochs": int(candidate.eligible.sum()),
                  "clock_only_epochs": int(merged._merge.eq("left_only").sum()),
                  "reference_only_epochs": int(merged._merge.eq("right_only").sum()),
                  "reference_only_eligible_epochs": int(
                      (merged._merge.eq("right_only") & old_eligible).sum()),
                  "clock_only_eligible_epochs": int(
                      (merged._merge.eq("left_only") & eligible).sum()),
                  "label_only_exclusions": int(merged.label_only_exclusion.sum()),
                  "context_changed_epochs": int(merged.context_changed.sum()),
                  "eligibility_rule_mismatches": int(
                      (common & old_eligible.ne(eligible & scored)).sum()),
                  "feature_mismatches": mismatches, "maximum_feature_differences": differences}
        merged["participant_id"] = record["participant_id"]
        return merged.drop(columns="_merge"), record


    @staticmethod
    def _feature_parity(shared: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
        """Count numerical and missing-value differences on shared epoch starts.

        Args:
            shared (pd.DataFrame): Aligned sensor and reference columns for common epochs.
        """
        mismatches, differences = {}, {}
        for column in SUMMARY_COLUMNS:
            left = shared[column].to_numpy(dtype=float)
            right = shared[f"{column}_reference"].to_numpy(dtype=float)
            mismatches[column] = int((~np.isclose(left, right, rtol=0, atol=1e-8,
                                                equal_nan=True)).sum())
            finite = np.isfinite(left) & np.isfinite(right)
            differences[column] = float(np.max(np.abs(left[finite] - right[finite]), initial=0))
        return mismatches, differences


    @staticmethod
    def context_causes(rows: pd.DataFrame) -> pd.DataFrame:
        """Explain changed contexts from cached schedule and unscored-epoch diagnostics.

        Args:
            rows (pd.DataFrame): Published epoch-level audit rows with bounded history lengths.
        """
        records = []
        for person, group in rows.groupby("participant_id", sort=True):
            changed = group.loc[group.context_changed]
            record = {"participant_id": str(person), "changed_epochs": len(changed),
                      "unscored_history_epochs": 0, "unscheduled_history_epochs": 0,
                      "boundary_history_epochs": 0}
            for row in changed.itertuples():
                start = row.time - (row.history_epochs - 1) * 30
                prior = group.time.ge(start) & group.time.lt(row.time)
                unscored = bool((prior & group.label_only_exclusion).any())
                unscheduled = bool((prior & group.epoch_scope.eq("left_only")
                                    & group.eligible.eq(True)).any())
                record["unscored_history_epochs"] += int(unscored)
                record["unscheduled_history_epochs"] += int(unscheduled)
                record["boundary_history_epochs"] += int(not unscored and not unscheduled)
            records.append(record)
        return pd.DataFrame(records)


    def _motion_reference(self, source: DevelopmentSensorSource) -> pd.DataFrame:
        """Verify the historical motion snapshot before projecting its sensor-only channel.

        Args:
            source (DevelopmentSensorSource): Frozen development membership and digest recorder.
        """
        root = self.motion_snapshot_dir
        manifest_path = source._inside(root / "manifest.json", root)
        summary_path = source._inside(root / "summary.json", root)
        rows_path = source._inside(root / "rows.csv", root)
        manifest = json.loads(manifest_path.read_text())
        for path in (summary_path, rows_path):
            if sha256(path) != manifest["outputs"].get(path.name):
                raise ValueError("Motion snapshot checksum differs")
            source.input_hashes[str(path)] = manifest["outputs"][path.name]
        source.input_hashes[str(manifest_path)] = sha256(manifest_path)
        summary = json.loads(summary_path.read_text())
        grid = summary.get("motion_grid", {})
        expected = {"step_seconds": 1.0, "max_sample_age_seconds": 0.2,
                    "points_per_epoch": 29, "pairs_per_epoch": 28}
        if any(grid.get(key) != value for key, value in expected.items()):
            raise ValueError("Motion snapshot must use the fixed one-second grid definition")
        if (set(summary["development_participants"]) != set(source.partitions)
                or set(summary["reserved_test_participants"]) != set(source.reserved)):
            raise ValueError("Motion snapshot membership differs from the frozen split")
        rows = pd.read_csv(rows_path, usecols=["participant_id", "time", "motion_grid_change_mean"],
                           dtype={"participant_id": str})
        if (set(rows.participant_id) != set(source.partitions)
                or rows.duplicated(["participant_id", "time"]).any()):
            raise ValueError("Motion snapshot rows must have unique development identities")
        return rows


    @staticmethod
    def _motion_comparison(rows: pd.DataFrame, saved: pd.DataFrame) -> dict[str, Any]:
        """Check fixed-lag motion against saved values on overlapping recorded identities.

        Args:
            rows (pd.DataFrame): Newly prepared sensor-only epochs.
            saved (pd.DataFrame): Verified historical motion channel.
        """
        joined = rows.merge(saved, on=["participant_id", "time"], suffixes=("", "_saved"),
                            how="inner", validate="one_to_one")
        if joined.empty:
            raise ValueError("Motion comparison requires overlapping epoch identities")
        if set(joined.participant_id) != set(rows.participant_id):
            raise ValueError("Motion comparison requires clock overlap for every participant")
        left = joined.motion_grid_change_mean.to_numpy(float)
        right = joined.motion_grid_change_mean_saved.to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        return {"compared_epochs": len(joined),
                "saved_epochs_outside_clock": len(saved)-len(joined),
                "mismatches": int((~np.isclose(left, right, rtol=0, atol=1e-8,
                                              equal_nan=True)).sum()),
                "maximum_difference": float(np.max(np.abs(left[finite]-right[finite]), initial=0))}


    def run(self) -> Path:
        """Publish an immutable development audit without model loading or optimization."""
        if self.output_dir.exists():
            raise FileExistsError("Audit exists; load it or choose a fresh output directory")
        if sha256(self.config_path) != self.config_sha256:
            raise ValueError("Audit configuration changed after initialization")
        source = self._source()
        plan = self.plan()
        records, frames, candidates = [], [], []
        for index, person in enumerate(sorted(source.partitions), start=1):
            candidate = self._prepare(source, person)
            compared, record = self._compare(candidate, source.load_reference(person))
            records.append(record)
            frames.append(compared)
            candidates.append(candidate)
            LOGGER.info("Audited participant %d/%d id=%s clock_epochs=%d context_changed=%d",
                        index, len(source.partitions), person, len(candidate),
                        record["context_changed_epochs"])
        motion_reference = self._motion_reference(source)
        compared_rows = pd.concat(frames, ignore_index=True)
        expected = compared_rows.loc[compared_rows.eligible_reference.eq(True),
                                     ["participant_id", "time"]]
        covered = expected.merge(motion_reference, on=["participant_id", "time"],
                                 how="left", indicator=True, validate="one_to_one")
        if not covered._merge.eq("both").all():
            raise ValueError("Motion snapshot must cover every prepared eligible epoch")
        motion = self._motion_comparison(pd.concat(candidates), motion_reference)
        failures = motion["mismatches"] + sum(
            sum(row["feature_mismatches"].values()) + row["eligibility_rule_mismatches"]
            for row in records)
        summary = {"status": "passed" if failures == 0 else "differences_detected",
                   "plan": plan, "participants": records, "motion_parity": motion,
                   "input_hashes": source.input_hashes,
                   "shared_eligible_context_parity": all(
                       row["context_changed_epochs"] == 0 for row in records),
                   "status_scope": "numerical_features_and_sensor_coverage",
                   "interpretation": "Context equality is separate from numerical feature parity"}
        for path in (self.prepared_dir / "provenance.json", self.split_path):
            if sha256(path) != source.input_hashes[str(path)]:
                raise ValueError("Source metadata changed during the audit")
        self._publish(compared_rows, summary)
        return self.output_dir


    def _publish(self, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
        """Stage and exclusively publish outputs with hashes and configuration identity.

        Args:
            rows (pd.DataFrame): Epoch-level eligibility and context diagnostics.
            summary (dict[str, Any]): Measured differences and development-only provenance.
        """
        if sha256(self.config_path) != self.config_sha256:
            raise ValueError("Audit configuration changed during the run")
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".sensor-audit-", dir=self.output_dir.parent) as temporary:
            staging = Path(temporary) / "snapshot"
            staging.mkdir()
            rows.to_csv(staging / "rows.csv", index=False)
            (staging / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False)+"\n",
            )
            manifest = {"schema_version": 1, "config_sha256": self.config_sha256,
                        "outputs": {name: sha256(staging / name)
                                    for name in ("rows.csv", "summary.json")}}
            module_dir = Path(__file__).resolve().parent
            manifest["implementation_hashes"] = {
                name: sha256(module_dir / name) for name in (
                    "sensor_preparation_audit.py", "sensor_audit_source.py", "sensor_epochs.py",
                    "features.py", "coverage.py", "motion_grid.py",
                )
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            publish_directory(staging, self.output_dir)


    def load_report(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Verify cached outputs without rereading sensors or restoring weights."""
        root = self.output_dir
        manifest_path = DevelopmentSensorSource._inside(root / "manifest.json", root)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported sensor audit snapshot version")
        if (manifest["config_sha256"] != self.config_sha256
                or sha256(self.config_path) != self.config_sha256):
            raise ValueError("Cached audit configuration differs")
        for name in ("rows.csv", "summary.json"):
            path = DevelopmentSensorSource._inside(root / name, root)
            if sha256(path) != manifest["outputs"][name]:
                raise ValueError("Cached audit output checksum differs")
        return (pd.read_csv(self.output_dir / "rows.csv", dtype={"participant_id": str}),
                json.loads((self.output_dir / "summary.json").read_text()))


def main() -> None:
    """Run the explicitly requested audit or print its metadata-only plan."""
    parser = argparse.ArgumentParser(description="Audit label-free raw-sensor preparation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    audit = LabelFreeSensorAudit(args.config)
    LOGGER.info("Sensor preparation audit: %s", audit.run() if args.run else audit.plan())


if __name__ == "__main__":
    main()
