import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.features import FEATURE_COLUMNS

MOTION_FEATURE = "motion_grid_change_mean"
MOTION_DEFINITION: dict[str, str | float] = {
    "name": "fixed_lag_motion", "step_seconds": 1.0, "max_sample_age_seconds": 0.2,
}


class MotionFeatureInput:
    """Join a verified causal diagnostic without importing audit targets or predictions."""

    def __init__(self, values: dict[str, Any] | None, config_path: Path) -> None:
        """Resolve an optional audit configuration relative to the training recipe.

        Args:
            values (dict[str, Any] | None): Fixed-lag feature source settings.
            config_path (Path): Training configuration containing the source reference.
        """
        self.validate_settings(values)
        self.enabled = values is not None
        self.audit_config = None
        if values is not None:
            self.audit_config = (config_path.resolve().parent / values["audit_config"]).resolve()


    @staticmethod
    def validate_settings(values: dict[str, Any] | None) -> None:
        """Reject incomplete source settings without reading an audit or raw observations.

        Args:
            values (dict[str, Any] | None): Optional training feature mapping.
        """
        if values is None:
            return
        if (not isinstance(values, dict) or set(values) != {"name", "audit_config"}
                or values["name"] != "fixed_lag_motion"
                or not isinstance(values["audit_config"], str)
                or not values["audit_config"].strip()):
            raise ValueError("motion_features requires fixed_lag_motion and an audit_config path")


    @staticmethod
    def feature_columns(values: dict[str, Any] | None) -> tuple[str, ...]:
        """Return ordered raw model columns without changing the prepared six-feature schema.

        Args:
            values (dict[str, Any] | None): Validated training feature settings.
        """
        MotionFeatureInput.validate_settings(values)
        return (*FEATURE_COLUMNS, MOTION_FEATURE) if values is not None else FEATURE_COLUMNS


    def prepare(
        self, train: pd.DataFrame, validation: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
        """Join sensor values and impute unavailable values from training targets only.

        Args:
            train (pd.DataFrame): Eligible training targets with participant and time identities.
            validation (pd.DataFrame): Eligible held-out targets with matching identity fields.
        """
        if not self.enabled:
            return train.copy(), validation.copy(), None
        sensor, summary, manifest = self._load_sensor()
        allowed = set(summary["development_participants"])
        reserved = set(summary["reserved_test_participants"])
        joined_train = self._join(train, sensor, allowed, reserved)
        joined_validation = self._join(validation, sensor, allowed, reserved)
        observed = joined_train[MOTION_FEATURE].dropna()
        if observed.empty:
            raise ValueError("Motion imputation requires an observed training target")
        mean = float(observed.mean())
        metadata = {
            "name": "fixed_lag_motion", "definition": dict(MOTION_DEFINITION),
            "audit_config_sha256": self._digest(self.audit_config),
            "snapshot_manifest_sha256": self._digest(manifest),
            "scaling_fit": "training_targets_only", "missing_policy": "training_mean",
            "training_observed_targets": len(observed),
            "training_missing_targets": int(joined_train[MOTION_FEATURE].isna().sum()),
            "validation_observed_targets": int(joined_validation[MOTION_FEATURE].notna().sum()),
            "validation_missing_targets": int(joined_validation[MOTION_FEATURE].isna().sum()),
            "imputation_mean": mean,
        }
        for frame in (joined_train, joined_validation):
            frame[MOTION_FEATURE] = frame[MOTION_FEATURE].fillna(mean)
        return joined_train, joined_validation, metadata


    def transform(self, frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
        """Join replay values and apply saved training imputation without fitting anything.

        Args:
            frame (pd.DataFrame): Eligible targets requiring the supplemental channel.
            state (dict[str, Any]): Training-fitted source and imputation metadata.
        """
        sensor, summary, manifest = self._load_sensor()
        if (state.get("definition") != MOTION_DEFINITION
                or state.get("audit_config_sha256") != self._digest(self.audit_config)
                or state.get("snapshot_manifest_sha256") != self._digest(manifest)
                or state.get("missing_policy") != "training_mean"
                or state.get("scaling_fit") != "training_targets_only"):
            raise ValueError("Motion replay source differs from its saved training metadata")
        mean = state.get("imputation_mean")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not np.isfinite(mean):
            raise ValueError("Motion replay requires a finite saved training imputation mean")
        result = self._join(frame, sensor, set(summary["development_participants"]),
                            set(summary["reserved_test_participants"]))
        result[MOTION_FEATURE] = result[MOTION_FEATURE].fillna(mean)
        return result


    def _load_sensor(self) -> tuple[pd.DataFrame, dict[str, Any], Path]:
        """Verify one historical snapshot and project only its causal sensor channel."""
        if not self.enabled:
            raise ValueError("Motion sensor loading requires an enabled source")
        from sleep_classifier.causal_feature_audit import CausalFeatureAudit

        audit = CausalFeatureAudit(self.audit_config)
        rows, summary = audit.load_report()
        grid = summary.get("motion_grid", {})
        if (grid.get("step_seconds") != 1.0 or grid.get("max_sample_age_seconds") != 0.2
                or grid.get("points_per_epoch") != 29 or grid.get("pairs_per_epoch") != 28):
            raise ValueError("Motion source must use the declared one-second fixed-lag definition")
        # Select sensor columns explicitly before joining any source data
        sensor = rows.loc[:, ["participant_id", "time", MOTION_FEATURE]].copy()
        if sensor.duplicated(["participant_id", "time"]).any():
            raise ValueError("Motion source identities must be unique")
        return sensor, summary, audit.output_dir / "manifest.json"


    @staticmethod
    def _join(
        frame: pd.DataFrame, sensor: pd.DataFrame, allowed: set[str], reserved: set[str],
    ) -> pd.DataFrame:
        """Retain every supplied target and require an existing development sensor identity.

        Args:
            frame (pd.DataFrame): One eligible participant partition.
            sensor (pd.DataFrame): Projected sensor-only audit values.
            allowed (set[str]): Declared development participants.
            reserved (set[str]): Reserved final-test participants.
        """
        if MOTION_FEATURE in frame:
            raise ValueError("Motion feature must not already exist in prepared rows")
        people = set(frame.participant_id.astype(str))
        if not people <= allowed or people & reserved:
            raise ValueError("Motion inputs must contain development participants only")
        if frame[["participant_id", "time"]].isna().any().any():
            raise ValueError("Motion target identities must be complete")
        result = frame.merge(sensor, on=["participant_id", "time"], how="left",
                             validate="one_to_one", indicator=True, sort=False)
        if not result["_merge"].eq("both").all():
            raise ValueError("Every model target must have a motion audit identity")
        values = result[MOTION_FEATURE].to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError("Motion values must be nonnegative or unavailable")
        return result.drop(columns="_merge")


    @staticmethod
    def _digest(path: Path) -> str:
        """Hash one immutable source artifact without exposing its contents.

        Args:
            path (Path): Verified configuration or snapshot manifest.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()
