import numpy as np
import pandas as pd

from sleep_classifier.coverage import deduplicate_sensor, epoch_coverage

FEATURE_COLUMNS = (
    "heart_rate_mean",
    "heart_rate_std",
    "heart_rate_min",
    "heart_rate_max",
    "motion_magnitude_mean",
    "motion_magnitude_std",
)

TARGET_MAP = {-1: -1, 0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
RULE_THRESHOLDS = {
    "max_heart_gap_seconds",
    "min_heart_readings",
    "min_motion_bin_fraction",
    "max_motion_gap_seconds",
}


class EpochFeatureBuilder:
    """Build sensor summaries and eligibility for labeled 30-second epochs."""

    def __init__(self, rule: dict) -> None:
        """Store and validate the coverage rule used for eligibility."""
        self.rule = rule
        self.thresholds = self._validate_rule(rule)

    def build(
        self, heart: pd.DataFrame, motion: pd.DataFrame, labels: pd.DataFrame
    ) -> pd.DataFrame:
        """Create feature, coverage, target, and segment columns."""
        self._validate_frames(heart, motion, labels)
        targets = self._targets(labels)
        heart_clean = deduplicate_sensor(heart)
        motion_clean = deduplicate_sensor(motion)
        coverage = self._coverage(heart_clean, motion_clean, labels)
        heart_times, motion_times, hr_values, magnitude = self._sensor_arrays(
            heart_clean, motion_clean
        )
        label_times = labels["time"].to_numpy(dtype=float)
        result = self._summarize_epochs(
            label_times, heart_times, motion_times, hr_values, magnitude
        )
        result.insert(0, "time", label_times)
        result["target"] = targets
        self._append_coverage(result, coverage)
        result["eligible"] = self._eligible_epochs(result)
        result["segment_id"] = self._segment_ids(label_times, result["eligible"].to_numpy())
        return result

    def _validate_frames(
        self, heart: pd.DataFrame, motion: pd.DataFrame, labels: pd.DataFrame
    ) -> None:
        """Check required columns before processing sensor frames."""
        required = {
            "heart": (heart, {"time", "heart_rate"}),
            "motion": (motion, {"time", "x", "y", "z"}),
            "labels": (labels, {"time", "stage"}),
        }
        for name, (frame, columns) in required.items():
            if not columns.issubset(frame.columns):
                missing = sorted(columns - set(frame.columns))
                raise ValueError(f"{name} frame is missing columns: {missing}")

    def _targets(self, labels: pd.DataFrame) -> np.ndarray:
        """Map source sleep stages to the four-class target scheme."""
        stage = labels["stage"].to_numpy(dtype=float)
        if not np.isfinite(stage).all() or not np.isin(stage, list(TARGET_MAP)).all():
            raise ValueError("Labels must use source stages -1 through 5")
        return np.array([TARGET_MAP[int(value)] for value in stage], dtype=int)

    def _coverage(
        self, heart: pd.DataFrame, motion: pd.DataFrame, labels: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Calculate heart-rate and motion coverage for each epoch."""
        return {
            "heart": epoch_coverage(heart, labels),
            "motion": epoch_coverage(motion, labels),
        }

    def _sensor_arrays(
        self, heart: pd.DataFrame, motion: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate sensor values and return arrays for epoch summaries."""
        heart_times = heart["time"].to_numpy(dtype=float)
        motion_times = motion["time"].to_numpy(dtype=float)
        if not np.isfinite(heart_times).all() or not np.isfinite(motion_times).all():
            raise ValueError("Sensor times must be finite")
        hr_values = heart["heart_rate"].to_numpy(dtype=float)
        axes = motion[["x", "y", "z"]].to_numpy(dtype=float)
        if not np.isfinite(hr_values).all() or not np.isfinite(axes).all():
            raise ValueError("Sensor values must be finite")
        return heart_times, motion_times, hr_values, np.sqrt(np.square(axes).sum(axis=1))

    def _summarize_epochs(
        self,
        label_times: np.ndarray,
        heart_times: np.ndarray,
        motion_times: np.ndarray,
        heart_values: np.ndarray,
        magnitude: np.ndarray,
    ) -> pd.DataFrame:
        """Summarize heart rate and motion magnitude in half-open epochs."""
        feature_rows = []
        for start in label_times:
            hr_left, hr_right = np.searchsorted(heart_times, [start, start + 30])
            motion_left, motion_right = np.searchsorted(motion_times, [start, start + 30])
            hr = heart_values[hr_left:hr_right]
            mag = magnitude[motion_left:motion_right]
            feature_rows.append(
                {
                    "heart_rate_mean": float(np.mean(hr)) if len(hr) else np.nan,
                    "heart_rate_std": float(np.std(hr, ddof=0)) if len(hr) else np.nan,
                    "heart_rate_min": float(np.min(hr)) if len(hr) else np.nan,
                    "heart_rate_max": float(np.max(hr)) if len(hr) else np.nan,
                    "motion_magnitude_mean": float(np.mean(mag)) if len(mag) else np.nan,
                    "motion_magnitude_std": float(np.std(mag, ddof=0)) if len(mag) else np.nan,
                }
            )
        return pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    def _append_coverage(self, result: pd.DataFrame, coverage: dict[str, pd.DataFrame]) -> None:
        """Add coverage measurements to the feature table."""
        for prefix, summary in coverage.items():
            for column in (
                "sample_count",
                "occupied_bin_fraction",
                "largest_observation_gap",
            ):
                result[f"{prefix}_{column}"] = summary[column].to_numpy()

    def _validate_rule(self, rule: dict) -> dict[str, float]:
        """Validate and normalize coverage thresholds."""
        if not RULE_THRESHOLDS.issubset(rule):
            raise ValueError(f"Rule is missing thresholds: {sorted(RULE_THRESHOLDS - set(rule))}")
        thresholds = {key: float(rule[key]) for key in RULE_THRESHOLDS}
        if not all(np.isfinite(value) for value in thresholds.values()):
            raise ValueError("Rule thresholds must be finite")
        if not (
            thresholds["min_heart_readings"] >= 1
            and thresholds["min_heart_readings"].is_integer()
            and 0 <= thresholds["min_motion_bin_fraction"] <= 1
            and 0 <= thresholds["max_heart_gap_seconds"] <= 30
            and 0 <= thresholds["max_motion_gap_seconds"] <= 30
        ):
            raise ValueError("Rule thresholds are outside their valid ranges")
        return thresholds

    def _eligible_epochs(self, result: pd.DataFrame) -> pd.Series:
        """Mark epochs that meet coverage thresholds and have scored targets."""
        return (
            (result.heart_sample_count >= self.thresholds["min_heart_readings"])
            & (result.heart_largest_observation_gap <= self.thresholds["max_heart_gap_seconds"])
            & (result.motion_sample_count > 0)
            & (result.motion_occupied_bin_fraction >= self.thresholds["min_motion_bin_fraction"])
            & (result.motion_largest_observation_gap <= self.thresholds["max_motion_gap_seconds"])
            & (result.target >= 0)
        )

    def _segment_ids(self, times: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        """Assign contiguous IDs to adjacent eligible epochs."""
        segment_ids = np.full(len(times), -1, dtype=int)
        current_segment = -1
        previous_valid = False
        for index, (time, valid) in enumerate(zip(times, eligible, strict=True)):
            contiguous = index > 0 and np.isclose(time - times[index - 1], 30, rtol=0, atol=1e-8)
            if valid and (not previous_valid or not contiguous):
                current_segment += 1
            if valid:
                segment_ids[index] = current_segment
            previous_valid = bool(valid)
        return segment_ids


def build_epoch_features(
    heart: pd.DataFrame,
    motion: pd.DataFrame,
    labels: pd.DataFrame,
    rule: dict,
) -> pd.DataFrame:
    return EpochFeatureBuilder(rule).build(heart, motion, labels)
