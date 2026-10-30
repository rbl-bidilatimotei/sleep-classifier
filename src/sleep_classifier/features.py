"""Build causal sensor summaries for labeled sleep epochs."""

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


def build_epoch_features(
    heart: pd.DataFrame,
    motion: pd.DataFrame,
    labels: pd.DataFrame,
    rule: dict,
) -> pd.DataFrame:
    """Summarize sensor values within each labeled half-open 30-second epoch.

    Args:
        heart (pd.DataFrame): Heart observations with ``time`` and ``heart_rate``.
        motion (pd.DataFrame): Acceleration observations with ``time``, ``x``, ``y``, and ``z``.
        labels (pd.DataFrame): Sorted labels with ``time`` and ``stage``.
        rule (dict): Coverage thresholds from the selected coverage rule.

    Returns:
        pd.DataFrame: Epoch features, coverage, target, eligibility, and segment ID.

    Raises:
        ValueError: If required columns, timestamps, stages, or rule thresholds are invalid.
    """
    required = {
        "heart": (heart, {"time", "heart_rate"}),
        "motion": (motion, {"time", "x", "y", "z"}),
        "labels": (labels, {"time", "stage"}),
    }
    for name, (frame, columns) in required.items():
        if not columns.issubset(frame.columns):
            missing = sorted(columns - set(frame.columns))
            raise ValueError(f"{name} frame is missing columns: {missing}")

    stage = labels["stage"].to_numpy(dtype=float)
    if not np.isfinite(stage).all() or not np.isin(stage, [-1, 0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Labels must use source stages -1 through 5")
    target_map = {-1: -1, 0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
    targets = np.array([target_map[int(value)] for value in stage], dtype=int)

    heart_clean = deduplicate_sensor(heart)
    motion_clean = deduplicate_sensor(motion)
    coverage_hr = epoch_coverage(heart_clean, labels)
    coverage_motion = epoch_coverage(motion_clean, labels)
    label_times = labels["time"].to_numpy(dtype=float)
    heart_times = heart_clean["time"].to_numpy(dtype=float)
    motion_times = motion_clean["time"].to_numpy(dtype=float)
    if not np.isfinite(heart_times).all() or not np.isfinite(motion_times).all():
        raise ValueError("Sensor times must be finite")

    hr_values = heart_clean["heart_rate"].to_numpy(dtype=float)
    axes = motion_clean[["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(hr_values).all() or not np.isfinite(axes).all():
        raise ValueError("Sensor values must be finite")
    magnitude = np.sqrt(np.square(axes).sum(axis=1))
    feature_rows = []
    for start in label_times:
        hr_left, hr_right = np.searchsorted(heart_times, [start, start + 30])
        motion_left, motion_right = np.searchsorted(motion_times, [start, start + 30])
        hr = hr_values[hr_left:hr_right]
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

    result = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    result.insert(0, "time", label_times)
    result["target"] = targets
    for prefix, coverage in (("heart", coverage_hr), ("motion", coverage_motion)):
        result[f"{prefix}_sample_count"] = coverage["sample_count"].to_numpy()
        result[f"{prefix}_occupied_bin_fraction"] = coverage["occupied_bin_fraction"].to_numpy()
        result[f"{prefix}_largest_observation_gap"] = coverage["largest_observation_gap"].to_numpy()

    threshold_keys = {
        "max_heart_gap_seconds",
        "min_heart_readings",
        "min_motion_bin_fraction",
        "max_motion_gap_seconds",
    }
    if not threshold_keys.issubset(rule):
        raise ValueError(f"Rule is missing thresholds: {sorted(threshold_keys - set(rule))}")
    thresholds = {key: float(rule[key]) for key in threshold_keys}
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
    eligible = (
        (result.heart_sample_count >= rule["min_heart_readings"])
        & (result.heart_largest_observation_gap <= rule["max_heart_gap_seconds"])
        & (result.motion_sample_count > 0)
        & (result.motion_occupied_bin_fraction >= rule["min_motion_bin_fraction"])
        & (result.motion_largest_observation_gap <= rule["max_motion_gap_seconds"])
        & (result.target >= 0)
    )
    result["eligible"] = eligible

    segment_ids = np.full(len(result), -1, dtype=int)
    current_segment = -1
    previous_valid = False
    for index, (time, valid) in enumerate(zip(label_times, eligible.to_numpy(), strict=True)):
        contiguous = index > 0 and np.isclose(time - label_times[index - 1], 30, rtol=0, atol=1e-8)
        if valid and (not previous_valid or not contiguous):
            current_segment += 1
        if valid:
            segment_ids[index] = current_segment
        previous_valid = bool(valid)
    result["segment_id"] = segment_ids
    return result
