import numpy as np
import pandas as pd


def deduplicate_sensor(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort sensor rows and remove exact duplicates by timestamp.

    Args:
        frame (pd.DataFrame): Sensor observations with a ``time`` column.

    Returns:
        pd.DataFrame: Sorted observations with one row per timestamp.

    Raises:
        ValueError: If a timestamp has conflicting sensor values.
    """
    if "time" not in frame.columns:
        raise ValueError("Sensor frame must contain a time column")
    if frame.empty:
        return frame.copy().sort_values("time", kind="stable").reset_index(drop=True)
    conflicts = frame.groupby("time", dropna=False).nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("Conflicting sensor values at duplicate timestamps")
    return (
        frame.drop_duplicates(subset="time", keep="first")
        .sort_values("time", kind="stable")
        .reset_index(drop=True)
    )


def epoch_coverage(frame: pd.DataFrame, labels: pd.DataFrame, bin_seconds: int = 1) -> pd.DataFrame:
    """Summarize unique sensor observations inside assumed label epochs.

    Label timestamps mark starts of half-open 30-second epochs. The occupied-bin
    fraction describes timestamp distribution and does not imply physiology.

    Args:
        frame (pd.DataFrame): Sensor observations with a ``time`` column.
        labels (pd.DataFrame): Labels with unique, sorted ``time`` values.
        bin_seconds (int): Width of fixed bins that divide a 30-second epoch.

    Returns:
        pd.DataFrame: Epoch starts, sample counts, occupied-bin fractions, and gaps.

    Raises:
        ValueError: If labels or bins do not follow the 30-second epoch grid.
    """
    if not isinstance(bin_seconds, (int, np.integer)) or bin_seconds <= 0 or 30 % bin_seconds:
        raise ValueError("bin_seconds must be a positive integer divisor of 30")
    if "time" not in labels.columns:
        raise ValueError("Labels must contain a time column")
    label_times = labels["time"].to_numpy(dtype=float)
    if not np.isfinite(label_times).all():
        raise ValueError("Label times must be finite")
    if len(label_times):
        offsets = label_times - label_times[0]
        on_grid = np.isclose(offsets, np.rint(offsets / 30) * 30, rtol=0, atol=1e-8)
        if np.any(np.diff(label_times) <= 0) or not on_grid.all():
            raise ValueError("Label times must be unique, sorted, and on a 30-second grid")

    sensors = deduplicate_sensor(frame)
    sample_times = sensors["time"].to_numpy(dtype=float)
    if not np.isfinite(sample_times).all():
        raise ValueError("Sensor times must be finite")

    bin_count = 30 // bin_seconds
    rows = []
    for start in label_times:
        end = start + 30
        left, right = np.searchsorted(sample_times, [start, end], side="left")
        times = sample_times[left:right]
        if len(times):
            gaps = np.diff(np.concatenate(([start], times, [end])))
            occupied = np.unique(((times - start) // bin_seconds).astype(int)).size
            largest_gap = float(gaps.max())
        else:
            occupied = 0
            largest_gap = 30.0
        rows.append(
            {
                "time": start,
                "sample_count": len(times),
                "occupied_bin_fraction": occupied / bin_count,
                "largest_observation_gap": largest_gap,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["time", "sample_count", "occupied_bin_fraction", "largest_observation_gap"],
    )


def step_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observed step values with endpoints from the next timestamp.

    Args:
        frame (pd.DataFrame): Step observations with ``time`` and ``steps`` columns.

    Returns:
        pd.DataFrame: Sorted start, observed end, and step count rows.
    """
    if "steps" not in frame.columns:
        raise ValueError("Step frame must contain a steps column")
    rows = deduplicate_sensor(frame)
    result = pd.DataFrame({"start": rows["time"], "steps": rows["steps"]})
    result["end"] = rows["time"].shift(-1)
    return result[["start", "end", "steps"]].reset_index(drop=True)
