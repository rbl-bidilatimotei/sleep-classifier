from numbers import Real

import numpy as np
import pandas as pd

from sleep_classifier.coverage import deduplicate_sensor

MOTION_GRID_COLUMNS: tuple[str, ...] = (
    "time",
    "motion_grid_change_mean",
    "motion_grid_point_count",
    "motion_grid_point_fraction",
    "motion_grid_pair_count",
    "motion_grid_pair_fraction",
    "motion_grid_age_max_seconds",
    "motion_grid_actual_lag_median_seconds",
    "motion_grid_actual_lag_min_seconds",
    "motion_grid_actual_lag_max_seconds",
)


class FixedLagMotionFeatures:
    """Summarize axis-vector changes on causal ticks within completed epochs."""

    def __init__(
        self, step_seconds: float = 1.0, max_sample_age_seconds: float = 0.2
    ) -> None:
        """Validate the tick spacing and permitted observation age.

        Args:
            step_seconds (float): Tick spacing within each 30-second epoch.
            max_sample_age_seconds (float): Maximum age of a selected raw observation.

        Raises:
            ValueError: If settings are nonfinite, boolean, or outside their valid ranges.
        """
        self._step_seconds = self._validate_number(step_seconds, "step_seconds")
        self._max_sample_age_seconds = self._validate_number(
            max_sample_age_seconds, "max_sample_age_seconds"
        )
        if not 0 < self.step_seconds < 15:
            raise ValueError("step_seconds must be greater than zero and less than 15")
        if not 0 <= self.max_sample_age_seconds < self.step_seconds:
            raise ValueError("max_sample_age_seconds must be in [0, step_seconds)")
        self._offsets = np.arange(1, int(np.ceil(30 / self.step_seconds))) * self.step_seconds
        self._offsets = self._offsets[self._offsets < 30]
        if len(self._offsets) < 2:
            raise ValueError("step_seconds must provide at least two interior ticks")


    @property
    def step_seconds(self) -> float:
        """Return the spacing between scheduled grid ticks."""
        return self._step_seconds


    @property
    def max_sample_age_seconds(self) -> float:
        """Return the maximum permitted age at a grid tick."""
        return self._max_sample_age_seconds


    @property
    def points_per_epoch(self) -> int:
        """Return the number of interior ticks scheduled per epoch."""
        return len(self._offsets)


    @property
    def pairs_per_epoch(self) -> int:
        """Return the number of adjacent tick pairs scheduled per epoch."""
        return self.points_per_epoch - 1


    def motion_summaries(self, motion: pd.DataFrame, epochs: pd.DataFrame) -> pd.DataFrame:
        """Return observed vector changes and grid support for every target epoch.

        Args:
            motion (pd.DataFrame): One participant's raw time, x, y, and z observations.
            epochs (pd.DataFrame): Chronological target starts on a 30-second grid.

        Returns:
            pd.DataFrame: All target times and their feature and support summaries.

        Raises:
            ValueError: If observations, timestamps, or the epoch schedule are invalid.
        """
        starts = self._epoch_starts(epochs)
        times, axes = self._motion_arrays(motion)
        ticks = starts[:, None] + self._offsets
        indices = np.searchsorted(times, ticks, side="right") - 1
        rows = [
            self._epoch_summary(start, epoch_ticks, epoch_indices, times, axes)
            for start, epoch_ticks, epoch_indices in zip(starts, ticks, indices, strict=True)
        ]
        return pd.DataFrame(rows, columns=MOTION_GRID_COLUMNS)


    @staticmethod
    def _validate_number(value: float, name: str) -> float:
        """Require a finite real setting without accepting boolean values.

        Args:
            value (float): Setting value to validate.
            name (str): Setting name for the error message.

        Returns:
            float: The finite setting value.
        """
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite real number")
        return result


    @staticmethod
    def _epoch_starts(epochs: pd.DataFrame) -> np.ndarray:
        """Validate unique chronological starts on a shared 30-second grid.

        Args:
            epochs (pd.DataFrame): Target intervals with a time column.

        Returns:
            np.ndarray: Finite epoch starts in their original order.
        """
        if "time" not in epochs.columns:
            raise ValueError("Epochs must contain a time column")
        starts = epochs["time"].to_numpy(dtype=float)
        if not np.isfinite(starts).all():
            raise ValueError("Epoch times must be finite")
        if len(starts):
            offsets = starts - starts[0]
            on_grid = np.isclose(offsets, np.rint(offsets / 30) * 30, rtol=0, atol=1e-8)
            if np.any(np.diff(starts) < 30 - 1e-8) or not on_grid.all():
                raise ValueError("Epoch times must be unique, sorted, and on a 30-second grid")
        return starts


    @staticmethod
    def _motion_arrays(motion: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Validate raw values and apply the shared sensor duplicate policy.

        Args:
            motion (pd.DataFrame): Raw time and axis observations.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sorted unique timestamps and corresponding axes.
        """
        required = {"time", "x", "y", "z"}
        if not required.issubset(motion.columns):
            raise ValueError(f"Motion frame is missing columns: {sorted(required - set(motion))}")
        clean = motion.loc[:, ["time", "x", "y", "z"]].astype(float)
        if not np.isfinite(clean.to_numpy()).all():
            raise ValueError("Motion times and axes must be finite")
        clean = deduplicate_sensor(clean)
        return clean["time"].to_numpy(dtype=float), clean[["x", "y", "z"]].to_numpy(dtype=float)


    def _epoch_summary(
        self,
        start: float,
        ticks: np.ndarray,
        indices: np.ndarray,
        times: np.ndarray,
        axes: np.ndarray,
    ) -> dict[str, float | int]:
        """Summarize supported adjacent ticks without joining across missing points.

        Args:
            start (float): Target interval start.
            ticks (np.ndarray): Interior grid timestamps for this interval.
            indices (np.ndarray): Latest raw indices at or before each tick.
            times (np.ndarray): Sorted raw timestamps for this participant.
            axes (np.ndarray): Raw axis vectors for this participant.

        Returns:
            dict[str, float | int]: One target time with feature and support measurements.
        """
        row: dict[str, float | int] = dict.fromkeys(MOTION_GRID_COLUMNS, np.nan)
        row.update(time=float(start), motion_grid_point_count=0, motion_grid_point_fraction=0.0,
                   motion_grid_pair_count=0, motion_grid_pair_fraction=0.0)
        if not len(times):
            return row

        safe_indices = np.maximum(indices, 0)
        selected_times = times[safe_indices]
        ages = ticks - selected_times
        valid = ((indices >= 0) & (selected_times >= start) & (selected_times < start + 30)
                 & (ages >= 0) & (ages <= self.max_sample_age_seconds + 1e-8))
        point_count = int(valid.sum())
        row["motion_grid_point_count"] = point_count
        row["motion_grid_point_fraction"] = point_count / self.points_per_epoch
        if point_count:
            row["motion_grid_age_max_seconds"] = float(ages[valid].max())

        actual_lags = np.diff(selected_times)
        valid_pairs = valid[:-1] & valid[1:] & (actual_lags > 0)
        pair_count = int(valid_pairs.sum())
        row["motion_grid_pair_count"] = pair_count
        row["motion_grid_pair_fraction"] = pair_count / self.pairs_per_epoch
        if pair_count:
            differences = np.diff(axes[safe_indices], axis=0)[valid_pairs]
            lags = actual_lags[valid_pairs]
            changes = np.linalg.norm(differences, axis=1)
            if not np.isfinite(changes).all():
                raise ValueError("Motion grid vector changes must be finite")
            row["motion_grid_change_mean"] = float(changes.mean())
            row["motion_grid_actual_lag_median_seconds"] = float(np.median(lags))
            row["motion_grid_actual_lag_min_seconds"] = float(lags.min())
            row["motion_grid_actual_lag_max_seconds"] = float(lags.max())
        return row
