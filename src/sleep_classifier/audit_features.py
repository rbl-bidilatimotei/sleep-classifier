import math
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.coverage import deduplicate_sensor

MOTION_AUDIT_COLUMNS: tuple[str, ...] = (
    "time",
    "motion_vector_change_mean",
    "motion_pair_count",
    "motion_valid_pair_fraction",
    "motion_pair_gap_median_seconds",
    "motion_pair_gap_max_seconds",
)


class CausalAuditFeatures:
    """Inspect past-relative heart rate and within-epoch motion changes."""

    def __init__(
        self, history_epochs: int = 10, min_history_epochs: int = 4,
        max_motion_pair_gap_seconds: float = 0.2,
    ) -> None:
        """Set the finite history and admissible adjacent-sample gap.

        Args:
            history_epochs (int): Maximum previous eligible epochs in each heart-rate mean.
            min_history_epochs (int): Previous epochs required before reporting a delta.
            max_motion_pair_gap_seconds (float): Largest adjacent gap used for vector changes.
        """
        if type(history_epochs) is not int or history_epochs <= 0:
            raise ValueError("history_epochs must be a positive integer")
        if (type(min_history_epochs) is not int
                or not 1 <= min_history_epochs <= history_epochs):
            raise ValueError("min_history_epochs must be in [1, history_epochs]")
        gap = max_motion_pair_gap_seconds
        if (isinstance(gap, bool) or not isinstance(gap, (int, float))
                or not math.isfinite(gap) or not 0 < gap <= 30):
            raise ValueError("max_motion_pair_gap_seconds must be finite and in (0, 30]")

        self.history_epochs = history_epochs
        self.min_history_epochs = min_history_epochs
        self.max_motion_pair_gap_seconds = float(gap)


    def add_heart_history(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add strictly previous-history summaries without changing target eligibility.

        Args:
            rows (pd.DataFrame): Eligible targets with participant, segment, time, and heart rate.
        """
        result = self._heart_rows(rows)
        result["heart_rate_past_mean"] = np.nan
        result["heart_rate_past_count"] = 0
        for _, group in result.groupby(["participant_id", "segment_id"], sort=False):
            past = group.heart_rate_mean.shift(1)
            means = past.rolling(self.history_epochs, min_periods=1).mean()
            counts = np.minimum(np.arange(len(group)), self.history_epochs)
            result.loc[group.index, "heart_rate_past_mean"] = means.to_numpy()
            result.loc[group.index, "heart_rate_past_count"] = counts

        result["heart_rate_past_count"] = result.heart_rate_past_count.astype(int)
        result["heart_history_available"] = (
            result.heart_rate_past_count >= self.min_history_epochs
        )
        result["heart_rate_delta_past"] = (
            result.heart_rate_mean - result.heart_rate_past_mean
        ).where(result.heart_history_available)
        return result


    @staticmethod
    def _heart_rows(rows: pd.DataFrame) -> pd.DataFrame:
        """Validate causal segment metadata while preserving every input column.

        Args:
            rows (pd.DataFrame): Eligible target and feature rows to sort and validate.
        """
        required = {"participant_id", "time", "target", "segment_id", "heart_rate_mean"}
        if not required.issubset(rows.columns):
            raise ValueError("Heart-rate rows lack required columns")
        result = rows.copy()
        if result[list(required)].isna().any().any():
            raise ValueError("Heart-rate epoch metadata must be complete")
        numeric = result[["time", "target", "segment_id", "heart_rate_mean"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Heart-rate epoch values must be finite")
        if not result.target.isin(range(4)).all():
            raise ValueError("Targets must use stage codes 0..3")
        segment_ids = numeric[:, 2]
        if (segment_ids < 0).any() or (segment_ids % 1 != 0).any():
            raise ValueError("Segment identifiers must be nonnegative integers")
        result["participant_id"] = result.participant_id.astype(str)
        if result.participant_id.str.len().eq(0).any():
            raise ValueError("Participant identifiers must be nonempty")
        if result.duplicated(["participant_id", "time"]).any():
            raise ValueError("Participant epoch times must be unique")
        result = result.sort_values(["participant_id", "time"], kind="stable")
        result = result.reset_index(drop=True)
        CausalAuditFeatures._validate_segments(result)
        return result


    @staticmethod
    def _validate_segments(rows: pd.DataFrame) -> None:
        """Require consecutive, contiguous thirty-second epochs within each segment.

        Args:
            rows (pd.DataFrame): Validated participant rows sorted by epoch start.
        """
        previous = rows.groupby("participant_id", sort=False).segment_id.shift()
        starts = rows.loc[rows.segment_id.ne(previous)]
        if starts.duplicated(["participant_id", "segment_id"]).any():
            raise ValueError("Segment identifiers must form consecutive blocks")
        groups = rows.groupby(["participant_id", "segment_id"], sort=False)
        differences = groups.time.diff().dropna().to_numpy(dtype=float)
        if not np.isclose(differences, 30.0, rtol=0, atol=1e-8).all():
            raise ValueError("Epochs within a segment must be contiguous at thirty seconds")
        for _, group in rows.groupby("participant_id", sort=False):
            CausalAuditFeatures._epoch_starts(group[["time"]])


    @staticmethod
    def _epoch_starts(epochs: pd.DataFrame) -> np.ndarray:
        """Validate unique chronological starts on a thirty-second epoch grid.

        Args:
            epochs (pd.DataFrame): Epoch rows with a time column.
        """
        if "time" not in epochs:
            raise ValueError("Epochs must contain time")
        starts = epochs.time.to_numpy(dtype=float)
        if not np.isfinite(starts).all():
            raise ValueError("Epoch start times must be finite")
        if not len(starts):
            return starts
        offsets = starts - starts[0]
        if (np.any(np.diff(starts) <= 0)
                or not np.isclose(offsets, np.rint(offsets / 30) * 30, rtol=0, atol=1e-8).all()):
            raise ValueError("Epoch starts must be unique, sorted, and on a thirty-second grid")
        return starts


    @staticmethod
    def _motion_rows(motion: pd.DataFrame) -> pd.DataFrame:
        """Validate raw axes and apply the existing exact-duplicate policy.

        Args:
            motion (pd.DataFrame): Raw time, x, y, and z observations for one participant.
        """
        columns = ["time", "x", "y", "z"]
        if not set(columns).issubset(motion.columns):
            raise ValueError("Motion rows must contain time, x, y, and z")
        result = motion.loc[:, columns].astype(float)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Motion observations must be finite")
        return deduplicate_sensor(result)


    def motion_summaries(self, motion: pd.DataFrame, epochs: pd.DataFrame) -> pd.DataFrame:
        """Measure adjacent vector changes only inside each completed interval.

        Args:
            motion (pd.DataFrame): Raw observations for a single participant.
            epochs (pd.DataFrame): Sorted target epoch starts; each interval is [time, time + 30).
        """
        starts = self._epoch_starts(epochs)
        clean = self._motion_rows(motion)
        times = clean.time.to_numpy(dtype=float)
        vectors = clean[["x", "y", "z"]].to_numpy(dtype=float)
        left = np.searchsorted(times, starts, side="left")
        right = np.searchsorted(times, starts + 30, side="left")
        summaries: list[dict[str, Any]] = []
        for start, lower, upper in zip(starts, left, right, strict=True):
            summaries.append(self._motion_epoch(float(start), times[lower:upper],
                                                vectors[lower:upper]))
        return pd.DataFrame(summaries, columns=MOTION_AUDIT_COLUMNS)


    def _motion_epoch(
        self, start: float, times: np.ndarray, vectors: np.ndarray,
    ) -> dict[str, Any]:
        """Report admissible changes and observed within-epoch pair gaps.

        Args:
            start (float): Completed interval's start time.
            times (np.ndarray): Unique sensor times inside the interval.
            vectors (np.ndarray): Axis vectors at those times, in chronological order.
        """
        gaps = np.diff(times)
        # Match the epoch-grid tolerance when decimal timestamps round near the limit
        valid = (gaps > 0) & (gaps <= self.max_motion_pair_gap_seconds + 1e-8)
        changes = np.linalg.norm(np.diff(vectors, axis=0)[valid], axis=1)
        if not np.isfinite(changes).all():
            raise ValueError("Motion vector changes must be finite")
        return {
            "time": start,
            "motion_vector_change_mean": float(changes.mean()) if len(changes) else np.nan,
            "motion_pair_count": int(valid.sum()),
            "motion_valid_pair_fraction": float(valid.mean()) if len(gaps) else np.nan,
            "motion_pair_gap_median_seconds": float(np.median(gaps)) if len(gaps) else np.nan,
            "motion_pair_gap_max_seconds": float(gaps.max()) if len(gaps) else np.nan,
        }
