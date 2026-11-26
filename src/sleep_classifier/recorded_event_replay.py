from heapq import merge
from operator import itemgetter
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.raw_event_adapter import RawEventAdapter


class RecordedEventReplay:
    """Simulate declared watermarks from sorted recorded events for an offline adapter audit."""

    def __init__(self, rule: dict[str, float], start: float, capacity: int) -> None:
        """Retain the fixed coverage, clock, and buffer settings.

        Args:
            rule (dict[str, float]): Frozen sensor coverage thresholds.
            start (float): Explicit session-relative clock origin.
            capacity (int): Combined pending-event limit for each new adapter.
        """
        self.rule = rule
        self.start = start
        self.capacity = capacity


    @staticmethod
    def _sensor_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Validate sensor-only columns and sort a copy without changing the raw source.

        Args:
            frame (pd.DataFrame): One sensor's recorded observations.
            columns (list[str]): Exact raw schema, without annotations or predictions.
        """
        if not frame.columns.is_unique or set(frame) != set(columns):
            raise ValueError("Recorded replay requires exact sensor-only columns")
        rows = frame[columns].astype(float)
        if rows.empty or not np.isfinite(rows.to_numpy()).all():
            raise ValueError("Recorded sensor observations must be nonempty and finite")
        return rows.sort_values("time", kind="stable")


    def run(
        self, participant_id: str, heart: pd.DataFrame, motion: pd.DataFrame,
        chunk_size: int, session_end: float,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Feed ordered events and consume completed epochs after each chunk watermark.

        Args:
            participant_id (str): Explicit development participant identifier.
            heart (pd.DataFrame): Checksum-verified recorded heart-rate observations.
            motion (pd.DataFrame): Checksum-verified recorded acceleration observations.
            chunk_size (int): Event count between watermark declarations.
            session_end (float): Conservative declared endpoint, independent of labels.
        """
        if type(chunk_size) is not int or not 0 < chunk_size <= self.capacity:
            raise ValueError("chunk_size must be a positive integer within the buffer limit")
        heart = self._sensor_rows(heart, ["time", "heart_rate"])
        motion = self._sensor_rows(motion, ["time", "x", "y", "z"])
        if (isinstance(session_end, bool) or not np.isfinite(session_end)
                or not self.start <= session_end <= min(heart.time.max(), motion.time.max())):
            raise ValueError("Recorded session end must remain within both observation streams")
        before = int(heart.time.lt(self.start).sum()+motion.time.lt(self.start).sum())
        after = int(heart.time.gt(session_end).sum()+motion.time.gt(session_end).sum())
        heart = heart.loc[heart.time.between(self.start, session_end)]
        motion = motion.loc[motion.time.between(self.start, session_end)]
        adapter = RawEventAdapter(participant_id, self.rule, self.start, self.capacity)
        heart_events = ((time, "heart_rate", (value,))
                        for time, value in heart.itertuples(index=False, name=None))
        motion_events = ((time, "motion", (x, y, z))
                         for time, x, y, z in motion.itertuples(index=False, name=None))
        rows: list[dict[str, Any]] = []
        count = 0
        for count, (time, sensor, values) in enumerate(
            merge(heart_events, motion_events, key=itemgetter(0)), start=1,
        ):
            if sensor == "heart_rate":
                adapter.add_heart_rate(time, values[0])
            else:
                adapter.add_motion(time, *values)
            if count % chunk_size == 0:
                self._advance(adapter, time, rows)
        adapter.finish(session_end)
        self._drain(adapter, rows)
        status = adapter.status() | {"input_events": count,
                                     "filtered_pre_session_events": before,
                                     "filtered_after_session_events": after}
        return pd.DataFrame(rows), status


    @staticmethod
    def _advance(adapter: RawEventAdapter, watermark: float, rows: list[dict[str, Any]]) -> None:
        """Declare replay progress only after all events strictly before it have arrived.

        Args:
            adapter (RawEventAdapter): Active bounded adapter.
            watermark (float): Last globally merged event time; equal-time events remain admissible.
            rows (list[dict[str, Any]]): Offline audit consumer, separate from adapter state.
        """
        adapter.advance_watermark("heart_rate", watermark)
        adapter.advance_watermark("motion", watermark)
        RecordedEventReplay._drain(adapter, rows)


    @staticmethod
    def _drain(adapter: RawEventAdapter, rows: list[dict[str, Any]]) -> None:
        """Consume one completed output at a time without growing adapter state.

        Args:
            adapter (RawEventAdapter): Adapter whose declared watermarks may close epochs.
            rows (list[dict[str, Any]]): Offline consumer's recorded feature outputs.
        """
        while (row := adapter.pop_completed()) is not None:
            rows.append(row)
