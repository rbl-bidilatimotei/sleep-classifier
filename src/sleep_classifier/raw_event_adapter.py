import math
from collections import deque
from numbers import Integral, Real
from typing import Any

import pandas as pd

from sleep_classifier.features import EpochFeatureBuilder
from sleep_classifier.motion_grid import FixedLagMotionFeatures

_SENSOR_COLUMNS = {
    "heart_rate": ("time", "heart_rate"),
    "motion": ("time", "x", "y", "z"),
}
_EPOCH_SECONDS = 30.0


class LateEventError(ValueError):
    """Indicate an event that precedes the accepted time for its sensor."""


class RawEventAdapter:
    """Build completed sensor epochs from bounded, timestamp-ordered event streams."""

    def __init__(
        self,
        participant_id: str,
        rule: dict[str, float],
        session_start_seconds: float = 0.0,
        max_buffered_events: int = 8192,
    ) -> None:
        """Declare one participant's session clock and buffer capacity.

        Args:
            participant_id (str): Nonempty participant identifier.
            rule (dict[str, float]): Sensor coverage thresholds for epoch eligibility.
            session_start_seconds (float): Timestamp of the first epoch start.
            max_buffered_events (int): Maximum pending unique events across both sensors.
        """
        if not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id must be a nonempty string")
        if (isinstance(max_buffered_events, bool) or not isinstance(max_buffered_events, Integral)
                or max_buffered_events <= 0):
            raise ValueError("max_buffered_events must be a positive integer")

        self._participant_id = participant_id
        self._session_start = self._finite_number(session_start_seconds, "session_start_seconds")
        first_end = self._session_start + _EPOCH_SECONDS
        if not math.isfinite(first_end) or first_end <= self._session_start:
            raise ValueError("session_start_seconds must support 30-second intervals")
        self._next_epoch_start = self._session_start
        self._max_buffered_events = int(max_buffered_events)
        self._feature_builder = EpochFeatureBuilder(rule)
        self._motion_builder = FixedLagMotionFeatures()
        self._buffers: dict[str, deque[tuple[float, ...]]] = {
            sensor: deque() for sensor in _SENSOR_COLUMNS
        }
        self._watermarks = dict.fromkeys(_SENSOR_COLUMNS, self._next_epoch_start)
        self._last_events: dict[str, tuple[float, ...]] = {}
        self._peak_buffered_events = 0
        self._duplicate_events = 0
        self._discarded_tail_events = 0
        self._emitted_epochs = 0
        self._segment_id = -1
        self._previous_eligible = False
        self._closing = False
        self._closed = False


    def add_heart_rate(self, time: float, heart_rate: float) -> None:
        """Add a heart-rate observation without advancing its watermark.

        Args:
            time (float): Observation timestamp in session seconds.
            heart_rate (float): Measured heart rate.
        """
        self._add_event("heart_rate", time, (heart_rate,))


    def add_motion(self, time: float, x: float, y: float, z: float) -> None:
        """Add an acceleration observation without advancing its watermark.

        Args:
            time (float): Observation timestamp in session seconds.
            x (float): Acceleration along the x axis.
            y (float): Acceleration along the y axis.
            z (float): Acceleration along the z axis.
        """
        self._add_event("motion", time, (x, y, z))


    def advance_watermark(self, sensor: str, time: float) -> None:
        """Declare that a sensor will send no more events before a timestamp.

        Args:
            sensor (str): The sensor stream, either heart_rate or motion.
            time (float): New watermark. Events at this timestamp stay admissible.
        """
        self._require_open()
        if not isinstance(sensor, str) or sensor not in self._watermarks:
            raise ValueError("sensor must be heart_rate or motion")
        time = self._finite_number(time, "time")
        if time < self._watermarks[sensor]:
            raise ValueError(f"{sensor} watermark cannot move backward")
        self._watermarks[sensor] = time


    def pop_completed(self) -> dict[str, Any] | None:
        """Return the next epoch once both watermarks reach its end.

        Returns:
            dict[str, Any] | None: One completed epoch, or None if no epoch is ready.
        """
        if self._closed:
            return None
        end = self._session_start + _EPOCH_SECONDS * (self._emitted_epochs + 1)
        watermark = min(self._watermarks.values())
        if end <= self._next_epoch_start or end > watermark:
            if self._closing:
                self._discard_tail()
            return None

        heart = self._epoch_frame("heart_rate", end)
        motion = self._epoch_frame("motion", end)
        summary = self._summarize_epoch(heart, motion, end)
        summary["watermark_at_emission"] = watermark

        # Commit only after both feature builders succeed
        for sensor, count in (("heart_rate", len(heart)), ("motion", len(motion))):
            for _ in range(count):
                self._buffers[sensor].popleft()
        if summary["eligible"]:
            self._segment_id = summary["segment_id"]
        self._previous_eligible = summary["eligible"]
        self._next_epoch_start = end
        self._emitted_epochs += 1
        return summary


    def finish(self, session_end_seconds: float) -> None:
        """Close input at a declared session end. Drain completed epochs with pop_completed.

        Args:
            session_end_seconds (float): End timestamp of the session, exclusive of observations.
        """
        self._require_open()
        end = self._finite_number(session_end_seconds, "session_end_seconds")
        if (end < self._next_epoch_start or end < max(self._watermarks.values())
                or any(end < event[0] for event in self._last_events.values())):
            raise ValueError("Session end cannot precede a watermark, accepted event, or epoch")
        for sensor in self._watermarks:
            self._watermarks[sensor] = end
        self._closing = True


    def status(self) -> dict[str, Any]:
        """Return a snapshot of buffer usage, watermarks, and session progress."""
        return {
            "buffered_events": self._buffered_events(),
            "peak_buffered_events": self._peak_buffered_events,
            "duplicate_events": self._duplicate_events,
            "discarded_tail_events": self._discarded_tail_events,
            "emitted_epochs": self._emitted_epochs,
            "watermarks": self._watermarks.copy(),
            "next_epoch_start": self._next_epoch_start,
            "closing": self._closing,
            "closed": self._closed,
        }


    @staticmethod
    def _finite_number(value: float, name: str) -> float:
        """Convert a finite real input to a float.

        Args:
            value (float): Input value to validate.
            name (str): Input name for the error message.
        """
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        try:
            result = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite real number") from error
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite real number")
        return result


    def _require_open(self) -> None:
        """Reject input after the caller declares a session end."""
        if self._closing:
            raise RuntimeError("Session input is closed")


    def _buffered_events(self) -> int:
        """Count pending unique events across both sensors."""
        return sum(len(buffer) for buffer in self._buffers.values())


    def _add_event(self, sensor: str, time: float, values: tuple[float, ...]) -> None:
        """Validate and retain a unique event in its sensor's timestamp order.

        Args:
            sensor (str): Sensor stream that supplied the event.
            time (float): Observation timestamp in session seconds.
            values (tuple[float, ...]): Observation values in sensor column order.
        """
        self._require_open()
        time = self._finite_number(time, "time")
        last = self._last_events.get(sensor)
        if (time < max(self._next_epoch_start, self._watermarks[sensor])
                or (last is not None and time < last[0])):
            raise LateEventError(f"{sensor} event time {time} precedes the accepted time")

        measurements = tuple(
            self._finite_number(value, name)
            for name, value in zip(_SENSOR_COLUMNS[sensor][1:], values, strict=True)
        )
        event = (time, *measurements)
        if last is not None and time == last[0]:
            if event != last:
                raise ValueError(f"Conflicting {sensor} events at time {time}")
            self._duplicate_events += 1
            return

        buffered = self._buffered_events()
        if buffered >= self._max_buffered_events:
            raise BufferError("Raw event buffer is full. Advance watermarks and drain epochs.")
        self._buffers[sensor].append(event)
        self._last_events[sensor] = event
        self._peak_buffered_events = max(self._peak_buffered_events, buffered + 1)


    def _epoch_frame(self, sensor: str, end: float) -> pd.DataFrame:
        """Copy one sensor's pending observations strictly before the epoch end.

        Args:
            sensor (str): Sensor stream to read.
            end (float): Exclusive epoch end timestamp.
        """
        rows = []
        for event in self._buffers[sensor]:
            if event[0] >= end:
                break
            rows.append(event)
        return pd.DataFrame(rows, columns=_SENSOR_COLUMNS[sensor], dtype=float)


    def _summarize_epoch(
        self, heart: pd.DataFrame, motion: pd.DataFrame, end: float,
    ) -> dict[str, Any]:
        """Build the output from the shared sensor and motion grid feature builders.

        Args:
            heart (pd.DataFrame): Heart-rate observations in the current epoch.
            motion (pd.DataFrame): Acceleration observations in the current epoch.
            end (float): Epoch end timestamp when the features become available.
        """
        epochs = pd.DataFrame({"time": [self._next_epoch_start]})
        summary = self._feature_builder.build_sensor_epochs(heart, motion, epochs)
        grid = self._motion_builder.motion_summaries(motion, epochs)
        row = summary.to_dict(orient="records")[0]
        row.update(grid.to_dict(orient="records")[0])
        row.update(participant_id=self._participant_id, available_at=end)
        if row["eligible"]:
            row["segment_id"] = self._segment_id + int(not self._previous_eligible)
        return row


    def _discard_tail(self) -> None:
        """Drop observations left after the last full epoch and close the session."""
        self._discarded_tail_events = self._buffered_events()
        for buffer in self._buffers.values():
            buffer.clear()
        self._closed = True
