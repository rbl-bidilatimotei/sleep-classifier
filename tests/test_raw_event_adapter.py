from random import Random
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sleep_classifier.features import EpochFeatureBuilder
from sleep_classifier.motion_grid import FixedLagMotionFeatures
from sleep_classifier.raw_event_adapter import LateEventError, RawEventAdapter

RULE = {
    "max_heart_gap_seconds": 15.0,
    "min_heart_readings": 3.0,
    "min_motion_bin_fraction": 0.8,
    "max_motion_gap_seconds": 5.0,
}


def _coverage_session(origin: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create covered intervals, an empty interval, and an internal motion gap.

    Args:
        origin (float): Session start in seconds.
    """
    heart, motion = [], []
    for epoch in (0, 1, 3, 4, 5):
        start = origin + 30 * epoch
        heart.extend((start + offset, 60.0 + epoch * 10 + offset) for offset in (0, 10, 20))
        for offset in range(30):
            if epoch == 3 and 9 <= offset < 17:
                continue
            lag = 0.125 if offset % 6 == 1 else 0.0
            motion.append((start + offset - lag, 1.0 + epoch, offset % 3, offset / 10))

    heart.insert(1, heart[0])
    motion.insert(3, motion[2])
    return (
        pd.DataFrame(heart, columns=["time", "heart_rate"]),
        pd.DataFrame(motion, columns=["time", "x", "y", "z"]),
    )


@pytest.mark.parametrize("max_chunk_size", [1, 13])
def test_rows_match_batch_coverage_and_fixed_lag_motion(max_chunk_size: int) -> None:
    """Match batch sensor rows across missing intervals and varied input chunk sizes.

    Args:
        max_chunk_size (int): Maximum number of records per input chunk.
    """
    origin = 12.25
    heart, motion = _coverage_session(origin)
    epochs = pd.DataFrame({"time": origin + 30 * np.arange(6)})
    expected = EpochFeatureBuilder(RULE).build_sensor_epochs(heart, motion, epochs)
    grid = FixedLagMotionFeatures().motion_summaries(motion, epochs)
    expected = expected.merge(grid, on="time", validate="one_to_one")
    expected = expected.assign(
        participant_id="person-a", available_at=expected.time + 30,
        watermark_at_emission=expected.time + 30,
    )

    adapter = RawEventAdapter("person-a", RULE, session_start_seconds=origin)
    random = Random(172)
    observed: list[dict[str, Any]] = []
    for start in epochs.time:
        end = start + 30
        for sensor, frame, add in (
            ("motion", motion, adapter.add_motion),
            ("heart_rate", heart, adapter.add_heart_rate),
        ):
            interval = frame.loc[(frame.time >= start) & (frame.time < end)]
            cursor = 0
            while cursor < len(interval):
                stop = cursor + random.randint(1, max_chunk_size)
                for sample in interval.iloc[cursor:stop].itertuples(index=False, name=None):
                    add(*sample)
                assert adapter.pop_completed() is None
                cursor = stop
            adapter.advance_watermark(sensor, end)
        row = adapter.pop_completed()
        assert row is not None
        observed.append(row)
        assert adapter.pop_completed() is None

    adapter.finish(origin + 180)
    assert adapter.pop_completed() is None
    actual = pd.DataFrame(observed)
    pd.testing.assert_frame_equal(actual, expected, check_like=True, rtol=0.0, atol=1e-8)
    assert actual.eligible.tolist() == [True, True, False, False, True, True]
    assert actual.segment_id.tolist() == [0, 0, -1, -1, 1, 1]
    assert adapter.status()["duplicate_events"] == 2


def test_waits_for_both_watermarks_and_keeps_boundary_events() -> None:
    """Keep events at the epoch end for the next epoch and wait for both watermarks."""
    adapter = RawEventAdapter("person-a", RULE)
    adapter.add_heart_rate(0.0, 60.0)
    adapter.add_heart_rate(30.0, 90.0)
    adapter.add_motion(0.0, 1.0, 0.0, 0.0)
    assert adapter.pop_completed() is None
    assert adapter.status()["watermarks"] == {"heart_rate": 0.0, "motion": 0.0}

    adapter.advance_watermark("heart_rate", 30.0)
    adapter.advance_watermark("motion", 29.99)
    assert adapter.pop_completed() is None
    adapter.advance_watermark("motion", 30.0)
    adapter.add_motion(30.0, 7.0, 0.0, 0.0)

    first = adapter.pop_completed()
    assert first is not None
    assert first["time"] == 0.0
    assert first["heart_rate_mean"] == 60.0
    assert first["motion_magnitude_mean"] == 1.0
    assert adapter.status()["buffered_events"] == 2
    assert adapter.pop_completed() is None

    adapter.finish(60.0)
    second = adapter.pop_completed()
    assert second is not None
    assert second["time"] == 30.0
    assert second["heart_rate_mean"] == 90.0
    assert second["motion_magnitude_mean"] == 7.0
    assert adapter.pop_completed() is None
    assert adapter.status()["discarded_tail_events"] == 0


def test_decimal_origin_anchors_boundaries_and_keeps_boundary_events() -> None:
    """Use the declared origin for every boundary and report the limiting watermark."""
    origin = 0.01
    adapter = RawEventAdapter("person-a", RULE, session_start_seconds=origin)
    for time, heart_rate, x in ((5.01, 60.0, 1.0), (35.01, 70.0, 2.0), (60.01, 90.0, 3.0)):
        adapter.add_heart_rate(time, heart_rate)
        adapter.add_motion(time, x, 0.0, 0.0)

    adapter.advance_watermark("heart_rate", 60.01)
    adapter.advance_watermark("motion", 90.01)

    for start, end, heart_rate, magnitude in (
        (0.01, 30.01, 60.0, 1.0), (30.01, 60.01, 70.0, 2.0),
    ):
        row = adapter.pop_completed()
        assert row is not None
        assert row["time"] == start
        assert row["available_at"] == end
        assert row["watermark_at_emission"] == 60.01
        assert row["available_at"] <= row["watermark_at_emission"]
        assert row["heart_sample_count"] == row["motion_sample_count"] == 1
        assert row["heart_rate_mean"] == heart_rate
        assert row["motion_magnitude_mean"] == magnitude
    assert adapter.pop_completed() is None
    assert adapter.status()["buffered_events"] == 2

    adapter.finish(90.01)
    row = adapter.pop_completed()
    assert row is not None
    assert row["time"] == 60.01
    assert row["available_at"] == row["watermark_at_emission"] == 90.01
    assert row["heart_sample_count"] == row["motion_sample_count"] == 1
    assert row["heart_rate_mean"] == 90.0
    assert row["motion_magnitude_mean"] == 3.0
    assert adapter.pop_completed() is None


def test_late_events_and_watermark_rewinds_preserve_state() -> None:
    """Reject late observations, including old duplicates, and preserve stream order."""
    adapter = RawEventAdapter("person-a", RULE, session_start_seconds=5.0)
    with pytest.raises(LateEventError):
        adapter.add_motion(4.0, 1.0, 0.0, 0.0)

    adapter.add_heart_rate(15.0, 60.0)
    adapter.add_motion(8.0, 1.0, 0.0, 0.0)
    adapter.advance_watermark("heart_rate", 16.0)
    snapshot = adapter.status()
    with pytest.raises(LateEventError):
        adapter.add_heart_rate(15.0, 60.0)
    with pytest.raises(LateEventError):
        adapter.add_motion(7.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        adapter.advance_watermark("heart_rate", 15.0)
    assert adapter.status() == snapshot

    adapter.add_heart_rate(16.0, 65.0)
    adapter.advance_watermark("heart_rate", 35.0)
    adapter.advance_watermark("motion", 35.0)
    assert adapter.pop_completed() is not None
    with pytest.raises(LateEventError):
        adapter.add_motion(34.0, 1.0, 0.0, 0.0)


def test_status_snapshot_changes_cannot_modify_adapter_state() -> None:
    """Return independent status and watermark mappings to the caller."""
    adapter = RawEventAdapter("person-a", RULE)
    adapter.add_heart_rate(1.0, 60.0)
    adapter.advance_watermark("motion", 30.0)

    snapshot = adapter.status()
    snapshot["buffered_events"] = 0
    snapshot["watermarks"]["heart_rate"] = 30.0
    assert adapter.status()["buffered_events"] == 1
    assert adapter.status()["watermarks"] == {"heart_rate": 0.0, "motion": 30.0}
    assert adapter.pop_completed() is None


def test_buffer_cap_counts_unique_events_across_sensors() -> None:
    """Apply backpressure before mutation and accept the same event after draining."""
    adapter = RawEventAdapter("person-a", RULE, max_buffered_events=2)
    adapter.add_heart_rate(1.0, 60.0)
    adapter.add_motion(4.0, 1.0, 0.0, 0.0)
    adapter.add_heart_rate(1.0, 60.0)
    adapter.add_motion(4.0, 1.0, 0.0, 0.0)
    snapshot = adapter.status()
    assert snapshot["duplicate_events"] == 2
    assert snapshot["buffered_events"] == snapshot["peak_buffered_events"] == 2

    with pytest.raises(ValueError):
        adapter.add_heart_rate(1.0, 61.0)
    with pytest.raises(ValueError):
        adapter.add_motion(4.0, 2.0, 0.0, 0.0)
    with pytest.raises(BufferError):
        adapter.add_heart_rate(32.0, 80.0)
    assert adapter.status() == snapshot

    adapter.advance_watermark("heart_rate", 30.0)
    adapter.advance_watermark("motion", 30.0)
    assert adapter.pop_completed() is not None
    adapter.add_heart_rate(31.0, 75.0)
    adapter.add_heart_rate(32.0, 80.0)
    assert adapter.status()["buffered_events"] == 2
    assert adapter.status()["duplicate_events"] == 2


def test_default_buffer_rejects_the_next_event_after_8192() -> None:
    """Limit unconsumed raw observations when watermarks do not advance."""
    adapter = RawEventAdapter("person-a", RULE)
    for time in range(8192):
        adapter.add_motion(time, 1.0, 0.0, 0.0)
    with pytest.raises(BufferError):
        adapter.add_heart_rate(0.0, 60.0)
    assert adapter.status()["buffered_events"] == 8192


def test_finish_drains_epochs_one_at_a_time_and_discards_tail() -> None:
    """Drain complete rows in order, count the partial interval, and close once."""
    adapter = RawEventAdapter("person-a", RULE)
    adapter.add_heart_rate(60.0, 67.0)
    adapter.add_motion(68.0, 1.0, 0.0, 0.0)
    adapter.finish(68.0)
    assert adapter.status()["closing"] is True
    assert adapter.status()["closed"] is False
    assert adapter.status()["emitted_epochs"] == 0
    snapshot = adapter.status()
    with pytest.raises(RuntimeError):
        adapter.add_heart_rate(68.0, 70.0)
    with pytest.raises(RuntimeError):
        adapter.add_motion(68.0, 2.0, 0.0, 0.0)
    with pytest.raises(RuntimeError):
        adapter.advance_watermark("motion", 68.0)
    with pytest.raises(RuntimeError):
        adapter.finish(68.0)
    assert adapter.status() == snapshot

    for emitted, start in enumerate((0.0, 30.0), start=1):
        row = adapter.pop_completed()
        assert row is not None
        assert row["time"] == start
        assert row["eligible"] is False
        assert row["segment_id"] == -1
        assert adapter.status()["emitted_epochs"] == emitted
        assert adapter.status()["next_epoch_start"] == start + 30

    assert adapter.pop_completed() is None
    snapshot = adapter.status()
    assert snapshot["closed"] is True
    assert snapshot["buffered_events"] == 0
    assert snapshot["discarded_tail_events"] == 2
    with pytest.raises(RuntimeError):
        adapter.finish(69.0)
    assert adapter.pop_completed() is None
    assert adapter.status() == snapshot


@pytest.mark.parametrize("value", [True, np.bool_(False), "7", None, np.inf, -np.inf, np.nan])
def test_invalid_numeric_event_inputs_preserve_state(value: Any) -> None:
    """Require finite real numbers for timestamps, sensor values, and watermarks.

    Args:
        value (Any): Invalid numeric input.
    """
    adapter = RawEventAdapter("person-a", RULE)
    snapshot = adapter.status()
    with pytest.raises(ValueError):
        adapter.add_heart_rate(value, 60.0)
    with pytest.raises(ValueError):
        adapter.add_heart_rate(10.0, value)
    with pytest.raises(ValueError):
        adapter.add_motion(value, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        adapter.add_motion(10.0, 1.0, 0.0, value)
    with pytest.raises(ValueError):
        adapter.advance_watermark("heart_rate", value)
    with pytest.raises(ValueError):
        adapter.finish(value)
    assert adapter.status() == snapshot


@pytest.mark.parametrize(
    "settings",
    [
        {"participant_id": ""}, {"participant_id": "  "}, {"participant_id": 123},
        {"session_start_seconds": True}, {"session_start_seconds": np.nan},
        {"max_buffered_events": True}, {"max_buffered_events": 1.5}, {"max_buffered_events": 0},
    ],
)
def test_invalid_session_settings(settings: dict[str, Any]) -> None:
    """Reject invalid participant identity, session origin, and buffer capacity.

    Args:
        settings (dict[str, Any]): Invalid option to pass to the adapter.
    """
    arguments = {"participant_id": "person-a", "rule": RULE} | settings
    with pytest.raises(ValueError):
        RawEventAdapter(**arguments)


def test_invalid_finish_and_unknown_sensor_preserve_state() -> None:
    """Reject unknown streams and end times before accepted progress."""
    adapter = RawEventAdapter("person-a", RULE, session_start_seconds=10.0)
    snapshot = adapter.status()
    with pytest.raises(ValueError):
        adapter.advance_watermark("unknown", 30.0)
    with pytest.raises(ValueError):
        adapter.finish(9.0)
    assert adapter.status() == snapshot

    adapter.advance_watermark("heart_rate", 20.0)
    snapshot = adapter.status()
    with pytest.raises(ValueError):
        adapter.finish(19.0)
    assert adapter.status() == snapshot

    adapter.add_motion(25.0, 1.0, 0.0, 0.0)
    snapshot = adapter.status()
    with pytest.raises(ValueError):
        adapter.finish(24.0)
    assert adapter.status() == snapshot
    adapter.finish(25.0)
    assert adapter.pop_completed() is None


def test_long_session_keeps_only_unconsumed_events_and_new_session_starts_clean() -> None:
    """Process more events than the default cap with constant per-epoch retention."""
    adapter = RawEventAdapter("person-a", RULE)
    epoch_count = 250
    for epoch in range(epoch_count):
        start = 30 * epoch
        for offset in (0, 10, 20):
            adapter.add_heart_rate(start + offset, 65.0)
        for offset in range(30):
            adapter.add_motion(start + offset, 1.0, 0.0, 0.0)
        adapter.advance_watermark("heart_rate", start + 30)
        adapter.advance_watermark("motion", start + 30)
        row = adapter.pop_completed()
        assert row is not None
        assert row["time"] == start
        assert row["segment_id"] == 0
        assert adapter.status()["buffered_events"] == 0

    adapter.finish(30 * epoch_count)
    assert adapter.pop_completed() is None
    assert adapter.status()["emitted_epochs"] == epoch_count
    assert adapter.status()["peak_buffered_events"] == 33

    next_session = RawEventAdapter("person-b", RULE)
    assert next_session.status()["emitted_epochs"] == 0
    assert next_session.status()["peak_buffered_events"] == 0
    next_session.finish(30.0)
    row = next_session.pop_completed()
    assert row is not None
    assert row["participant_id"] == "person-b"
    assert row["time"] == 0.0
    assert row["heart_sample_count"] == row["motion_sample_count"] == 0
