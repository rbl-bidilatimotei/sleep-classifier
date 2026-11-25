from typing import Any

import numpy as np
import pandas as pd
import pytest

from sleep_classifier.features import EpochFeatureBuilder
from sleep_classifier.motion_grid import FixedLagMotionFeatures
from sleep_classifier.recorded_event_replay import RecordedEventReplay

RULE = {
    "max_heart_gap_seconds": 15.0,
    "min_heart_readings": 3.0,
    "min_motion_bin_fraction": 0.8,
    "max_motion_gap_seconds": 5.0,
}


@pytest.mark.parametrize("chunk_size", [128, 1024])
def test_recorded_chunks_match_batch_with_session_filters(chunk_size: int) -> None:
    """Match all batch fields and preserve the raw frames with either chunk size.

    Args:
        chunk_size (int): Number of events between replay watermarks.
    """
    origin, end = 10.5, 102.5
    heart_times = np.array([origin - 1, *np.arange(origin, end, 10), end, end + 1])
    motion_times = np.array([origin - 2, *np.arange(origin, end, 0.0625), end, end + 2])
    heart = pd.DataFrame({"time": heart_times, "heart_rate": 60 + heart_times / 10})
    motion = pd.DataFrame({
        "time": motion_times, "x": motion_times / 10, "y": np.sin(motion_times), "z": 1.0,
    })
    epochs = pd.DataFrame({"time": origin + 30 * np.arange(3)})
    expected = EpochFeatureBuilder(RULE).build_sensor_epochs(heart, motion, epochs)
    grid = FixedLagMotionFeatures().motion_summaries(motion, epochs)
    expected = expected.merge(grid, on="time", validate="one_to_one").assign(
        participant_id="person-a", available_at=epochs.time + 30,
    )
    reversed_heart, reversed_motion = heart.iloc[::-1].copy(), motion.iloc[::-1].copy()

    actual, status = RecordedEventReplay(RULE, origin, 8192).run(
        "person-a", reversed_heart, reversed_motion, chunk_size, end,
    )

    assert actual.watermark_at_emission.ge(actual.available_at).all()
    assert actual.watermark_at_emission.le(end).all()
    pd.testing.assert_frame_equal(actual.drop(columns="watermark_at_emission"), expected,
                                  check_like=True, rtol=0.0, atol=1e-8)
    pd.testing.assert_frame_equal(reversed_heart, heart.iloc[::-1])
    pd.testing.assert_frame_equal(reversed_motion, motion.iloc[::-1])
    assert status["closed"] is True
    assert status["buffered_events"] == 0
    assert status["emitted_epochs"] == 3
    assert 0 < status["peak_buffered_events"] <= 8192
    assert status["filtered_pre_session_events"] == status["filtered_after_session_events"] == 2
    assert status["input_events"] == len(heart) + len(motion) - 4
    assert status["discarded_tail_events"] == 35


@pytest.mark.parametrize("duplicate_value", [70.0, 71.0])
def test_equal_timestamp_crosses_chunk_watermark(duplicate_value: float) -> None:
    """Accept identical readings at the watermark and reject conflicting readings.

    Args:
        duplicate_value (float): Heart-rate value on the second side of the chunk boundary.
    """
    heart = pd.DataFrame({"time": [0.0, 30.0, 30.0, 60.0],
                          "heart_rate": [60.0, 70.0, duplicate_value, 80.0]})
    motion = pd.DataFrame({"time": [0.0, 30.0, 60.0], "x": [1.0, 2.0, 3.0],
                           "y": 0.0, "z": 0.0})
    replay = RecordedEventReplay(RULE, 0.0, 12)
    if duplicate_value != 70.0:
        with pytest.raises(ValueError, match="Conflicting heart_rate events"):
            replay.run("person-a", heart, motion, 3, 60.0)
        return

    actual, status = replay.run("person-a", heart, motion, 3, 60.0)
    assert actual.time.tolist() == [0.0, 30.0]
    assert actual.watermark_at_emission.tolist() == [30.0, 60.0]
    assert actual.heart_sample_count.tolist() == actual.motion_sample_count.tolist() == [1, 1]
    assert actual.heart_rate_mean.tolist() == [60.0, 70.0]
    assert actual.motion_magnitude_mean.tolist() == [1.0, 2.0]
    assert status["input_events"] == 7
    assert status["duplicate_events"] == 1
    assert status["discarded_tail_events"] == 2


def test_replay_propagates_backpressure_before_an_epoch_can_close() -> None:
    """Fail when dense input fills the declared cap before a complete epoch exists."""
    heart = pd.DataFrame({"time": [0.0, 30.0], "heart_rate": [60.0, 70.0]})
    motion = pd.DataFrame({"time": [0.5, 1.0, 30.0], "x": 1.0, "y": 0.0, "z": 0.0})
    with pytest.raises(BufferError):
        RecordedEventReplay(RULE, 0.0, 2).run("person-a", heart, motion, 2, 30.0)


@pytest.mark.parametrize("chunk_size", [0, 3, True, 1.5])
def test_replay_requires_chunk_size_within_capacity(chunk_size: Any) -> None:
    """Reject invalid chunk sizes before opening the sensor tables.

    Args:
        chunk_size (Any): Invalid event count or a count that exceeds the cap.
    """
    with pytest.raises(ValueError, match="chunk_size"):
        RecordedEventReplay(RULE, 0.0, 2).run(
            "person-a", pd.DataFrame(), pd.DataFrame(), chunk_size, 30.0,
        )


def test_replay_rejects_annotated_sensor_input() -> None:
    """Reject annotations at the recorded raw event boundary."""
    heart = pd.DataFrame({"time": [0.0, 30.0], "heart_rate": [60.0, 70.0], "target": [0, 1]})
    motion = pd.DataFrame({"time": [0.0, 30.0], "x": 1.0, "y": 0.0, "z": 0.0})
    with pytest.raises(ValueError, match="sensor-only columns"):
        RecordedEventReplay(RULE, 0.0, 8192).run("person-a", heart, motion, 128, 30.0)
