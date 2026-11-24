import numpy as np
import pandas as pd
import pytest

from sleep_classifier.features import FEATURE_COLUMNS, EpochFeatureBuilder
from sleep_classifier.sensor_epochs import SessionEpochClock

RULE = {
    "max_heart_gap_seconds": 15,
    "min_heart_readings": 3,
    "min_motion_bin_fraction": 0.8,
    "max_motion_gap_seconds": 5,
}
COVERAGE_COLUMNS = [
    f"{sensor}_{measurement}"
    for sensor in ("heart", "motion")
    for measurement in ("sample_count", "occupied_bin_fraction", "largest_observation_gap")
]


def _sensors(end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create regular observations up to and including the specified endpoint.

    Args:
        end (int): Final observation time, a multiple of ten seconds.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Heart rate and motion frames.
    """
    heart_times = np.arange(0, end + 1, 10, dtype=float)
    motion_times = np.arange(end + 1, dtype=float)
    heart = pd.DataFrame({"time": heart_times, "heart_rate": 60 + heart_times / 10})
    motion = pd.DataFrame({"time": motion_times, "x": 1.0, "y": 0.0, "z": 0.0})
    return heart, motion


def test_sensor_eligibility_is_independent_of_labels() -> None:
    """Keep covered unscored epochs eligible only in the sensor view."""
    heart, motion = _sensors(90)
    heart = heart[heart.time >= 30]
    epochs = SessionEpochClock().completed_epochs(heart, motion)
    builder = EpochFeatureBuilder(RULE)

    sensors = builder.build_sensor_epochs(heart, motion, epochs)
    assert sensors.columns.tolist() == [
        "time", *FEATURE_COLUMNS, *COVERAGE_COLUMNS, "eligible", "segment_id"
    ]
    assert sensors.eligible.tolist() == [False, True, True]
    assert sensors.segment_id.tolist() == [-1, 0, 0]
    assert sensors.heart_sample_count.tolist() == [0, 3, 3]

    labels = epochs.assign(stage=[0, -1, 5])
    labeled = builder.build(heart, motion, labels)
    assert labeled.target.tolist() == [0, -1, 3]
    assert labeled.eligible.tolist() == [False, False, True]
    assert labeled.segment_id.tolist() == [-1, -1, 0]
    shared = ["time", *FEATURE_COLUMNS, *COVERAGE_COLUMNS]
    pd.testing.assert_frame_equal(sensors[shared], labeled[shared])


@pytest.mark.parametrize("column", ["stage", "target"])
def test_sensor_builder_requires_time_only(column: str) -> None:
    """Reject sleep stages and targets in a sensor-only schedule.

    Args:
        column (str): Sleep label column to add to the epoch schedule.
    """
    heart, motion = _sensors(30)
    epochs = pd.DataFrame({"time": [0.0], column: [1]})
    with pytest.raises(ValueError):
        EpochFeatureBuilder(RULE).build_sensor_epochs(heart, motion, epochs)


@pytest.mark.parametrize("sensor", ["heart", "motion"])
def test_duplicate_observations_follow_existing_cleaning_rule(sensor: str) -> None:
    """Collapse identical observations and reject conflicting readings.

    Args:
        sensor (str): Name of the frame that receives the duplicate.
    """
    heart, motion = _sensors(30)
    epochs = SessionEpochClock().completed_epochs(heart, motion)
    builder = EpochFeatureBuilder(RULE)
    original = builder.build_sensor_epochs(heart, motion, epochs)
    frames = {"heart": heart, "motion": motion}
    duplicate = frames[sensor].iloc[:1].copy()
    frames[sensor] = pd.concat([frames[sensor], duplicate], ignore_index=True)

    pd.testing.assert_frame_equal(
        original, builder.build_sensor_epochs(frames["heart"], frames["motion"], epochs)
    )

    column = "heart_rate" if sensor == "heart" else "x"
    frames[sensor].loc[frames[sensor].index[-1], column] = 500.0
    with pytest.raises(ValueError, match="Conflicting sensor values"):
        builder.build_sensor_epochs(frames["heart"], frames["motion"], epochs)


def test_clock_aligns_to_explicit_origin_and_uses_both_sensors() -> None:
    """Schedule complete intervals from the stated origin and respect their edges."""
    heart = pd.DataFrame(
        {"time": [0.0, 12.5, 42.0, 42.5, 72.5, 102.5], "heart_rate": [999, 60, 70, 80, 90, 100]}
    )
    motion = pd.DataFrame(
        {"time": [0.0, 12.5, 42.0, 42.5, 72.5], "x": [999, 1, 3, 7, 9], "y": 0, "z": 0}
    )

    epochs = SessionEpochClock(start_seconds=12.5).completed_epochs(heart, motion)
    pd.testing.assert_frame_equal(epochs, pd.DataFrame({"time": [12.5, 42.5]}))

    result = EpochFeatureBuilder(RULE).build_sensor_epochs(heart, motion, epochs)
    assert result.heart_rate_mean.tolist() == [65.0, 80.0]
    assert result.motion_magnitude_mean.tolist() == [2.0, 7.0]
    assert result.heart_sample_count.tolist() == [2, 1]
    assert result.motion_sample_count.tolist() == [2, 1]


def test_future_changes_preserve_completed_epoch_rows() -> None:
    """Keep all earlier rows stable after changing and extending future observations."""
    heart, motion = _sensors(60)
    clock = SessionEpochClock()
    builder = EpochFeatureBuilder(RULE)
    initial = builder.build_sensor_epochs(heart, motion, clock.completed_epochs(heart, motion))

    future_heart, future_motion = _sensors(90)
    future_heart.loc[future_heart.time >= 60, "heart_rate"] = 200.0
    future_motion.loc[future_motion.time >= 60, "x"] = 500.0
    expanded = builder.build_sensor_epochs(
        future_heart, future_motion, clock.completed_epochs(future_heart, future_motion)
    )

    assert expanded.time.tolist() == [0.0, 30.0, 60.0]
    pd.testing.assert_frame_equal(initial, expanded.iloc[: len(initial)])


@pytest.mark.parametrize("times", [[], [0.0, 29.99]])
def test_clock_requires_a_completed_interval(times: list[float]) -> None:
    """Reject sessions with insufficient support in either sensor.

    Args:
        times (list[float]): Empty or short sensor timeline.
    """
    supported = pd.DataFrame({"time": [0.0, 30.0]})
    insufficient = pd.DataFrame({"time": times})
    clock = SessionEpochClock()
    with pytest.raises(ValueError):
        clock.completed_epochs(insufficient, supported)
    with pytest.raises(ValueError):
        clock.completed_epochs(supported, insufficient)


@pytest.mark.parametrize("origin", [np.nan, np.inf, -np.inf, True, False])
def test_clock_rejects_invalid_origins(origin: float | bool) -> None:
    """Reject nonfinite or Boolean clock origins.

    Args:
        origin (float | bool): Invalid epoch origin.
    """
    with pytest.raises(ValueError):
        SessionEpochClock(start_seconds=origin)


@pytest.mark.parametrize("bad_time", [np.nan, np.inf, -np.inf])
def test_clock_rejects_nonfinite_sensor_times(bad_time: float) -> None:
    """Validate every timestamp in both raw sensor frames.

    Args:
        bad_time (float): Nonfinite observation timestamp.
    """
    supported = pd.DataFrame({"time": [0.0, 30.0]})
    invalid = pd.DataFrame({"time": [0.0, bad_time, 30.0]})
    clock = SessionEpochClock()
    with pytest.raises(ValueError):
        clock.completed_epochs(invalid, supported)
    with pytest.raises(ValueError):
        clock.completed_epochs(supported, invalid)
