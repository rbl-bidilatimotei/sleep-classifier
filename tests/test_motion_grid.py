import numpy as np
import pandas as pd
import pytest

from sleep_classifier.motion_grid import MOTION_GRID_COLUMNS, FixedLagMotionFeatures


def _motion(
    times: list[float] | np.ndarray, x: list[float] | np.ndarray | None = None
) -> pd.DataFrame:
    """Create synthetic raw motion with optional changing x values.

    Args:
        times (list[float] | np.ndarray): Raw observation timestamps.
        x (list[float] | np.ndarray | None): Optional x values, otherwise zero.

    Returns:
        pd.DataFrame: Raw observations with all required columns.
    """
    return pd.DataFrame({"time": times, "x": np.zeros(len(times)) if x is None else x,
                         "y": np.zeros(len(times)), "z": np.zeros(len(times))})


def test_default_schedule_retains_zero_and_missing_epochs() -> None:
    """Retain every target row and distinguish observed zero from missing motion."""
    builder = FixedLagMotionFeatures()
    epochs = pd.DataFrame({"time": [0.0, 30.0, 90.0], "target": [0, 1, 2]})
    result = builder.motion_summaries(_motion(np.arange(1.0, 30.0)), epochs)

    assert builder.step_seconds == 1.0
    assert builder.max_sample_age_seconds == 0.2
    assert builder.points_per_epoch == 29
    assert builder.pairs_per_epoch == 28
    assert tuple(result.columns) == MOTION_GRID_COLUMNS
    assert result.time.tolist() == epochs.time.tolist()
    assert result.motion_grid_point_count.tolist() == [29, 0, 0]
    assert result.motion_grid_pair_count.tolist() == [28, 0, 0]
    assert result.motion_grid_point_fraction.tolist() == [1, 0, 0]
    assert result.motion_grid_pair_fraction.tolist() == [1, 0, 0]
    assert result.loc[0, "motion_grid_change_mean"] == 0.0
    assert result.loc[1:, "motion_grid_change_mean"].isna().all()
    assert result.loc[0, "motion_grid_age_max_seconds"] == 0.0
    for column in MOTION_GRID_COLUMNS[-3:]:
        assert result.loc[0, column] == 1.0
        assert result.loc[1:, column].isna().all()
    assert result.loc[1:, "motion_grid_age_max_seconds"].isna().all()


def test_equal_magnitudes_keep_direction_changes() -> None:
    """Compute axis-vector differences when scalar magnitudes remain constant."""
    motion = _motion([1.0, 2.0], [1.0, 0.0]).assign(y=[0.0, 1.0])
    result = FixedLagMotionFeatures().motion_summaries(motion, pd.DataFrame({"time": [0]}))

    assert np.linalg.norm(motion[["x", "y", "z"]], axis=1).tolist() == [1, 1]
    assert result.loc[0, "motion_grid_change_mean"] == pytest.approx(np.sqrt(2))
    assert result.loc[0, "motion_grid_pair_count"] == 1
    assert result.loc[0, "motion_grid_pair_fraction"] == pytest.approx(1 / 28)


def test_interior_ticks_exclude_both_epoch_endpoints() -> None:
    """Exclude boundary ticks and prevent pairs between successive completed intervals."""
    motion = _motion([0.0, 1.0, 2.0, 29.0, 30.0, 31.0, 32.0, 60.0],
                     [999.0, 0.0, 1.0, 999.0, 999.0, 10.0, 12.0, 999.0])
    result = FixedLagMotionFeatures(max_sample_age_seconds=0).motion_summaries(
        motion, pd.DataFrame({"time": [0.0, 30.0]})
    )

    assert result.motion_grid_point_count.tolist() == [3, 2]
    assert result.motion_grid_pair_count.tolist() == [1, 1]
    assert result.motion_grid_change_mean.tolist() == [1.0, 2.0]


def test_latest_past_observation_is_selected_without_interpolation() -> None:
    """Select the latest observed vector without borrowing a future interpolation endpoint."""
    motion = _motion([0.8, 0.9, 1.1, 1.9, 2.1], [999.0, 2.0, 999.0, 5.0, 999.0])
    result = FixedLagMotionFeatures().motion_summaries(motion, pd.DataFrame({"time": [0]}))

    assert result.loc[0, "motion_grid_change_mean"] == 3.0
    assert result.loc[0, "motion_grid_point_count"] == 2
    assert result.loc[0, "motion_grid_pair_count"] == 1
    assert result.loc[0, "motion_grid_age_max_seconds"] == pytest.approx(0.1)
    assert result.loc[0, "motion_grid_actual_lag_median_seconds"] == pytest.approx(1.0)


def test_future_timestamp_has_no_rounding_tolerance() -> None:
    """Reject a future observation even when its lead is below the age tolerance."""
    motion = _motion([1.0 + 1e-12, 2.0], [0.0, 10.0])
    result = FixedLagMotionFeatures().motion_summaries(motion, pd.DataFrame({"time": [0]}))

    assert result.loc[0, "motion_grid_point_count"] == 1
    assert result.loc[0, "motion_grid_pair_count"] == 0
    assert np.isnan(result.loc[0, "motion_grid_change_mean"])


@pytest.mark.parametrize("age, expected", [(0.2, 1), (0.2 + 5e-9, 1), (0.2 + 2e-8, 0)])
def test_staleness_accepts_only_documented_rounding_tolerance(age: float, expected: int) -> None:
    """Apply the small numerical allowance only to the past observation's age.

    Args:
        age (float): Age at the first scheduled tick.
        expected (int): Expected supported point count.
    """
    result = FixedLagMotionFeatures().motion_summaries(
        _motion([1 - age]), pd.DataFrame({"time": [0]})
    )
    assert result.loc[0, "motion_grid_point_count"] == expected
    assert result.loc[0, "motion_grid_pair_count"] == 0


def test_missing_ticks_do_not_form_a_longer_pair() -> None:
    """Keep a missing interior tick from joining two nonadjacent supported ticks."""
    result = FixedLagMotionFeatures().motion_summaries(
        _motion([1.0, 3.0], [0.0, 5.0]), pd.DataFrame({"time": [0]})
    )

    assert result.loc[0, "motion_grid_point_count"] == 2
    assert result.loc[0, "motion_grid_pair_count"] == 0
    assert np.isnan(result.loc[0, "motion_grid_change_mean"])
    assert result.loc[0, list(MOTION_GRID_COLUMNS[-3:])].isna().all()


def test_no_previous_epoch_observation_is_carried_forward() -> None:
    """Reject previous-epoch readings even when the age rounding allowance permits them."""
    builder = FixedLagMotionFeatures(step_seconds=1, max_sample_age_seconds=1 - 5e-9)
    result = builder.motion_summaries(_motion([29.999999999]), pd.DataFrame({"time": [30]}))

    assert result.loc[0, "motion_grid_point_count"] == 0
    assert result.loc[0, "motion_grid_pair_count"] == 0


def test_same_raw_timestamp_cannot_form_a_valid_pair() -> None:
    """Exclude repeated selected timestamps allowed only by the tiny age allowance."""
    builder = FixedLagMotionFeatures(max_sample_age_seconds=1 - 5e-9)
    result = builder.motion_summaries(_motion([1.0]), pd.DataFrame({"time": [0]}))

    assert result.loc[0, "motion_grid_pair_count"] == 0
    assert np.isnan(result.loc[0, "motion_grid_change_mean"])


def test_actual_lags_describe_selected_raw_pairs() -> None:
    """Report actual source lags rather than substituting the configured grid spacing."""
    result = FixedLagMotionFeatures().motion_summaries(
        _motion([0.8, 1.9, 2.8], [0.0, 2.0, 6.0]), pd.DataFrame({"time": [0]})
    )

    assert result.loc[0, "motion_grid_change_mean"] == 3.0
    assert result.loc[0, "motion_grid_age_max_seconds"] == pytest.approx(0.2)
    assert result.loc[0, "motion_grid_actual_lag_median_seconds"] == pytest.approx(1.0)
    assert result.loc[0, "motion_grid_actual_lag_min_seconds"] == pytest.approx(0.9)
    assert result.loc[0, "motion_grid_actual_lag_max_seconds"] == pytest.approx(1.1)


def test_duplicate_policy_sorts_exact_duplicates_and_rejects_conflicts() -> None:
    """Apply the public duplicate policy before selecting raw observations."""
    builder = FixedLagMotionFeatures()
    motion = _motion([2.0, 1.0, 1.0], [3.0, 1.0, 1.0])
    original = motion.copy(deep=True)
    result = builder.motion_summaries(motion, pd.DataFrame({"time": [0]}))
    assert result.loc[0, "motion_grid_change_mean"] == 2.0
    pd.testing.assert_frame_equal(motion, original)

    motion.loc[2, "x"] = 2.0
    with pytest.raises(ValueError, match="Conflicting"):
        builder.motion_summaries(motion, pd.DataFrame({"time": [0]}))


def test_future_append_does_not_change_completed_summaries() -> None:
    """Keep completed summaries unchanged when later epochs and readings are appended."""
    builder = FixedLagMotionFeatures()
    motion = _motion([1.0, 2.0, 31.0, 32.0], [1.0, 2.0, 999.0, 0.0])
    epochs = pd.DataFrame({"time": [0.0, 30.0]})
    prefix = builder.motion_summaries(motion.iloc[:2], epochs.iloc[:1])
    full = builder.motion_summaries(motion, epochs)

    pd.testing.assert_frame_equal(prefix, full.iloc[:1].reset_index(drop=True))


@pytest.mark.parametrize("step, points", [(0.5, 59), (2.0, 14), (7.0, 4), (14.0, 2)])
def test_general_schedule_uses_only_interior_ticks(step: float, points: int) -> None:
    """Use the configured spacing and divide support by the complete schedule.

    Args:
        step (float): Scheduled tick spacing.
        points (int): Expected count of strictly interior grid ticks.
    """
    builder = FixedLagMotionFeatures(step_seconds=step, max_sample_age_seconds=0)
    result = builder.motion_summaries(
        _motion([step, 2 * step], [0.0, 10.0]), pd.DataFrame({"time": [0]})
    )

    assert builder.points_per_epoch == points
    assert builder.pairs_per_epoch == points - 1
    assert result.loc[0, "motion_grid_change_mean"] == 10.0
    assert result.loc[0, "motion_grid_point_fraction"] == pytest.approx(2 / points)
    assert result.loc[0, "motion_grid_pair_fraction"] == pytest.approx(1 / (points - 1))
    assert result.loc[0, "motion_grid_actual_lag_median_seconds"] == step


def test_fixed_lag_changes_match_across_raw_sampling_rates() -> None:
    """Remove sample-spacing dependence for an ideal fixed linear axis trajectory."""
    builder = FixedLagMotionFeatures()
    summaries: list[float] = []
    adjacent_changes: list[float] = []
    for spacing in [0.02, 0.1]:
        times = np.arange(round(30 / spacing)) * spacing
        motion = _motion(times, times * 2)
        adjacent_changes.append(float(np.diff(motion.x).mean()))
        result = builder.motion_summaries(motion, pd.DataFrame({"time": [0]}))
        summaries.append(float(result.loc[0, "motion_grid_change_mean"]))

    assert adjacent_changes == pytest.approx([0.04, 0.2])
    assert summaries == pytest.approx([2.0, 2.0])


@pytest.mark.parametrize("times", [[0, 15], [30, 0], [0, 0], [0, np.inf], [0, np.nan], [0, 31]])
def test_invalid_epoch_starts_are_rejected(times: list[float]) -> None:
    """Reject overlapping, repeated, unordered, nonfinite, or off-grid target starts.

    Args:
        times (list[float]): Invalid target timestamps.
    """
    with pytest.raises(ValueError):
        FixedLagMotionFeatures().motion_summaries(_motion([]), pd.DataFrame({"time": times}))


@pytest.mark.parametrize("column", ["time", "x", "y", "z"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_motion_values_are_rejected(column: str, value: float) -> None:
    """Validate every raw timestamp and axis before selecting tick support.

    Args:
        column (str): Raw field to invalidate.
        value (float): Nonfinite value to insert.
    """
    motion = _motion([1.0])
    motion.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        FixedLagMotionFeatures().motion_summaries(motion, pd.DataFrame({"time": [0]}))


@pytest.mark.parametrize("column", ["time", "x", "y", "z"])
def test_missing_motion_columns_are_rejected(column: str) -> None:
    """Require the complete raw observation schema even for empty input.

    Args:
        column (str): Required raw field to remove.
    """
    with pytest.raises(ValueError, match="missing columns"):
        FixedLagMotionFeatures().motion_summaries(
            _motion([]).drop(columns=column), pd.DataFrame({"time": [0]})
        )


def test_missing_epoch_time_column_is_rejected() -> None:
    """Require target timestamps before calculating a grid."""
    with pytest.raises(ValueError, match="time column"):
        FixedLagMotionFeatures().motion_summaries(_motion([]), pd.DataFrame())


def test_empty_epochs_keep_the_exact_output_schema() -> None:
    """Return an empty table with the documented columns for an empty target schedule."""
    result = FixedLagMotionFeatures().motion_summaries(_motion([]), pd.DataFrame({"time": []}))
    assert result.empty
    assert tuple(result.columns) == MOTION_GRID_COLUMNS


@pytest.mark.parametrize("settings", [
    {"step_seconds": True}, {"step_seconds": np.bool_(False)}, {"step_seconds": "1"},
    {"step_seconds": np.nan}, {"step_seconds": np.inf}, {"step_seconds": 0},
    {"step_seconds": -1}, {"step_seconds": 15}, {"step_seconds": 30},
    {"max_sample_age_seconds": True}, {"max_sample_age_seconds": np.nan},
    {"max_sample_age_seconds": np.inf}, {"max_sample_age_seconds": -0.1},
    {"max_sample_age_seconds": 1.0}, {"max_sample_age_seconds": 2.0},
])
def test_invalid_grid_settings_are_rejected(settings: dict[str, float | bool | str]) -> None:
    """Reject ambiguous, nonfinite, or unsupported grid configuration values.

    Args:
        settings (dict[str, float | bool | str]): Invalid constructor arguments.
    """
    with pytest.raises(ValueError):
        FixedLagMotionFeatures(**settings)



def test_grid_duplicate_policy_uses_only_numeric_sensor_columns() -> None:
    """Ignore unrelated metadata while applying the existing exact sensor-duplicate rule."""
    frame = pd.DataFrame({"time": ["1", "1", "2"], "x": ["0", "0", "1"],
                          "y": ["0", "0", "0"], "z": ["1", "1", "1"],
                          "unrelated_note": ["first", "second", "third"]})
    result = FixedLagMotionFeatures().motion_summaries(frame, pd.DataFrame({"time": [0.0]}))
    assert result.iloc[0].motion_grid_pair_count == 1
    assert result.iloc[0].motion_grid_change_mean == pytest.approx(1.0)
