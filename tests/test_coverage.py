"""Contracts for sensor and epoch coverage summaries."""

import pandas as pd
import pytest

from sleep_classifier.coverage import deduplicate_sensor, epoch_coverage, step_bins


def test_deduplicate_sensor_sorts_and_removes_exact_duplicate_rows():
    frame = pd.DataFrame({"time": [2, 1, 1], "heart_rate": [70, 60, 60]})
    result = deduplicate_sensor(frame)
    assert result.to_dict("records") == [
        {"time": 1, "heart_rate": 60},
        {"time": 2, "heart_rate": 70},
    ]
    assert frame.time.tolist() == [2, 1, 1]


def test_deduplicate_sensor_rejects_conflicting_values_at_same_time():
    frame = pd.DataFrame({"time": [1, 1], "heart_rate": [60, 61]})
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate_sensor(frame)


def test_epoch_coverage_uses_half_open_intervals_and_unique_samples():
    frame = pd.DataFrame({"time": [30, 0, 0, 29, 60], "heart_rate": [70, 60, 60, 80, 75]})
    labels = pd.DataFrame({"time": [0, 30]})
    result = epoch_coverage(frame, labels)
    assert result.sample_count.tolist() == [2, 1]
    assert result.occupied_bin_fraction.tolist() == pytest.approx([2 / 30, 1 / 30])
    assert result.largest_observation_gap.tolist() == [29, 30]


def test_epoch_coverage_counts_exact_boundary_and_excludes_epoch_end():
    frame = pd.DataFrame({"time": [0, 29, 30, 59, 60], "heart_rate": [1, 2, 3, 4, 5]})
    result = epoch_coverage(frame, pd.DataFrame({"time": [0, 30]}))
    assert result.sample_count.tolist() == [2, 2]
    assert result.largest_observation_gap.tolist() == [29, 29]


def test_epoch_coverage_empty_epoch_reports_zero_bins_and_full_gap():
    result = epoch_coverage(
        pd.DataFrame({"time": [], "heart_rate": []}), pd.DataFrame({"time": [30]})
    )
    assert result.loc[0, "sample_count"] == 0
    assert result.loc[0, "occupied_bin_fraction"] == 0
    assert result.loc[0, "largest_observation_gap"] == 30


def test_epoch_coverage_uses_custom_bin_width():
    frame = pd.DataFrame({"time": [0, 5, 11], "heart_rate": [1, 2, 3]})
    result = epoch_coverage(frame, pd.DataFrame({"time": [0]}), bin_seconds=5)
    assert result.loc[0, "occupied_bin_fraction"] == pytest.approx(3 / 6)
    assert result.loc[0, "largest_observation_gap"] == 19


@pytest.mark.parametrize("times", [[30, 0], [0, 0], [0, 31], [float("nan")], [float("inf")]])
def test_epoch_coverage_rejects_invalid_label_timeline(times):
    with pytest.raises(ValueError, match="Label"):
        epoch_coverage(pd.DataFrame({"time": []}), pd.DataFrame({"time": times}))


@pytest.mark.parametrize("bin_seconds", [0, -1, 4, 30.0])
def test_epoch_coverage_requires_positive_integer_divisor_bins(bin_seconds):
    with pytest.raises(ValueError, match="bin_seconds"):
        epoch_coverage(pd.DataFrame({"time": []}), pd.DataFrame({"time": [0]}), bin_seconds)


def test_step_bins_deduplicates_and_uses_next_observed_time():
    frame = pd.DataFrame({"time": [10, 0, 0, 25], "steps": [3, 2, 2, 0]})
    result = step_bins(frame)
    assert result[["start", "steps"]].to_dict("records") == [
        {"start": 0, "steps": 2},
        {"start": 10, "steps": 3},
        {"start": 25, "steps": 0},
    ]
    assert result.end.iloc[:2].tolist() == [10, 25]
    assert pd.isna(result.end.iloc[-1])


def test_step_bins_empty_input_preserves_columns():
    result = step_bins(pd.DataFrame({"time": [], "steps": []}))
    assert result.columns.tolist() == ["start", "end", "steps"]
    assert result.empty


def test_epoch_grid_can_start_at_nonzero_offset():
    frame = pd.DataFrame({"time": [5, 35, 65], "heart_rate": [60, 70, 80]})
    result = epoch_coverage(frame, pd.DataFrame({"time": [5, 35]}))
    assert result.sample_count.tolist() == [1, 1]


def test_later_observations_do_not_change_earlier_coverage():
    frame = pd.DataFrame({"time": [0, 10], "heart_rate": [60, 70]})
    labels = pd.DataFrame({"time": [0]})
    before = epoch_coverage(frame, labels)
    later = pd.concat([frame, pd.DataFrame({"time": [30, 100], "heart_rate": [80, 90]})])
    pd.testing.assert_frame_equal(before, epoch_coverage(later, labels))
