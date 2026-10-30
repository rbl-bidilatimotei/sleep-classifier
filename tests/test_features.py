import numpy as np
import pandas as pd
import pytest

from sleep_classifier.features import build_epoch_features

RULE = {
    "max_heart_gap_seconds": 15,
    "min_heart_readings": 3,
    "min_motion_bin_fraction": 0.8,
    "max_motion_gap_seconds": 5,
}


def sensors(end=60):
    times = np.arange(end, dtype=float)
    heart_times = times[::10]
    heart = pd.DataFrame({"time": heart_times, "heart_rate": 60 + 10 * np.arange(len(heart_times))})
    motion = pd.DataFrame({"time": times, "x": times, "y": 0.0, "z": 0.0})
    return heart, motion


def test_epoch_uses_half_open_boundaries_and_six_summary_features():
    heart = pd.DataFrame({"time": [0, 10, 20, 30], "heart_rate": [60, 70, 80, 200]})
    motion = pd.DataFrame({"time": [0, 1, 29, 30], "x": [3, 0, 0, 0], "y": 4, "z": 0})
    labels = pd.DataFrame({"time": [0], "stage": [0]})

    row = build_epoch_features(heart, motion, labels, RULE).iloc[0]

    assert row.heart_rate_mean == 70
    assert row.heart_rate_std == pytest.approx(np.std([60, 70, 80]))
    assert row.heart_rate_max == 80
    assert row.motion_magnitude_mean == pytest.approx(13 / 3)
    assert row.motion_sample_count == 3
    assert row.target == 0


def test_duplicate_observations_are_deduplicated_and_conflicts_fail():
    heart = pd.DataFrame({"time": [0, 0, 10, 20], "heart_rate": [60, 60, 70, 80]})
    motion = pd.DataFrame({"time": [0, 10, 20], "x": [1, 1, 1], "y": 0, "z": 0})
    labels = pd.DataFrame({"time": [0], "stage": [2]})
    assert build_epoch_features(heart, motion, labels, RULE).iloc[0].heart_sample_count == 3

    heart.loc[1, "heart_rate"] = 61
    with pytest.raises(ValueError, match="Conflicting sensor values"):
        build_epoch_features(heart, motion, labels, RULE)


def test_unscored_and_n4_targets_and_absent_frames():
    heart = pd.DataFrame({"time": [0, 10, 20], "heart_rate": [60, 70, 80]})
    motion = pd.DataFrame({"time": [0, 1, 2], "x": [1, 1, 1], "y": 0, "z": 0})
    labels = pd.DataFrame({"time": [0, 30], "stage": [-1, 4]})

    result = build_epoch_features(heart, motion, labels, RULE)

    assert result.target.tolist() == [-1, 2]
    assert result.loc[0, "heart_rate_mean"] == 70
    assert np.isnan(result.loc[1, "heart_rate_mean"])
    assert not result.eligible.any()
    assert result.segment_id.tolist() == [-1, -1]


def test_segments_break_after_rejection_and_missing_label_time():
    heart, motion = sensors(150)
    # The second epoch lacks sufficient heart-rate observations
    heart = heart.loc[(heart.time < 30) | (heart.time >= 60)].copy()
    labels = pd.DataFrame({"time": [0, 30, 60, 120], "stage": [0, 1, 2, 3]})

    result = build_epoch_features(heart, motion, labels, RULE)

    assert result.eligible.tolist() == [True, False, True, True]
    assert result.segment_id.tolist() == [0, -1, 1, 2]


def test_appending_future_observations_does_not_change_earlier_features():
    heart, motion = sensors(60)
    labels = pd.DataFrame({"time": [0, 30], "stage": [1, 5]})
    original = build_epoch_features(heart, motion, labels, RULE)
    future_heart = pd.concat([heart, pd.DataFrame({"time": [60, 70], "heart_rate": [150, 160]})])
    future_motion = pd.concat(
        [motion, pd.DataFrame({"time": [60, 61], "x": [1000, 1000], "y": 0, "z": 0})]
    )

    expanded = build_epoch_features(future_heart, future_motion, labels, RULE)

    pd.testing.assert_frame_equal(original, expanded)


def test_empty_label_timeline_preserves_feature_schema():
    heart, motion = sensors()
    result = build_epoch_features(heart, motion, pd.DataFrame({"time": [], "stage": []}), RULE)
    assert result.empty
    assert "heart_rate_mean" in result
    assert "segment_id" in result


def test_nonfinite_values_are_rejected():
    heart, motion = sensors()
    heart["heart_rate"] = heart.heart_rate.astype(float)
    heart.loc[0, "heart_rate"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        build_epoch_features(heart, motion, pd.DataFrame({"time": [0], "stage": [1]}), RULE)


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_heart_readings", 0),
        ("min_motion_bin_fraction", 1.1),
        ("max_heart_gap_seconds", np.nan),
    ],
)
def test_invalid_rule_cannot_mark_missing_features_eligible(key, value):
    heart, motion = sensors()
    with pytest.raises(ValueError, match="threshold"):
        build_epoch_features(
            heart, motion, pd.DataFrame({"time": [0], "stage": [1]}), {**RULE, key: value}
        )
