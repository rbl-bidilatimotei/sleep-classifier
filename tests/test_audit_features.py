from typing import Any

import numpy as np
import pandas as pd
import pytest

from sleep_classifier.audit_features import CausalAuditFeatures


def _heart_rows() -> pd.DataFrame:
    """Create one contiguous eligible history with an unrelated preserved feature."""
    return pd.DataFrame({
        "participant_id": ["a"] * 6,
        "time": np.arange(6) * 30.0,
        "target": [0, 1, 1, 2, 3, 1],
        "segment_id": [0] * 6,
        "heart_rate_mean": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0],
        "motion_magnitude_mean": [1.0] * 6,
    })


def test_heart_history_uses_only_previous_epochs_and_preserves_rows() -> None:
    """Expose no delta until sufficient strictly previous observations are available."""
    source = _heart_rows()
    result = CausalAuditFeatures(history_epochs=3, min_history_epochs=2).add_heart_history(source)

    assert len(result) == len(source)
    pd.testing.assert_frame_equal(result[source.columns], source)
    assert result.heart_rate_past_count.tolist() == [0, 1, 2, 3, 3, 3]
    assert np.isnan(result.loc[0, "heart_rate_past_mean"])
    assert result.loc[1, "heart_rate_past_mean"] == 60.0
    assert result.loc[2, "heart_rate_past_mean"] == 61.0
    assert result.loc[5, "heart_rate_past_mean"] == 66.0
    assert result.heart_history_available.tolist() == [False, False, True, True, True, True]
    assert result.loc[:1, "heart_rate_delta_past"].isna().all()
    assert result.loc[2, "heart_rate_delta_past"] == 3.0
    assert result.loc[5, "heart_rate_delta_past"] == 4.0
    assert "heart_rate_past_mean" not in source.columns


def test_heart_history_is_invariant_to_future_append_or_future_value_change() -> None:
    """Keep all past results unchanged when future sensor values change."""
    builder = CausalAuditFeatures(history_epochs=3, min_history_epochs=2)
    source = _heart_rows()
    prefix = builder.add_heart_history(source.iloc[:4])
    full = builder.add_heart_history(source)
    changed = source.copy()
    changed.loc[4:, "heart_rate_mean"] = [300.0, 500.0]
    modified = builder.add_heart_history(changed)

    pd.testing.assert_frame_equal(prefix, full.iloc[:4].reset_index(drop=True))
    pd.testing.assert_frame_equal(prefix, modified.iloc[:4].reset_index(drop=True))


def test_heart_history_resets_at_participant_and_segment_boundaries() -> None:
    """Prevent histories from crossing a participant boundary or an eligible-data gap."""
    first = _heart_rows().iloc[:4].copy()
    second = first.iloc[:2].copy().assign(time=[300.0, 330.0], segment_id=1)
    other = first.iloc[:2].copy().assign(participant_id="b", heart_rate_mean=[100.0, 110.0])
    source = pd.concat([second, other, first], ignore_index=True)
    result = CausalAuditFeatures(min_history_epochs=1).add_heart_history(source)

    a = result.loc[result.participant_id == "a"].reset_index(drop=True)
    b = result.loc[result.participant_id == "b"].reset_index(drop=True)
    assert a.loc[4, "heart_rate_past_count"] == 0
    assert np.isnan(a.loc[4, "heart_rate_delta_past"])
    assert a.loc[5, "heart_rate_past_mean"] == 60.0
    assert b.loc[0, "heart_rate_past_count"] == 0
    assert b.loc[1, "heart_rate_past_mean"] == 100.0


@pytest.mark.parametrize("mutation", ["duplicate", "gap", "interleave", "nonfinite", "segment"])
def test_heart_history_rejects_invalid_identity_or_causality(mutation: str) -> None:
    """Reject epoch metadata that cannot support a causal prior history.

    Args:
        mutation (str): Invalid epoch condition to construct.
    """
    rows = _heart_rows()
    if mutation == "duplicate":
        rows.loc[1, "time"] = rows.loc[0, "time"]
    elif mutation == "gap":
        rows.loc[2:, "time"] += 30
    elif mutation == "interleave":
        rows.loc[2, "segment_id"] = 1
    elif mutation == "nonfinite":
        rows.loc[2, "heart_rate_mean"] = np.nan
    else:
        rows.loc[2, "segment_id"] = -1

    with pytest.raises(ValueError):
        CausalAuditFeatures().add_heart_history(rows)


def test_motion_changes_detect_direction_changes_hidden_by_constant_magnitude() -> None:
    """Distinguish moving directions from static axes with identical vector magnitudes."""
    epochs = pd.DataFrame({"time": [0.0]})
    static = pd.DataFrame({"time": [0.0, 0.1, 0.2], "x": [1, 1, 1], "y": [0, 0, 0],
                           "z": [0, 0, 0]})
    rotating = static.assign(x=[1, 0, 1], y=[0, 1, 0])
    assert np.linalg.norm(static[["x", "y", "z"]], axis=1).tolist() == [1, 1, 1]
    assert np.linalg.norm(rotating[["x", "y", "z"]], axis=1).tolist() == [1, 1, 1]

    builder = CausalAuditFeatures()
    static_summary = builder.motion_summaries(static, epochs).iloc[0]
    moving_summary = builder.motion_summaries(rotating, epochs).iloc[0]
    assert static_summary.motion_vector_change_mean == 0.0
    assert moving_summary.motion_vector_change_mean == pytest.approx(np.sqrt(2))
    assert moving_summary.motion_pair_count == 2
    assert moving_summary.motion_valid_pair_fraction == 1.0


def test_motion_gap_pairs_are_excluded_but_their_gaps_remain_visible() -> None:
    """Report large observed gaps without allowing their vector changes into the feature."""
    motion = pd.DataFrame({"time": [0.0, 0.1, 1.0, 1.1], "x": [0, 1, 100, 101],
                           "y": [0, 0, 0, 0], "z": [0, 0, 0, 0]})
    result = CausalAuditFeatures().motion_summaries(motion, pd.DataFrame({"time": [0]})).iloc[0]

    assert result.motion_vector_change_mean == pytest.approx(1.0)
    assert result.motion_pair_count == 2
    assert result.motion_valid_pair_fraction == pytest.approx(2 / 3)
    assert result.motion_pair_gap_median_seconds == pytest.approx(0.1)
    assert result.motion_pair_gap_max_seconds == pytest.approx(0.9)


def test_motion_pairs_never_cross_epoch_boundaries_and_future_does_not_change_past() -> None:
    """Exclude the pair straddling a completed interval and preserve its previous summary."""
    motion = pd.DataFrame({"time": [29.9, 30.0, 30.1], "x": [100, 0, 1],
                           "y": [0, 0, 0], "z": [0, 0, 0]})
    builder = CausalAuditFeatures()
    epochs = pd.DataFrame({"time": [0.0, 30.0]})
    result = builder.motion_summaries(motion, epochs)
    prefix = builder.motion_summaries(motion.iloc[:1], epochs.iloc[:1])

    pd.testing.assert_frame_equal(prefix, result.iloc[:1].reset_index(drop=True))
    assert result.loc[0, "motion_pair_count"] == 0
    assert np.isnan(result.loc[0, "motion_vector_change_mean"])
    assert result.loc[1, "motion_pair_count"] == 1
    assert result.loc[1, "motion_vector_change_mean"] == 1.0


@pytest.mark.parametrize("times", [[], [0.1], [0.0, 2.0]])
def test_motion_unavailable_changes_remain_missing(times: list[float]) -> None:
    """Keep missing features distinct from observed zero movement.

    Args:
        times (list[float]): Zero, one, or excessively separated raw observations.
    """
    motion = pd.DataFrame({"time": times, "x": [0.0] * len(times),
                           "y": [0.0] * len(times), "z": [0.0] * len(times)})
    result = CausalAuditFeatures().motion_summaries(motion, pd.DataFrame({"time": [0]})).iloc[0]

    assert result.motion_pair_count == 0
    assert np.isnan(result.motion_vector_change_mean)
    if len(times) < 2:
        assert np.isnan(result.motion_valid_pair_fraction)
        assert np.isnan(result.motion_pair_gap_median_seconds)
    else:
        assert result.motion_valid_pair_fraction == 0.0
        assert result.motion_pair_gap_max_seconds == 2.0


def test_motion_exact_duplicates_are_removed_and_conflicts_rejected() -> None:
    """Reuse the sensor duplicate policy before constructing adjacent pairs."""
    motion = pd.DataFrame({"time": [0.0, 0.1, 0.1], "x": [0, 1, 1], "y": [0, 0, 0],
                           "z": [0, 0, 0]})
    builder = CausalAuditFeatures()
    result = builder.motion_summaries(motion, pd.DataFrame({"time": [0]})).iloc[0]
    assert result.motion_pair_count == 1
    assert result.motion_vector_change_mean == 1.0
    motion.loc[2, "x"] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        builder.motion_summaries(motion, pd.DataFrame({"time": [0]}))


@pytest.mark.parametrize("times", [[0, 15], [30, 0], [0, 0], [0, np.inf]])
def test_motion_rejects_invalid_epoch_grid(times: list[float]) -> None:
    """Reject overlapping, repeated, nonfinite, or unordered target intervals.

    Args:
        times (list[float]): Invalid epoch starts.
    """
    motion = pd.DataFrame(columns=["time", "x", "y", "z"])
    with pytest.raises(ValueError):
        CausalAuditFeatures().motion_summaries(motion, pd.DataFrame({"time": times}))


@pytest.mark.parametrize("settings", [
    {"history_epochs": True}, {"history_epochs": 0}, {"min_history_epochs": 0},
    {"history_epochs": 2, "min_history_epochs": 3}, {"min_history_epochs": 1.5},
    {"max_motion_pair_gap_seconds": True}, {"max_motion_pair_gap_seconds": np.nan},
    {"max_motion_pair_gap_seconds": 0}, {"max_motion_pair_gap_seconds": 31},
])
def test_feature_audit_rejects_invalid_settings(settings: dict[str, Any]) -> None:
    """Reject ambiguous or nonphysical audit settings before reading observations.

    Args:
        settings (dict[str, Any]): Invalid constructor override.
    """
    with pytest.raises(ValueError):
        CausalAuditFeatures(**settings)
