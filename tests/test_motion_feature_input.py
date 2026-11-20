import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.motion_feature_input import (
    MOTION_DEFINITION,
    MOTION_FEATURE,
    MotionFeatureInput,
)
from sleep_classifier.sequences import CausalSequenceDataset
from sleep_classifier.window_heart_rate import WindowLocalHeartRate


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MotionFeatureInput:
    """Provide a synthetic verified cache with adversarial target and prediction fields.

    Args:
        tmp_path (Path): Isolated source configuration and manifest directory.
        monkeypatch (pytest.MonkeyPatch): Scoped audit-reader replacement.
    """
    config = tmp_path / "audit.yaml"
    config.write_text("output_dir: snapshot\n")
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    rows = pd.DataFrame({
        "participant_id": ["train"] * 3 + ["held"] * 2,
        "time": [0., 30., 60., 0., 30.],
        MOTION_FEATURE: [0., 2., np.nan, 1000., np.nan],
        "target": [3] * 5, "prediction_reference": [2] * 5,
    })
    summary = {"motion_grid": {"step_seconds": 1., "max_sample_age_seconds": .2,
                               "points_per_epoch": 29, "pairs_per_epoch": 28},
               "development_participants": ["train", "held"],
               "reserved_test_participants": ["reserved"]}
    audit = SimpleNamespace(output_dir=directory, load_report=Mock(return_value=(rows, summary)))
    monkeypatch.setattr("sleep_classifier.causal_feature_audit.CausalFeatureAudit",
                        Mock(return_value=audit))
    return MotionFeatureInput({"name": "fixed_lag_motion", "audit_config": "audit.yaml"},
                              tmp_path / "model.yaml")


def _targets(person: str, count: int) -> pd.DataFrame:
    """Create original-feature targets without supplemental audit information.

    Args:
        person (str): Participant identity for this partition.
        count (int): Number of contiguous scored epochs.
    """
    frame = pd.DataFrame(np.zeros((count, 6)), columns=FEATURE_COLUMNS)
    return frame.assign(participant_id=person, time=np.arange(count) * 30., target=0,
                        eligible=True, segment_id=0)


def test_motion_join_uses_training_only_imputation_and_sensor_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain zeros and targets without importing labels or validation statistics.

    Args:
        tmp_path (Path): Isolated source files.
        monkeypatch (pytest.MonkeyPatch): Synthetic cache-reader replacement.
    """
    source = _source(tmp_path, monkeypatch)
    train, validation, metadata = source.prepare(_targets("train", 3), _targets("held", 2))

    assert train[MOTION_FEATURE].tolist() == [0., 2., 1.]
    assert validation[MOTION_FEATURE].tolist() == [1000., 1.]
    assert train.target.eq(0).all() and validation.target.eq(0).all()
    assert "prediction_reference" not in train
    assert metadata["imputation_mean"] == 1.
    assert metadata["training_missing_targets"] == 1
    assert metadata["validation_missing_targets"] == 1
    assert len(metadata["snapshot_manifest_sha256"]) == 64
    assert metadata["definition"] == MOTION_DEFINITION
    assert metadata["scaling_fit"] == "training_targets_only"
    assert metadata["missing_policy"] == "training_mean"


@pytest.mark.parametrize("person,count,match", [
    ("reserved", 1, "development participants"),
    ("unknown", 1, "development participants"),
    ("train", 4, "audit identity"),
])
def test_motion_join_rejects_out_of_scope_or_absent_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, person: str, count: int, match: str,
) -> None:
    """Reject test observations and missing identity matches before model construction.

    Args:
        tmp_path (Path): Isolated source files.
        monkeypatch (pytest.MonkeyPatch): Synthetic cache-reader replacement.
        person (str): Invalid or incomplete participant selection.
        count (int): Number of requested epochs.
        match (str): Expected error detail.
    """
    source = _source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=match):
        source.prepare(_targets(person, count), _targets("held", 2))


@pytest.mark.parametrize("values", [{}, {"name": "other", "audit_config": "audit.yaml"},
                                   {"name": "fixed_lag_motion", "audit_config": ""}])
def test_motion_source_rejects_invalid_settings(values: dict[str, Any]) -> None:
    """Validate source declarations without accessing any files.

    Args:
        values (dict[str, Any]): Invalid source mapping.
    """
    with pytest.raises(ValueError, match="motion_features"):
        MotionFeatureInput(values, Path("missing.yaml"))


def test_seven_channel_windows_preserve_legacy_heart_rate_transform() -> None:
    """Pass through motion while keeping heart-rate fitting and masking unchanged."""
    frame = _targets("train", 6)
    frame["heart_rate_mean"] = np.arange(6)
    frame[MOTION_FEATURE] = np.arange(6) * .1
    columns = (*FEATURE_COLUMNS, MOTION_FEATURE)
    scaler = StandardScaler().fit(frame.loc[:, columns])
    data = CausalSequenceDataset(frame, scaler, context_epochs=6, feature_columns=columns)
    settings = {"name": "heart_rate_deviation", "history_epochs": 3, "min_history_epochs": 2}
    state = WindowLocalHeartRate.fit(data, float(scaler.scale_[0]), settings, input_size=7)
    legacy_samples = [{"features": sample["features"][:, :6]} for sample in data]
    legacy_state = WindowLocalHeartRate.fit(legacy_samples, float(scaler.scale_[0]), settings)
    legacy_scaler = StandardScaler().fit(frame.loc[:, FEATURE_COLUMNS])
    np.testing.assert_array_equal(scaler.mean_[:6], legacy_scaler.mean_)
    np.testing.assert_array_equal(scaler.scale_[:6], legacy_scaler.scale_)
    assert state == legacy_state
    features = data[-1]["features"].unsqueeze(0)
    lengths = torch.tensor([6])
    actual = WindowLocalHeartRate(settings, state, input_size=7)(features, lengths)
    expected = WindowLocalHeartRate(settings, state)(features[:, :, :6], lengths)
    torch.testing.assert_close(actual[:, :, :6], expected[:, :, :6], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :, 7:], expected[:, :, 6:], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, :, 6], features[:, :, 6], rtol=0, atol=0)


def test_motion_checkpoint_restores_without_audit_files(tmp_path: Path) -> None:
    """Restore an expanded patch model using saved semantics and fitted state only.

    Args:
        tmp_path (Path): Nonexistent source path proving inference needs no cache.
    """
    settings = {
        "model_type": "patch_transformer", "context_epochs": 6, "patch_size": 3,
        "transformer": {"d_model": 16, "nhead": 2, "num_layers": 1,
                        "feedforward_dim": 32, "dropout": .2, "max_context": 6},
        "window_features": {"name": "heart_rate_deviation", "history_epochs": 3,
                            "min_history_epochs": 2},
        "motion_features": {"name": "fixed_lag_motion", "audit_config": str(tmp_path / "absent")},
    }
    state = {"schema_version": 1, "heart_rate_scale": 1., "delta_mean": 0., "delta_scale": 1.,
             "available_training_targets": 4, "scaling_fit": "training_targets_only"}
    factory = ModelFactory(settings)
    model = factory.build(7, 4, window_feature_state=state).eval()
    control = ModelFactory({key: value for key, value in settings.items()
                            if key != "motion_features"}).build(6, 4, window_feature_state=state)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters - sum(p.numel() for p in control.parameters()) == 48
    checkpoint = {"model_type": "patch_transformer",
                  "model_config": factory.checkpoint_config(7, 4),
                  "context_epochs": 6, "state_dict": copy.deepcopy(model.state_dict()),
                  "window_feature_state": state, "scaler_scale": [1.] * 7,
                  "model_features": factory.model_features([*FEATURE_COLUMNS, MOTION_FEATURE])}
    assert checkpoint["model_config"]["motion_features"] == MOTION_DEFINITION
    restored = ModelFactory.from_checkpoint(checkpoint)
    features = torch.randn(2, 6, 7)
    lengths = torch.tensor([6, 4])
    torch.testing.assert_close(
        restored(features, lengths), model(features, lengths), rtol=0, atol=0,
    )


def test_motion_replay_reuses_saved_imputation_and_rejects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use saved training statistics during replay and detect changed source metadata.

    Args:
        tmp_path (Path): Isolated audit configuration and manifest.
        monkeypatch (pytest.MonkeyPatch): Synthetic cache-reader replacement.
    """
    source = _source(tmp_path, monkeypatch)
    _, validation, state = source.prepare(_targets("train", 3), _targets("held", 2))
    actual = source.transform(_targets("held", 2), state)
    pd.testing.assert_frame_equal(actual, validation)
    source.audit_config.write_text("changed: true\n")
    with pytest.raises(ValueError, match="saved training metadata"):
        source.transform(_targets("held", 2), state)


def test_checkpoint_replay_loads_seven_channels_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay motion inputs with stored imputation and scaler values only.

    Args:
        tmp_path (Path): Isolated audit and checkpoint artifacts.
        monkeypatch (pytest.MonkeyPatch): Synthetic source and prepared-reader replacements.
    """
    from sleep_classifier.checkpoint_audit import CheckpointReplayAudit

    source = _source(tmp_path, monkeypatch)
    train, validation, motion_state = source.prepare(_targets("train", 3), _targets("held", 2))
    columns = (*FEATURE_COLUMNS, MOTION_FEATURE)
    scaler = StandardScaler().fit(train.loc[:, columns])
    settings = {"model_type": "patch_transformer", "context_epochs": 3, "patch_size": 3,
                "transformer": {"max_context": 3, "dropout": 0.},
                "motion_features": {"name": "fixed_lag_motion", "audit_config": "audit.yaml"}}
    factory = ModelFactory(settings)
    model = factory.build(7, 4).eval()
    checkpoint = {"state_dict": model.state_dict(), "model_type": "patch_transformer",
                  "model_config": factory.checkpoint_config(7, 4),
                  "features": list(columns), "scaler_mean": scaler.mean_.tolist(),
                  "scaler_scale": scaler.scale_.tolist(), "context_epochs": 3,
                  "motion_features": motion_state}
    (tmp_path / "results").mkdir()
    torch.save(checkpoint, tmp_path / "results/model_checkpoint.pt")
    (tmp_path / "gru.yaml").write_text(
        "motion_features:\n  name: fixed_lag_motion\n  audit_config: audit.yaml\n"
    )
    reader = Mock()
    reader.load.return_value = {"partitions": {"validation": _targets("held", 2)}}
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.PreparedFeatureDataset",
                        Mock(return_value=reader))
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.StandardScaler.fit",
                        Mock(side_effect=AssertionError("Replay must not fit")))
    _, data, actual = CheckpointReplayAudit(tmp_path)._load()

    pd.testing.assert_frame_equal(actual, validation)
    assert data[0]["features"].shape == (1, 7)
    assert data[1]["features"][-1, -1] == (1. - scaler.mean_[-1]) / scaler.scale_[-1]
