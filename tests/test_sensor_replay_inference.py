from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from bundle_fixtures import make_checkpoint  # noqa: E402

from sleep_classifier import sensor_only_replay  # noqa: E402
from sleep_classifier.motion_feature_input import MOTION_FEATURE, MotionFeatureInput  # noqa: E402
from sleep_classifier.sensor_replay_inference import FrozenSensorStream  # noqa: E402
from sleep_classifier.sensor_replay_inputs import RAW_CHANNELS, SensorReplayInputs  # noqa: E402

DIGEST = "a" * 64
PARTICIPANTS = ["p1", "p2"]
TOLERANCE = 2e-6
PROBABILITY_COLUMNS = [f"probability_{stage}" for stage in range(4)]


@pytest.fixture
def checkpoint() -> dict[str, Any]:
    """Return a random-init checkpoint with saved synthetic preprocessing."""
    return make_checkpoint()


@pytest.fixture
def audit_rows() -> pd.DataFrame:
    """Build shuffled synthetic audit rows with differing sensor and reference metadata."""
    rows = pd.DataFrame({
        feature: np.arange(8, dtype=float) + position
        for position, feature in enumerate(RAW_CHANNELS)
    }).assign(
        participant_id=["p1"] * 5 + ["p2"] * 2 + ["other"],
        time=[0.0, 30.0, 60.0, 90.0, 120.0, 0.0, 30.0, 0.0],
        segment_id=[0, 0, 0, 0, 1, 0, np.nan, 0],
        eligible=[True, True, False, True, True, True, np.nan, True],
        target=[-1, 0, 1, 2, 3, 0, 1, 2],
        eligible_reference=[False, True, True, True, False, True, True, True],
        segment_reference=[9, 0, 0, 1, 1, 0, 0, 0],
        motion_reference=100.0,
    )
    rows.loc[1, MOTION_FEATURE] = np.nan
    return rows.iloc[::-1]


def test_candidate_uses_sensor_eligibility_and_saved_motion_mean(
    audit_rows: pd.DataFrame, checkpoint: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project eligible sensor rows and fill only their missing motion values.

    Args:
        audit_rows (pd.DataFrame): Shuffled sensor and reference metadata.
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
        monkeypatch (pytest.MonkeyPatch): Block fitting and labeled-source loading.
    """
    forbidden = Mock(side_effect=AssertionError("Candidate loaded or fitted reference data"))
    monkeypatch.setattr(SensorReplayInputs, "reference", forbidden)
    monkeypatch.setattr(MotionFeatureInput, "prepare", forbidden)
    monkeypatch.setattr(pd.DataFrame, "mean", forbidden)
    monkeypatch.setattr(pd.Series, "mean", forbidden)
    imputation_mean = checkpoint["motion_features"]["imputation_mean"]

    candidate = SensorReplayInputs.candidate(audit_rows, PARTICIPANTS, imputation_mean)

    assert candidate.columns.tolist() == ["participant_id", "time", "segment_id", *RAW_CHANNELS]
    assert list(candidate[["participant_id", "time"]].itertuples(index=False, name=None)) == [
        ("p1", 0.0), ("p1", 30.0), ("p1", 90.0), ("p1", 120.0), ("p2", 0.0),
    ]
    assert candidate.segment_id.tolist() == [0, 0, 0, 1, 0]
    assert candidate[MOTION_FEATURE].tolist() == [6.0, imputation_mean, 9.0, 10.0, 11.0]
    assert pd.isna(audit_rows.loc[1, MOTION_FEATURE])
    forbidden.assert_not_called()


@pytest.mark.parametrize("mean", [None, True, "0.125", np.nan, np.inf, -np.inf])
def test_candidate_rejects_invalid_saved_mean(audit_rows: pd.DataFrame, mean: object) -> None:
    """Require a finite numeric motion replacement from the checkpoint.

    Args:
        audit_rows (pd.DataFrame): Synthetic preparation audit rows.
        mean (object): Invalid saved motion replacement.
    """
    with pytest.raises(ValueError, match="finite saved motion imputation mean"):
        SensorReplayInputs.candidate(audit_rows, PARTICIPANTS, mean)


@pytest.mark.parametrize(
    "column", ["participant_id", "eligible", "segment_id", "time", *RAW_CHANNELS],
)
def test_candidate_requires_sensor_columns(audit_rows: pd.DataFrame, column: str) -> None:
    """Require the complete sensor schema even when reference fields exist.

    Args:
        audit_rows (pd.DataFrame): Synthetic preparation audit rows.
        column (str): Required sensor field to omit.
    """
    with pytest.raises(ValueError, match="required channels"):
        SensorReplayInputs.candidate(audit_rows.drop(columns=column), PARTICIPANTS, 0.125)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("eligible", "true", "eligibility.*booleans"),
        ("time", 30.0, "unique"),
        ("time", np.inf, "finite"),
        (RAW_CHANNELS[0], np.nan, "finite"),
        (MOTION_FEATURE, np.inf, "finite"),
        ("segment_id", np.nan, "finite"),
        ("segment_id", -1, "nonnegative integers"),
        ("segment_id", 0.5, "nonnegative integers"),
    ],
)
def test_candidate_rejects_invalid_sensor_values(
    audit_rows: pd.DataFrame, column: str, value: float | str, message: str,
) -> None:
    """Reject malformed eligible sensor rows before inference.

    Args:
        audit_rows (pd.DataFrame): Synthetic preparation audit rows.
        column (str): Sensor field to replace in the initial epoch.
        value (float | str): Invalid input value.
        message (str): Expected validation error.
    """
    audit_rows.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        SensorReplayInputs.candidate(audit_rows, PARTICIPANTS, 0.125)


def test_candidate_requires_eligible_epochs_for_each_participant(audit_rows: pd.DataFrame) -> None:
    """Reject a held-out participant represented only by reference-eligible rows.

    Args:
        audit_rows (pd.DataFrame): Synthetic preparation audit rows.
    """
    audit_rows.loc[audit_rows.participant_id.eq("p2"), "eligible"] = False
    with pytest.raises(ValueError, match="Every held-out participant"):
        SensorReplayInputs.candidate(audit_rows, PARTICIPANTS, 0.125)


def test_cpu_predictions_ignore_target_and_reference_columns(
    audit_rows: pd.DataFrame, checkpoint: dict[str, Any],
) -> None:
    """Preserve predictions and resets when label-side metadata is replaced.

    Args:
        audit_rows (pd.DataFrame): Shuffled sensor and reference metadata.
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
    """
    stream = FrozenSensorStream(checkpoint, DIGEST, "cpu")
    mean = checkpoint["motion_features"]["imputation_mean"]
    candidate = SensorReplayInputs.candidate(audit_rows, PARTICIPANTS, mean)
    expected = stream.predict(candidate, "clean")
    metadata = ["target", "eligible_reference", "segment_reference", "motion_reference"]
    permuted = audit_rows.copy()
    permuted[metadata] = audit_rows[metadata].iloc[::-1].set_axis(audit_rows.index)
    changed = audit_rows.assign(
        target="unavailable labels",
        eligible_reference="invalid reference eligibility",
        segment_reference=-100,
        motion_reference=np.inf,
    )
    for label, rows in [("permuted", permuted), ("poisoned", changed)]:
        projected = SensorReplayInputs.candidate(rows, PARTICIPANTS, mean)
        pd.testing.assert_frame_equal(projected, candidate)
        injected = projected.join(rows.loc[projected.index, metadata])
        pd.testing.assert_frame_equal(stream.predict(injected, label), expected)
    assert expected.history_epochs.tolist() == [1, 2, 1, 1, 1]
    assert expected.reset_reason.fillna("").tolist() == [
        "", "", "time_gap", "segment_change", "participant_change",
    ]
    assert stream.predictor.device.type == "cpu"
    assert stream.integrity() == {
        "weights_unchanged": True,
        "evaluation_mode": True,
        "gradients_created": False,
        "state_sha256": stream.before["state_sha256"],
        "training_performed": False,
    }


@pytest.mark.parametrize("change", ["weights", "training_mode", "gradients"])
def test_integrity_rejects_model_changes(checkpoint: dict[str, Any], change: str) -> None:
    """Detect changes to the frozen model between construction and integrity checks.

    Args:
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
        change (str): Model invariant to violate.
    """
    stream = FrozenSensorStream(checkpoint, DIGEST, "cpu")
    model = stream.predictor.model
    parameter = next(model.parameters())
    if change == "weights":
        with torch.no_grad():
            parameter.add_(1.0)
    elif change == "training_mode":
        model.train()
    else:
        parameter.grad = torch.zeros_like(parameter)

    with pytest.raises(ValueError, match="Frozen replay changed model state or created gradients"):
        stream.integrity()


def test_cpu_predictions_are_prefix_causal(
    audit_rows: pd.DataFrame, checkpoint: dict[str, Any],
) -> None:
    """Keep every earlier prediction identical when future sensor epochs arrive.

    Args:
        audit_rows (pd.DataFrame): Shuffled sensor and reference metadata.
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
    """
    stream = FrozenSensorStream(checkpoint, DIGEST, "cpu")
    one_session = audit_rows.loc[audit_rows.participant_id.eq("p1")].copy()
    one_session.loc[one_session.time.eq(60), "eligible"] = True
    mean = checkpoint["motion_features"]["imputation_mean"]
    prefix = SensorReplayInputs.candidate(one_session.loc[one_session.time.lt(60)], ["p1"], mean)
    with_future = SensorReplayInputs.candidate(one_session, ["p1"], mean)

    expected = stream.predict(prefix, "prefix")
    actual = stream.predict(with_future, "with future")

    assert len(actual) > len(expected)
    assert actual.history_epochs.iloc[len(expected)] == 3
    prefix_predictions = actual.iloc[:len(expected)]
    assert prefix_predictions.reset_reason.isna().all() and expected.reset_reason.isna().all()
    pd.testing.assert_frame_equal(
        prefix_predictions.drop(columns="reset_reason"), expected.drop(columns="reset_reason"),
    )


@pytest.fixture
def control_predictions(
    audit_rows: pd.DataFrame, checkpoint: dict[str, Any],
) -> pd.DataFrame:
    """Generate fresh probabilities for valid control evidence.

    Args:
        audit_rows (pd.DataFrame): Synthetic preparation audit rows.
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
    """
    mean = checkpoint["motion_features"]["imputation_mean"]
    candidate = SensorReplayInputs.candidate(audit_rows, PARTICIPANTS, mean)
    return FrozenSensorStream(checkpoint, DIGEST, "cpu").predict(candidate, "control")


def test_control_parity_accepts_saved_predictions(control_predictions: pd.DataFrame) -> None:
    """Accept identical legacy predictions and empty CSV reset fields.

    Args:
        control_predictions (pd.DataFrame): Pristine synthetic control evidence.
    """
    saved = control_predictions.assign(
        target=0, reset_reason=control_predictions.reset_reason.fillna(""),
    )
    parity = FrozenSensorStream.control_parity(control_predictions, saved, TOLERANCE)

    assert parity == {
        "status": "passed",
        "epochs": len(saved),
        "maximum_probability_difference": 0.0,
        "class_disagreements": 0,
        "history_lengths_match": True,
        "reset_reasons_match": True,
    }


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("participant_id", "other"),
        ("time", 120.0),
        ("prediction", -1),
        ("history_epochs", 99),
        ("reset_reason", "time_gap"),
        ("probability_0", np.nan),
    ],
)
def test_control_parity_rejects_changed_saved_evidence(
    control_predictions: pd.DataFrame, column: str, value: int | float | str,
) -> None:
    """Reject corruption in control identities, classes, history, and probabilities.

    Args:
        control_predictions (pd.DataFrame): Pristine synthetic control evidence.
        column (str): Saved evidence field to corrupt.
        value (int | float | str): Replacement value.
    """
    saved = control_predictions.assign(target=0)
    saved.loc[0, column] = value
    with pytest.raises(ValueError):
        FrozenSensorStream.control_parity(control_predictions, saved, TOLERANCE)


def test_control_parity_rejects_normalized_probability_drift(
    control_predictions: pd.DataFrame,
) -> None:
    """Require probability parity when the class and probability sum agree.

    Args:
        control_predictions (pd.DataFrame): Pristine synthetic control evidence.
    """
    saved = control_predictions.assign(target=0)
    selected = int(saved.loc[0, "prediction"])
    saved.loc[0, PROBABILITY_COLUMNS] = [
        0.97 if stage == selected else 0.01 for stage in range(4)
    ]
    with pytest.raises(ValueError):
        FrozenSensorStream.control_parity(control_predictions, saved, TOLERANCE)


def test_control_parity_rejects_nonnormalized_probabilities(
    control_predictions: pd.DataFrame,
) -> None:
    """Validate normalization independently of per-stage drift tolerance.

    Args:
        control_predictions (pd.DataFrame): Pristine synthetic control evidence.
    """
    saved = control_predictions.assign(target=0)
    saved.loc[0, PROBABILITY_COLUMNS] += 0.75 * TOLERANCE
    with pytest.raises(ValueError):
        FrozenSensorStream.control_parity(control_predictions, saved, TOLERANCE)


def test_reference_uses_prepared_labels_and_saved_motion_transform(
    checkpoint: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the separate legacy input stream from its original validation source.

    Args:
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
        tmp_path (Path): Temporary folder for frozen source files.
        monkeypatch (pytest.MonkeyPatch): Supply a synthetic verified motion snapshot.
    """
    fold = tmp_path / "fold-01"
    validation = fold / "features" / "validation"
    validation.mkdir(parents=True)
    audit_path = fold / "motion.yaml"
    audit_path.write_text("{}\n")
    manifest = fold / "manifest.json"
    manifest.write_text("{}\n")
    (fold / "gru.yaml").write_text(yaml.safe_dump({
        "motion_features": {"name": "fixed_lag_motion", "audit_config": "motion.yaml"},
    }))
    frame = pd.DataFrame({name: [7.0, 8.0, 9.0, 10.0] for name in RAW_CHANNELS[:-1]}).assign(
        time=[0.0, 30.0, 60.0, 90.0],
        segment_id=[4, 4, 4, 5],
        eligible=[True, True, False, True],
        target=[0, -1, 1, 2],
    )
    frame.iloc[::-1].to_csv(validation / "p1.csv", index=False)
    sensor = pd.DataFrame({
        "participant_id": ["p1"] * 4,
        "time": frame.time,
        MOTION_FEATURE: [np.nan, 12.0, 13.0, 14.0],
    })
    snapshot = Mock(return_value=(sensor, {
        "development_participants": ["p1"], "reserved_test_participants": ["reserved"],
    }, manifest))
    monkeypatch.setattr(MotionFeatureInput, "_load_sensor", snapshot)
    checkpoint["motion_features"].update({
        "audit_config_sha256": MotionFeatureInput._digest(audit_path),
        "snapshot_manifest_sha256": MotionFeatureInput._digest(manifest),
    })

    reference = SensorReplayInputs.reference(fold, ["p1"], checkpoint)
    predictions = FrozenSensorStream(checkpoint, DIGEST, "cpu").predict(reference, "legacy")

    assert reference.target.tolist() == [0, 2]
    assert reference.time.tolist() == [0.0, 90.0]
    assert reference.segment_id.tolist() == [4, 5]
    assert reference[MOTION_FEATURE].tolist() == [0.125, 14.0]
    assert predictions.history_epochs.tolist() == [1, 1]
    assert predictions.reset_reason.fillna("").tolist() == ["", "segment_change"]
    snapshot.assert_called_once()


def test_study_predicts_candidate_before_opening_labeled_control(
    audit_rows: pd.DataFrame, checkpoint: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep labeled validation inputs and saved predictions out of candidate inference.

    Args:
        audit_rows (pd.DataFrame): Synthetic sensor audit with distinct reference metadata.
        checkpoint (dict[str, Any]): Synthetic weights and fitted metadata.
        tmp_path (Path): Temporary location for saved chronological predictions.
        monkeypatch (pytest.MonkeyPatch): Bind verified synthetic sources and trace labeled reads.
    """
    fold = {"name": "fold-01", "validation": PARTICIPANTS}
    reference = audit_rows.loc[[1, 2, 3], ["participant_id", "time", "segment_id", *RAW_CHANNELS]]
    reference = reference.assign(segment_id=[10, 10, 11], target=[1, 2, 3])
    reference[MOTION_FEATURE] = reference[MOTION_FEATURE].fillna(
        checkpoint["motion_features"]["imputation_mean"],
    )
    reference_predictions = FrozenSensorStream(checkpoint, DIGEST, "cpu").predict(
        reference, "saved chronological",
    )
    saved_path = tmp_path / "fold-01.csv"
    reference_predictions.assign(target=reference.target.to_numpy()).iloc[::-1].to_csv(
        saved_path, index=False,
    )
    source = Mock(fold_dir=tmp_path / "frozen-research")
    source.snapshot.return_value = {
        "fold": {"checkpoint_sha256": DIGEST, "absolute_tolerance": TOLERANCE},
    }
    source.checkpoint.return_value = (checkpoint, {})
    source_factory = Mock(return_value=source)
    monkeypatch.setattr(sensor_only_replay, "EvaluationExportSource", source_factory)
    study = object.__new__(sensor_only_replay.SensorOnlyReplayStudy)
    study.chronological = SimpleNamespace(
        config_path=tmp_path / "chronological.yaml", output_dir=tmp_path,
    )
    study.device = "cpu"
    events = Mock()
    reference_loader = Mock(return_value=reference)
    saved_loader = Mock(wraps=pd.read_csv)
    monkeypatch.setattr(SensorReplayInputs, "reference", reference_loader)
    monkeypatch.setattr(pd, "read_csv", saved_loader)
    events.attach_mock(reference_loader, "read_reference")
    events.attach_mock(saved_loader, "read_saved")

    with patch.object(
        FrozenSensorStream, "predict", autospec=True, side_effect=FrozenSensorStream.predict,
    ) as predict:
        events.attach_mock(predict, "predict")
        candidate, saved, report = study._run_fold(fold, audit_rows)

    assert [event[0] for event in events.mock_calls] == [
        "predict", "read_reference", "read_saved", "predict",
    ]
    assert predict.call_args_list[0].args[1].columns.tolist() == [
        "participant_id", "time", "segment_id", *RAW_CHANNELS,
    ]
    assert predict.call_args_list[1].args[1] is reference
    reference_loader.assert_called_once_with(source.fold_dir, PARTICIPANTS, checkpoint)
    saved_loader.assert_called_once_with(saved_path, dtype={"participant_id": str})
    source_factory.assert_called_once_with(study.chronological.config_path, fold["name"])
    assert "target" not in candidate
    assert saved.target.tolist() == [1, 2, 3]
    assert report["control"]["status"] == "passed"
    assert report["integrity"]["saved_transforms_unchanged"]
    assert report["comparison"]["scopes"] == {
        "candidate_epochs": 5,
        "reference_epochs": 3,
        "shared_scored_epochs": 2,
        "candidate_only_epochs": 3,
        "reference_only_epochs": 1,
    }
