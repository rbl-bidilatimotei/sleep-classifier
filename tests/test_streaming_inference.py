import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.motion_feature_input import MOTION_DEFINITION, MOTION_FEATURE  # noqa: E402
from sleep_classifier.sequences import CausalSequenceDataset, SequenceBatchCollator  # noqa: E402
from sleep_classifier.streaming_inference import StreamingEpochPredictor  # noqa: E402
from sleep_classifier.window_heart_rate import WindowLocalHeartRate  # noqa: E402


def _checkpoint(motion: bool = False) -> dict[str, Any]:
    """Build a tiny patch checkpoint with a fitted window-local heart-rate transform.

    Args:
        motion (bool): Include the optional prepared motion input channel.
    """
    settings = {
        "model_type": "patch_transformer", "context_epochs": 5, "patch_size": 2,
        "transformer": {"d_model": 4, "nhead": 2, "feedforward_dim": 8, "dropout": 0.3},
        "window_features": {
            "name": "heart_rate_deviation", "history_epochs": 3, "min_history_epochs": 2,
        },
    }
    columns = list(FEATURE_COLUMNS)
    if motion:
        settings["motion_features"] = MOTION_DEFINITION
        columns.append(MOTION_FEATURE)
    features = len(columns)
    scales = [float(index + 2) for index in range(features)]
    state = {
        "schema_version": 1, "heart_rate_scale": scales[columns.index("heart_rate_mean")],
        "delta_mean": 0.5, "delta_scale": 2.0, "available_training_targets": 12,
        "scaling_fit": "training_targets_only",
    }
    factory = ModelFactory(settings)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(63)
        model = factory.build(features, 4, state)
    return {
        "model_type": factory.model_type, "context_epochs": settings["context_epochs"],
        "model_config": factory.checkpoint_config(features, 4), "features": columns,
        "model_features": factory.model_features(columns), "window_feature_state": state,
        "scaler_mean": [float(index) * 0.25 for index in range(features)],
        "scaler_scale": scales, "state_dict": model.state_dict(),
    }


def _features(columns: tuple[str, ...], index: int) -> dict[str, float]:
    """Generate one synthetic raw epoch with variable heart rate.

    Args:
        columns (tuple[str, ...]): Ordered raw feature schema.
        index (int): Synthetic epoch position.
    """
    row = {name: float((index + 1) * (position + 2)) for position, name in enumerate(columns)}
    row["heart_rate_mean"] = 55.0 + 2 * index + index % 3
    return row


def _unexpected_fit(*args: Any, **kwargs: Any) -> None:
    """Fail if inference tries to fit either transform.

    Args:
        *args (Any): Unexpected positional fit inputs.
        **kwargs (Any): Unexpected named fit inputs.
    """
    pytest.fail("Inference attempted to fit a transform")


def _batch_reference(
    checkpoint: dict[str, Any], rows: list[dict[str, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate stored weights on independent dataset windows and check heart-rate augmentation.

    Args:
        checkpoint (dict[str, Any]): Saved model, scaler, and transform metadata.
        rows (list[dict[str, float]]): Synthetic feature rows in feed order.
    """
    columns = tuple(checkpoint["features"])
    frame = pd.DataFrame(rows).assign(
        participant_id="p1", segment_id=0, time=np.arange(len(rows)) * 30, eligible=True, target=0,
    )
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"])
    scaler.feature_names_in_ = np.asarray(columns, dtype=object)
    scaler.n_features_in_ = len(columns)
    data = CausalSequenceDataset(frame, scaler, checkpoint["context_epochs"], columns)
    batch = SequenceBatchCollator()([data[index] for index in range(len(data))])
    model = ModelFactory.from_checkpoint(checkpoint)
    with torch.inference_mode():
        augmented = model.transform(batch["features"], batch["lengths"])
        probabilities = torch.softmax(model.backbone(augmented, batch["lengths"]), dim=1)
    _check_heart_rate_reference(augmented, batch["lengths"], rows, checkpoint)
    return probabilities, batch["lengths"]


def _check_heart_rate_reference(
    augmented: torch.Tensor, lengths: torch.Tensor, rows: list[dict[str, float]],
    checkpoint: dict[str, Any],
) -> None:
    """Verify derived channels against raw heart-rate means within each retained window.

    Args:
        augmented (torch.Tensor): Features after window-local augmentation.
        lengths (torch.Tensor): Number of valid epochs in each padded example.
        rows (list[dict[str, float]]): Original synthetic feature rows.
        checkpoint (dict[str, Any]): Saved normalization and history metadata.
    """
    normalization = checkpoint["window_feature_state"]
    heart = np.asarray([row["heart_rate_mean"] for row in rows])
    history = checkpoint["model_config"]["window_features"]["history_epochs"]
    for target, length in enumerate(lengths.tolist()):
        start = target + 1 - length
        assert augmented[target, 0, -2:].tolist() == [0.0, 0.0]
        if length <= 2:
            assert augmented[target, length - 1, -2:].tolist() == [0.0, 0.0]
            continue
        raw_delta = heart[target] - heart[max(start, target - history):target].mean()
        expected = (raw_delta - normalization["delta_mean"]) / normalization["delta_scale"]
        assert augmented[target, length - 1, -2:].tolist() == pytest.approx([expected, 1.0])


@pytest.mark.parametrize("motion", [False, True])
def test_streaming_matches_independent_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, motion: bool,
) -> None:
    """Match padded batch inference through eviction, with no fit, labels, or state updates.

    Args:
        tmp_path (Path): Temporary directory for the synthetic checkpoint.
        monkeypatch (pytest.MonkeyPatch): Scoped prohibition on scaler fitting.
        motion (bool): Include the optional motion feature in the raw schema.
    """
    checkpoint = _checkpoint(motion)
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(StandardScaler, "fit", _unexpected_fit)
    monkeypatch.setattr(WindowLocalHeartRate, "fit", staticmethod(_unexpected_fit))
    predictor = StreamingEpochPredictor(path)
    assert predictor.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert predictor.feature_columns == tuple(checkpoint["features"])
    assert predictor.context_epochs == checkpoint["context_epochs"]
    assert not predictor.model.training
    rows = [_features(predictor.feature_columns, index) for index in range(8)]
    expected, lengths = _batch_reference(checkpoint, rows)
    for index, features in enumerate(rows):
        result = predictor.predict_epoch("p1", 0, 30 * index, dict(reversed(features.items())))
        assert result == {
            "participant_id": "p1", "time": 30.0 * index, "reset_reason": None,
            "history_epochs": lengths[index].item(), "prediction": expected[index].argmax().item(),
            "probabilities": pytest.approx(expected[index].tolist(), rel=0, abs=2e-6),
        }
    assert all(parameter.grad is None for parameter in predictor.model.parameters())
    for name, value in predictor.model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["state_dict"][name], rtol=0, atol=0)


def test_streaming_resets_history_at_each_boundary(tmp_path: Path) -> None:
    """Start one-epoch histories at participant, segment, time, and manual reset boundaries.

    Args:
        tmp_path (Path): Temporary directory for the synthetic checkpoint.
    """
    path = tmp_path / "model.pt"
    torch.save(_checkpoint(), path)
    predictor = StreamingEpochPredictor(path)
    features = _features(predictor.feature_columns, 0)
    first = predictor.predict_epoch("p1", 0, 0, features)
    cases = [
        ("p1", 0, 30, None, 2), ("p2", 0, 0, "participant_change", 1),
        ("p2", 0, 30, None, 2), ("p2", 1, 60, "segment_change", 1),
        ("p2", 1, 90, None, 2), ("p2", 1, 150, "time_gap", 1),
    ]
    for participant, segment, time, reason, history in cases:
        result = predictor.predict_epoch(participant, segment, time, features)
        assert (result["reset_reason"], result["history_epochs"]) == (reason, history)
        if reason:
            assert result["probabilities"] == first["probabilities"]
    predictor.reset()
    assert predictor.predict_epoch("p1", 0, 0, features) == first


def test_invalid_epoch_does_not_change_history(tmp_path: Path) -> None:
    """Reject invalid metadata and feature values while preserving order and context.

    Args:
        tmp_path (Path): Temporary directory for the synthetic checkpoint.
    """
    path = tmp_path / "model.pt"
    torch.save(_checkpoint(), path)
    predictor = StreamingEpochPredictor(path)
    features = _features(predictor.feature_columns, 0)
    predictor.predict_epoch("p1", 0, 0, features)
    channel = predictor.feature_columns[0]
    cases = [
        (("p1", 1, 0, features), "strictly increase"),
        (("p1", 1, -30, features), "strictly increase"),
        (("p1", 0, float("inf"), features), "time must be a finite"),
        (("p1", -1, 30, features), "segment_id"),
        (("", 0, 30, features), "participant_id"),
        (("p2", 0, 0, features | {"target": 0.0}), "feature keys"),
        (("p1", 0, 30, {k: v for k, v in features.items() if k != channel}), "feature keys"),
        (("p1", 0, 30, features | {channel: float("nan")}), "finite number"),
        (("p1", 0, 30, features | {channel: 1e100}), "finite in float32"),
    ]
    for inputs, message in cases:
        with pytest.raises(ValueError, match=message):
            predictor.predict_epoch(*inputs)
    result = predictor.predict_epoch("p1", 0, 30, features)
    assert result["history_epochs"] == 2
    assert result["reset_reason"] is None


@pytest.mark.parametrize(("name", "value", "message"), [
    ("features", list(reversed(FEATURE_COLUMNS)), "feature schema"),
    ("features", [*FEATURE_COLUMNS, MOTION_FEATURE], "feature schema"),
    ("context_epochs", True, "context_epochs"),
    ("context_epochs", 0, "context_epochs"),
    ("scaler_mean", [float("inf")] * 6, "scaler_mean"),
    ("scaler_scale", [1.0] * 5, "scaler_scale"),
    ("scaler_scale", [0.0] * 6, "scaler_scale"),
    ("scaler_scale", [-1.0] * 6, "scaler_scale"),
])
def test_rejects_invalid_checkpoint_metadata(
    tmp_path: Path, name: str, value: Any, message: str,
) -> None:
    """Reject checkpoints with incompatible feature, history, or scaler metadata.

    Args:
        tmp_path (Path): Temporary directory for the synthetic checkpoint.
        name (str): Checkpoint field under test.
        value (Any): Invalid replacement field value.
        message (str): Expected validation message fragment.
    """
    path = tmp_path / "model.pt"
    torch.save(_checkpoint() | {name: value}, path)
    with pytest.raises(ValueError, match=message):
        StreamingEpochPredictor(path)


@pytest.mark.parametrize(("device", "message"), [
    ("auto", "explicit cpu, mps, or cuda"),
    ("mps", "mps device is unavailable"),
    ("cuda", "cuda device is unavailable"),
])
def test_rejects_unsupported_or_unavailable_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, message: str,
) -> None:
    """Require the requested device to be explicit and available.

    Args:
        tmp_path (Path): Temporary directory for the synthetic checkpoint.
        monkeypatch (pytest.MonkeyPatch): Scoped simulation of unavailable devices.
        device (str): Requested device name.
        message (str): Expected validation message fragment.
    """
    path = tmp_path / "model.pt"
    torch.save(_checkpoint(), path)
    monkeypatch.setattr(torch.backends.mps, "is_available", Mock(return_value=False))
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    with pytest.raises(ValueError, match=message):
        StreamingEpochPredictor(path, device=device)


def test_future_epochs_leave_prefix_predictions_unchanged(tmp_path: Path) -> None:
    """Keep prefix predictions independent of later changes to synthetic input.

    Args:
        tmp_path (Path): Temporary directory for the synthetic checkpoint.
    """
    path = tmp_path / "model.pt"
    torch.save(_checkpoint(), path)
    predictor = StreamingEpochPredictor(path)
    rows = [_features(predictor.feature_columns, index) for index in range(8)]
    first = [predictor.predict_epoch("p1", 0, 30 * index, row) for index, row in enumerate(rows)]
    predictor.reset()
    second = []
    for index, row in enumerate(rows):
        if index >= 5:
            row = {name: value * 10 for name, value in row.items()}
        second.append(predictor.predict_epoch("p1", 0, 30 * index, row))
    assert second[:5] == first[:5]
    assert second[5]["probabilities"] != first[5]["probabilities"]
