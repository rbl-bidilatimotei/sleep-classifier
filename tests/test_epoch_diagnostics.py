import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch import nn  # noqa: E402

from sleep_classifier.epoch_diagnostics import (  # noqa: E402
    EpochDiagnostics,
    EpochDiagnosticsSettings,
)


def _fixture() -> tuple[pd.DataFrame, np.ndarray]:
    """Return four labeled observations with correct and confident incorrect predictions."""
    frame = pd.DataFrame({"participant_id": ["p1", "p1", "p2", "p2"],
                          "time": [0, 30, 0, 30], "target": [0, 1, 0, 2]})
    probabilities = np.asarray([[0.8, 0.1, 0.05, 0.05], [0.1, 0.2, 0.6, 0.1],
                                [0.01, 0.89, 0.05, 0.05], [0.1, 0.1, 0.7, 0.1]])
    return frame, probabilities


def _recorder(output_dir: Path) -> EpochDiagnostics:
    """Create a synthetic recorder that retains predictions and one checkpoint.

    Args:
        output_dir (Path): Temporary model output directory.
    """
    settings = EpochDiagnosticsSettings({"enabled": True, "checkpoint_epochs": [1]})
    return EpochDiagnostics(settings, output_dir, {"model_type": "synthetic", "features": ["f"]},
                            gradient_clip=1.0)


def test_epoch_evidence_decomposes_nll_and_preserves_missing_classes(tmp_path: Path) -> None:
    """Verify additive participant/class losses and confidence statistics independently.

    Args:
        tmp_path (Path): Temporary model evidence directory.
    """
    frame, probabilities = _fixture()
    recorder = _recorder(tmp_path)
    recorder.begin_epoch()
    recorder.record_gradient(0.5)
    recorder.record_gradient(2.0)
    expected = -np.log([0.8, 0.2, 0.01, 0.7])
    path = recorder.record_epoch(1, frame, probabilities, 0.6, [0.001], nn.Linear(1, 4),
                                 float(expected.mean()), 0.7, 0.4)
    saved = json.loads(path.read_text())
    validation = saved["validation"]

    assert saved["schema_version"] == 1
    assert validation["nll"] == pytest.approx(expected.mean())
    assert validation["nll_sum"] == pytest.approx(expected.sum())
    assert validation["accuracy"] == 0.5
    assert validation["mean_confidence_wrong"] == pytest.approx((0.6 + 0.89) / 2)
    assert validation["high_confidence_error_fraction"] == 0.25
    assert validation["classes"]["3"] == {
        "count": 0, "nll": None, "nll_sum": 0.0, "nll_contribution": 0.0, "accuracy": None,
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }
    assert validation["confusion_matrix"] == [[1, 1, 0, 0], [0, 0, 1, 0],
                                               [0, 0, 1, 0], [0, 0, 0, 0]]
    wake, light, deep = (validation["classes"][str(label)] for label in range(3))
    assert wake["precision"] == 1.0 and wake["recall"] == 0.5 and wake["support"] == 2
    assert wake["f1"] == pytest.approx(2 / 3)
    assert light["precision"] == light["recall"] == light["f1"] == 0.0
    assert light["support"] == 1
    assert deep["precision"] == 0.5 and deep["recall"] == 1.0 and deep["support"] == 1
    assert deep["f1"] == pytest.approx(2 / 3)
    for grouping in ("classes", "participants"):
        assert sum(row["nll_sum"] for row in validation[grouping].values()) == pytest.approx(
            expected.sum(),
        )
        assert sum(row["nll_contribution"] for row in validation[grouping].values()) == (
            pytest.approx(expected.mean())
        )
    assert saved["training"] == {"evaluation_nll": 0.6, "stochastic_nll": 0.7}
    assert saved["optimizer"]["learning_rates"] == [0.001]
    assert saved["optimizer"]["gradient_norm_pre_clip_mean"] == 1.25
    assert saved["optimizer"]["gradient_norm_pre_clip_max"] == 2
    assert saved["optimizer"]["clipped_step_fraction"] == 0.5


def test_epoch_files_preserve_probabilities_weights_and_rng(tmp_path: Path) -> None:
    """Retain hashed portable artifacts without changing inputs, weights, mode, or RNG.

    Args:
        tmp_path (Path): Temporary model evidence directory.
    """
    frame, probabilities = _fixture()
    original_frame, original_probabilities = frame.copy(deep=True), probabilities.copy()
    model = nn.Linear(1, 4)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    torch_rng = torch.get_rng_state()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    recorder = _recorder(tmp_path)
    path = recorder.record_epoch(1, frame, probabilities, None, [0.001], model, 1.0, 0.8, 0.4)
    saved = json.loads(path.read_text())

    for artifact in saved["files"].values():
        location = recorder.directory / artifact["path"]
        assert not Path(artifact["path"]).is_absolute()
        assert artifact["sha256"] == hashlib.sha256(location.read_bytes()).hexdigest()
    predictions = pd.read_csv(recorder.directory / saved["files"]["validation_predictions"]["path"])
    np.testing.assert_allclose(predictions[[f"probability_{label}" for label in range(4)]],
                               probabilities)
    np.testing.assert_allclose(predictions.nll, -np.log([0.8, 0.2, 0.01, 0.7]))
    checkpoint = torch.load(recorder.directory / saved["files"]["checkpoint"]["path"],
                            weights_only=True)
    assert checkpoint["epoch"] == 1 and checkpoint["features"] == ["f"]
    for name, before in weights.items():
        assert torch.equal(before, model.state_dict()[name])
        assert torch.equal(before, checkpoint["state_dict"][name])
        assert checkpoint["state_dict"][name].device.type == "cpu"
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    actual_numpy_rng = np.random.get_state()
    np.testing.assert_array_equal(numpy_rng[1], actual_numpy_rng[1])
    assert numpy_rng[0] == actual_numpy_rng[0] and numpy_rng[2:] == actual_numpy_rng[2:]
    pd.testing.assert_frame_equal(frame, original_frame)
    np.testing.assert_array_equal(probabilities, original_probabilities)
    with pytest.raises(FileExistsError):
        recorder.record_epoch(1, frame, probabilities, None, [0.001], model, 1.0, 0.8, 0.4)


def test_epoch_recording_can_disable_large_artifacts_and_reset_gradients(tmp_path: Path) -> None:
    """Keep scalar evidence when prediction and intermediate-checkpoint retention are disabled.

    Args:
        tmp_path (Path): Temporary model evidence directory.
    """
    settings = EpochDiagnosticsSettings({"enabled": True, "save_predictions": False,
                                         "evaluate_training": False})
    recorder = EpochDiagnostics(settings, tmp_path, {}, 1.0)
    recorder.record_gradient(2.0)
    recorder.begin_epoch()
    frame, probabilities = _fixture()
    path = recorder.record_epoch(2, frame, probabilities, None, [0.001], nn.Linear(1, 4),
                                 1.0, 0.8, 0.4)
    record = json.loads(path.read_text())
    assert record["files"] == {}
    assert record["optimizer"]["gradient_steps"] == 0
    assert record["optimizer"]["gradient_norm_pre_clip_mean"] is None
    assert record["training"]["evaluation_nll"] is None
    assert not (recorder.directory / "predictions").exists()
    assert not (recorder.directory / "checkpoints").exists()


@pytest.mark.parametrize("settings", [
    {"unknown": True}, {"enabled": 1}, {"save_predictions": "yes"},
    {"evaluate_training": None}, {"checkpoint_epochs": [True]},
    {"checkpoint_epochs": [0]}, {"checkpoint_epochs": [1, 1]},
    {"checkpoint_epochs": "all"}, {"high_confidence_threshold": 1},
    {"high_confidence_threshold": float("nan")}, {"high_confidence_threshold": True},
])
def test_invalid_diagnostic_settings_are_rejected(settings: dict[str, object]) -> None:
    """Reject malformed evidence controls before any training begins.

    Args:
        settings (dict[str, object]): Invalid diagnostic configuration.
    """
    with pytest.raises(ValueError):
        EpochDiagnosticsSettings(settings)


@pytest.mark.parametrize("mutation", ["nan", "sum", "negative", "shape", "duplicate", "target"])
def test_invalid_epoch_probabilities_and_alignment_are_rejected(
    tmp_path: Path, mutation: str,
) -> None:
    """Reject corrupt probabilities and invalid target populations before saving evidence.

    Args:
        tmp_path (Path): Temporary model evidence directory.
        mutation (str): Invalid probability or target condition to introduce.
    """
    frame, probabilities = _fixture()
    if mutation == "nan":
        probabilities[0, 0] = np.nan
    elif mutation == "sum":
        probabilities[0, 0] = 0.5
    elif mutation == "negative":
        probabilities[0] = [-0.1, 0.5, 0.3, 0.3]
    elif mutation == "shape":
        probabilities = probabilities[:-1]
    elif mutation == "duplicate":
        frame.loc[1, "time"] = 0
    else:
        frame.loc[0, "target"] = -1
    with pytest.raises(ValueError):
        _recorder(tmp_path).record_epoch(1, frame, probabilities, None, [0.001],
                                         nn.Linear(1, 4), 1.0, 0.8, 0.4)
    assert not (tmp_path / "diagnostics").exists()
