import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")
import torch  # noqa: E402

from sleep_classifier.checkpoint_audit import CheckpointReplayAudit  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.neural_training import NeuralExperiment  # noqa: E402
from sleep_classifier.window_feature_model import WindowFeatureClassifier  # noqa: E402
from sleep_classifier.window_heart_rate import WindowLocalHeartRate  # noqa: E402

WINDOW_SETTINGS = {"name": "heart_rate_deviation", "history_epochs": 10, "min_history_epochs": 4}


def _settings() -> dict[str, Any]:
    """Read the fixed reference recipe without enabling data preparation or optimization."""
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/transformer-patch.yaml").read_text())


def _state() -> dict[str, Any]:
    """Provide finite synthetic normalization fitted on declared training targets."""
    return {
        "schema_version": 1, "heart_rate_scale": 2.0, "delta_mean": 0.2,
        "delta_scale": 1.5, "available_training_targets": 20,
        "scaling_fit": "training_targets_only",
    }


def _checkpoint(model: torch.nn.Module, factory: ModelFactory) -> dict[str, Any]:
    """Build a complete portable synthetic augmented checkpoint.

    Args:
        model (torch.nn.Module): Initialized augmented patch classifier.
        factory (ModelFactory): Factory holding the enabled window-local recipe.
    """
    return {
        "model_type": factory.model_type, "context_epochs": factory.settings["context_epochs"],
        "model_config": factory.checkpoint_config(6, 4),
        "features": list(FEATURE_COLUMNS), "model_features": factory.model_features(
            list(FEATURE_COLUMNS)
        ),
        "window_feature_state": _state(),
        "scaler_scale": [2.0] * 6, "scaler_mean": [0.0] * 6,
        "state_dict": copy.deepcopy(model.state_dict()),
    }


def test_disabled_window_features_preserve_initial_weights_and_predictions() -> None:
    """Keep the default model path identical when the optional recipe is absent or null."""
    settings = _settings()
    torch.manual_seed(42)
    first = ModelFactory(settings).build(6, 4).eval()
    torch.manual_seed(42)
    second = ModelFactory(settings | {"window_features": None}).build(6, 4).eval()
    assert type(first) is type(second)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(torch.equal(value, second.state_dict()[name])
               for name, value in first.state_dict().items())
    inputs = torch.randn(3, 30, 6)
    lengths = torch.tensor([30, 10, 1])
    torch.testing.assert_close(first(inputs, lengths), second(inputs, lengths), rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in first.parameters()) == 2756


def test_augmented_factory_keeps_raw_input_and_restores_predictions() -> None:
    """Restore all effective preprocessing without fitting or extending the raw window."""
    factory = ModelFactory(_settings() | {"window_features": WINDOW_SETTINGS})
    model = factory.build(6, 4, window_feature_state=_state()).eval()
    assert isinstance(model, WindowFeatureClassifier)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2852
    assert factory.checkpoint_config(6, 4)["input_size"] == 6
    assert factory.checkpoint_config(6, 4)["window_features"] == WINDOW_SETTINGS
    assert len(factory.model_features(list(FEATURE_COLUMNS))) == 8
    features = torch.randn(4, 30, 6)
    lengths = torch.tensor([30, 10, 4, 1])
    restored = ModelFactory.from_checkpoint(_checkpoint(model, factory))
    torch.testing.assert_close(
        model(features, lengths), restored(features, lengths), rtol=0, atol=0,
    )
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())


def test_availability_control_preserves_initial_weights_and_random_stream() -> None:
    """Keep full and availability-only model initialization identical under the same seed."""
    full_factory = ModelFactory(_settings() | {"window_features": WINDOW_SETTINGS})
    masked_settings = WINDOW_SETTINGS | {"masked_channels": ["heart_rate_delta_past"]}
    masked_factory = ModelFactory(_settings() | {"window_features": masked_settings})
    torch.manual_seed(42)
    full = full_factory.build(6, 4, _state())
    full_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    masked = masked_factory.build(6, 4, _state())
    assert torch.equal(torch.get_rng_state(), full_rng)
    assert full.state_dict().keys() == masked.state_dict().keys()
    assert all(torch.equal(value, masked.state_dict()[name])
               for name, value in full.state_dict().items())
    assert list(dict(full.named_parameters())) == list(dict(masked.named_parameters()))
    assert list(dict(full.named_buffers())) == list(dict(masked.named_buffers()))
    assert sum(parameter.numel() for parameter in masked.parameters()) == 2852
    assert masked_factory.checkpoint_config(6, 4)["input_size"] == 6
    assert masked_factory.checkpoint_config(6, 4)["window_features"] == masked_settings
    assert masked_factory.model_features(list(FEATURE_COLUMNS)) == full_factory.model_features(
        list(FEATURE_COLUMNS)
    )
    assert masked.transform.normalization_state == full.transform.normalization_state


@pytest.mark.parametrize("mutation", ["missing", "scale", "features"])
def test_augmented_checkpoint_rejects_inconsistent_state(mutation: str) -> None:
    """Reject ambiguous or contradictory preprocessing metadata before inference.

    Args:
        mutation (str): Checkpoint metadata inconsistency.
    """
    factory = ModelFactory(_settings() | {"window_features": WINDOW_SETTINGS})
    checkpoint = _checkpoint(factory.build(6, 4, _state()), factory)
    if mutation == "missing":
        checkpoint.pop("window_feature_state")
    elif mutation == "scale":
        checkpoint["scaler_scale"][0] = 3.0
    else:
        checkpoint["model_features"] = list(FEATURE_COLUMNS)
    with pytest.raises(ValueError):
        ModelFactory.from_checkpoint(checkpoint)


def test_augmented_factory_requires_normalization_and_full_history_definition() -> None:
    """Reject construction without fitted state or with a history outside the raw context."""
    factory = ModelFactory(_settings() | {"window_features": WINDOW_SETTINGS})
    with pytest.raises(ValueError, match="training-fitted"):
        factory.build(6, 4)
    with pytest.raises(ValueError, match="shorter"):
        ModelFactory(_settings() | {"context_epochs": 10, "window_features": WINDOW_SETTINGS})
    with pytest.raises(ValueError, match="six-feature"):
        factory.build(8, 4, _state())


def _write_synthetic_fold(root: Path) -> Path:
    """Create one small development-only patch experiment with distinct validation heart rates.

    Args:
        root (Path): Temporary synthetic experiment root.
    """
    fold = root / "fold-01"
    features = fold / "features"
    provenance = {
        "split": {"train": ["1"], "validation": ["3"], "test": ["2"]},
        "feature_columns": list(FEATURE_COLUMNS),
        "participants": [
            {"participant_id": "1", "partition": "train", "eligible_epochs": 8},
            {"participant_id": "3", "partition": "validation", "eligible_epochs": 8},
        ],
    }
    for partition, person, heart in (
        ("train", "1", [50, 52, 54, 58, 57, 61, 60, 65]),
        ("validation", "3", [140, 160, 150, 190, 170, 210, 220, 250]),
    ):
        directory = features / partition
        directory.mkdir(parents=True)
        frame = pd.DataFrame({feature: np.linspace(0.0, 1.0, 8) for feature in FEATURE_COLUMNS})
        frame["heart_rate_mean"] = heart
        frame = frame.assign(time=np.arange(8) * 30, target=np.arange(8) % 4,
                             segment_id=0, eligible=True)
        frame.to_csv(directory / f"{person}.csv", index=False)
    (features / "provenance.json").write_text(json.dumps(provenance))
    (fold / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "features"}))
    settings = _settings() | {
        "prepare_config": "prepare.yaml", "output_dir": "results", "epochs": 1,
        "batch_size": 4, "device": "cpu", "window_features": WINDOW_SETTINGS,
        "diagnostics": {"enabled": True, "evaluate_training": True,
                        "save_predictions": True, "checkpoint_epochs": [1]},
    }
    path = fold / "gru.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_training_only_scaling_and_synthetic_checkpoint_replay(tmp_path: Path) -> None:
    """Use only training targets for scaling and replay stored state without refitting.

    Args:
        tmp_path (Path): Temporary synthetic fold root.
    """
    experiment = NeuralExperiment(_write_synthetic_fold(tmp_path))
    check = experiment.run_check()
    assert check["input_shape"] == [4, 4, 6]
    assert check["effective_input_shape"] == [4, 4, 8]
    assert check["parameter_count"] == 2852
    state = check["window_feature_state"]
    assert state["available_training_targets"] == 4
    raw_heart = np.asarray([50, 52, 54, 58, 57, 61, 60, 65], dtype=float)
    expected = np.asarray([raw_heart[index] - raw_heart[:index].mean()
                           for index in range(4, 8)])
    assert state["delta_mean"] == pytest.approx(expected.mean(), abs=1e-5)
    assert state["delta_scale"] == pytest.approx(expected.std(), abs=1e-5)

    metrics = json.loads(experiment.run().read_text())
    assert metrics["features"] == list(FEATURE_COLUMNS)
    assert len(metrics["model_features"]) == 8
    assert metrics["window_features"]["scaling"] == state
    replay = CheckpointReplayAudit(experiment.output_dir.parent).build()
    assert replay["weights_unchanged"] and not replay["gradients_created"]
    assert replay["matches_saved_atol_2e_6"]


@pytest.mark.parametrize("name", ["heart_rate_scale", "delta_mean", "delta_scale"])
def test_augmented_checkpoint_rejects_contradictory_buffers(name: str) -> None:
    """Reject state tensors that silently change the declared preprocessing transform.

    Args:
        name (str): Transform buffer to corrupt in the stored model weights.
    """
    factory = ModelFactory(_settings() | {"window_features": WINDOW_SETTINGS})
    checkpoint = _checkpoint(factory.build(6, 4, _state()), factory)
    checkpoint["state_dict"][f"transform.{name}"] += 0.5
    with pytest.raises(ValueError, match="buffers"):
        ModelFactory.from_checkpoint(checkpoint)


def _unexpected_fit(
    samples: Any, heart_rate_scale: float, settings: dict[str, Any],
) -> dict[str, Any]:
    """Fail if checkpoint inference attempts to fit normalization again.

    Args:
        samples (Any): Inference must not supply training samples.
        heart_rate_scale (float): Stored source-channel scale.
        settings (dict[str, Any]): Stored feature recipe.
    """
    raise AssertionError("Inference attempted to fit window normalization")


@pytest.mark.parametrize("masked_channels", [
    None, ["heart_rate_delta_past"], ["heart_history_available"],
    ["heart_rate_delta_past", "heart_history_available"],
])
def test_checkpoint_inference_never_refits_window_scaling(
    monkeypatch: pytest.MonkeyPatch, masked_channels: list[str] | None,
) -> None:
    """Restore channel masks and predictions entirely from stored state without fitting.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped prohibition on transform fitting.
        masked_channels (list[str] | None): Optional valid channel mask to persist.
    """
    window_settings = dict(WINDOW_SETTINGS)
    if masked_channels is not None:
        window_settings["masked_channels"] = masked_channels
    factory = ModelFactory(_settings() | {"window_features": window_settings})
    original = factory.build(6, 4, _state()).eval()
    checkpoint = _checkpoint(original, factory)
    monkeypatch.setattr(WindowLocalHeartRate, "fit", staticmethod(_unexpected_fit))
    model = ModelFactory.from_checkpoint(checkpoint)
    assert checkpoint["model_config"]["window_features"] == window_settings
    assert model.transform.settings == window_settings
    assert model.transform.normalization_state == _state()
    inputs = torch.randn(2, 30, 6)
    lengths = torch.tensor([30, 3])
    with torch.no_grad():
        output = model(inputs, lengths)
        torch.testing.assert_close(output, original(inputs, lengths), rtol=0, atol=0)
        augmented = model.transform(inputs, lengths)
    assert output.shape == (2, 4)
    assert augmented.shape == (2, 30, 8)
    for index, name in enumerate(("heart_rate_delta_past", "heart_history_available"), start=6):
        if name in (masked_channels or []):
            assert torch.equal(augmented[:, :, index], torch.zeros_like(augmented[:, :, index]))


@pytest.mark.parametrize("model_type", ["gru", "transformer"])
def test_window_features_restore_across_existing_architectures(model_type: str) -> None:
    """Keep preprocessing outside architecture constructors while preserving checkpoint context.

    Args:
        model_type (str): Existing backbone architecture to restore.
    """
    settings = _settings()
    settings.pop("patch_size")
    settings["model_type"] = model_type
    settings["window_features"] = WINDOW_SETTINGS
    if model_type == "gru":
        settings.pop("transformer")
        settings["hidden_size"] = 4
    factory = ModelFactory(settings)
    model = factory.build(6, 4, _state()).eval()
    restored = ModelFactory.from_checkpoint(_checkpoint(model, factory))
    features = torch.randn(2, 30, 6)
    lengths = torch.tensor([30, 3])
    torch.testing.assert_close(
        model(features, lengths), restored(features, lengths), rtol=0, atol=0,
    )
