from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("torch")

import torch  # noqa: E402

from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.window_heart_rate import WindowLocalHeartRate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]



def _history_recipe(history_epochs: int) -> dict[str, Any]:
    """Resolve the measured full-HR recipe with one proposed past-reference horizon.

    Args:
        history_epochs (int): Maximum strictly previous epochs in the HR reference.
    """
    settings = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    settings["window_features"]["history_epochs"] = history_epochs
    return settings



def _normalization_state(mean: float, scale: float) -> dict[str, Any]:
    """Provide synthetic training-only normalization with a fixed original HR scale.

    Args:
        mean (float): Synthetic fitted deviation mean in BPM.
        scale (float): Synthetic positive fitted deviation scale.
    """
    return {"schema_version": 1, "heart_rate_scale": 2.0, "delta_mean": mean,
            "delta_scale": scale, "available_training_targets": 100,
            "scaling_fit": "training_targets_only"}



def test_longer_reference_changes_deviation_without_changing_flag_or_initial_weights() -> None:
    """Retain model construction while allowing the new deviation and fitted normalization."""
    torch.manual_seed(42)
    reference = ModelFactory(_history_recipe(10)).build(6, 4, _normalization_state(0.0, 1.0))
    torch.manual_seed(42)
    candidate = ModelFactory(_history_recipe(20)).build(6, 4, _normalization_state(1.0, 2.0))
    original_parameters = dict(reference.named_parameters())
    changed_parameters = dict(candidate.named_parameters())
    assert original_parameters.keys() == changed_parameters.keys()
    assert all(torch.equal(value, changed_parameters[name])
               for name, value in original_parameters.items())
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 2852

    features = torch.zeros(1, 30, 6)
    features[0, :, FEATURE_COLUMNS.index("heart_rate_mean")] = torch.arange(30)
    lengths = torch.tensor([30])
    before = reference.transform(features, lengths)
    after = candidate.transform(features, lengths)
    assert before[0, -1, 6].item() == pytest.approx(11.0)
    assert after[0, -1, 6].item() == pytest.approx(10.0)
    assert torch.equal(before[:, :, :6], after[:, :, :6])
    assert torch.equal(before[:, :, 7], after[:, :, 7])



def test_longer_reference_normalization_uses_the_same_available_training_targets() -> None:
    """Permit changed deviation moments without changing original scaling or target support."""
    samples: list[dict[str, torch.Tensor]] = []
    for slope in (1.0, 2.0):
        features = torch.zeros(30, 6)
        features[:, FEATURE_COLUMNS.index("heart_rate_mean")] = torch.arange(30) * slope
        samples.append({"features": features})
    samples.append({"features": torch.zeros(4, 6)})
    before = WindowLocalHeartRate.fit(samples, 2.0, _history_recipe(10)["window_features"])
    after = WindowLocalHeartRate.fit(samples, 2.0, _history_recipe(20)["window_features"])
    assert before["available_training_targets"] == after["available_training_targets"] == 2
    assert before["heart_rate_scale"] == after["heart_rate_scale"] == 2.0
    assert before["delta_mean"] == pytest.approx(16.5)
    assert after["delta_mean"] == pytest.approx(31.5)
    assert before["delta_scale"] == pytest.approx(5.5)
    assert after["delta_scale"] == pytest.approx(10.5)



def test_twenty_epoch_reference_ignores_future_and_padded_observations() -> None:
    """Preserve causal prefix and padding contracts under the longer reference."""
    transform = WindowLocalHeartRate(
        _history_recipe(20)["window_features"], _normalization_state(1.0, 2.0),
    )
    features = torch.randn(2, 30, 6)
    lengths = torch.tensor([30, 6])
    baseline = transform(features, lengths)
    future = features.clone()
    future[:, 15:] += 1000
    torch.testing.assert_close(transform(future, lengths)[:, :15], baseline[:, :15], rtol=0, atol=0)
    padded = features.clone()
    padded[1, 6:] = float("nan")
    torch.testing.assert_close(transform(padded, lengths), baseline, rtol=0, atol=0)



def test_longer_reference_checkpoint_restores_horizon_and_state_without_refitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the longer transform definition and fitted moments at inference.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped prohibition on normalization fitting.
    """
    state = _normalization_state(1.0, 2.0)
    factory = ModelFactory(_history_recipe(20))
    model = factory.build(6, 4, state).eval()
    scales = [1.0] * 6
    scales[FEATURE_COLUMNS.index("heart_rate_mean")] = state["heart_rate_scale"]
    checkpoint = {"model_type": "patch_transformer", "context_epochs": 30,
                  "model_config": factory.checkpoint_config(6, 4),
                  "state_dict": model.state_dict(), "window_feature_state": state,
                  "model_features": factory.model_features(list(FEATURE_COLUMNS)),
                  "scaler_scale": scales}
    forbidden = Mock(side_effect=AssertionError("Inference must not fit normalization"))
    monkeypatch.setattr(WindowLocalHeartRate, "fit", forbidden)
    restored = ModelFactory.from_checkpoint(checkpoint)
    assert restored.transform.settings["history_epochs"] == 20
    assert restored.transform.normalization_state == state
    values = torch.randn(2, 30, 6)
    lengths = torch.tensor([30, 8])
    with torch.no_grad():
        torch.testing.assert_close(
            restored(values, lengths), model(values, lengths), rtol=0, atol=0,
        )
    forbidden.assert_not_called()
