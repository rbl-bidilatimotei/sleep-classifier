import copy
from typing import Any

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from sleep_classifier.gru_model import CausalGRUClassifier  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.transformer_model import CausalTransformerClassifier  # noqa: E402


def _settings(model_type: str) -> dict[str, Any]:
    """Build a small architecture recipe for synthetic factory checks.

    Args:
        model_type (str): Supported architecture name.
    """
    if model_type == "gru":
        return {"hidden_size": 4, "context_epochs": 3}

    return {
        "model_type": "transformer",
        "context_epochs": 3,
        "transformer": {"d_model": 8, "nhead": 2, "feedforward_dim": 16,
                        "dropout": 0.0, "max_context": 3},
    }


def test_default_recipe_preserves_the_original_gru_constructor() -> None:
    """Keep legacy GRU constructor settings and parameter count unchanged."""
    factory = ModelFactory({"hidden_size": 32})
    assert isinstance(factory.build(6, 4), CausalGRUClassifier)
    assert factory.checkpoint_config(6, 4) == {
        "input_size": 6, "hidden_size": 32, "num_classes": 4,
    }
    assert sum(parameter.numel() for parameter in factory.build(6, 4).parameters()) == 3972


@pytest.mark.parametrize("model_type", ["gru", "transformer"])
def test_checkpoint_restore_preserves_predictions_and_legacy_gru_metadata(model_type: str) -> None:
    """Restore architecture-specific weights without changing predictions.

    Args:
        model_type (str): Architecture used for the synthetic checkpoint.
    """
    factory = ModelFactory(_settings(model_type))
    model = factory.build(6, 4).eval()
    features = torch.randn(2, 3, 6)
    lengths = torch.tensor([2, 3])
    checkpoint = {
        "state_dict": copy.deepcopy(model.state_dict()),
        "model_config": factory.checkpoint_config(6, 4),
        "context_epochs": 3,
    }
    if model_type != "gru":
        checkpoint["model_type"] = model_type
    restored = ModelFactory.from_checkpoint(checkpoint).eval()

    torch.testing.assert_close(
        restored(features, lengths), model(features, lengths), atol=0, rtol=0
    )
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    assert isinstance(restored, CausalGRUClassifier if model_type == "gru"
                      else CausalTransformerClassifier)


@pytest.mark.parametrize("changes", [
    {"model_type": "unknown"},
    {"transformer": {"d_model": 7, "nhead": 2}},
    {"transformer": {"max_context": 2}},
    {"transformer": {"dropout": float("nan")}},
    {"transformer": {"dropout": True}},
    {"transformer": {"d_model": True}},
    {"transformer": {"unused_setting": 1}},
    {"hidden_size": 4},
])
def test_invalid_or_unused_transformer_settings_fail_early(changes: dict[str, Any]) -> None:
    """Reject invalid architecture settings before training starts.

    Args:
        changes (dict[str, Any]): Invalid replacement settings.
    """
    with pytest.raises(ValueError):
        ModelFactory(_settings("transformer") | changes)


def test_transformer_recipe_keeps_the_gru_training_and_fold_controls() -> None:
    """Keep architecture comparisons on the same training settings and participant folds."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1] / "configs"
    gru = yaml.safe_load((root / "gru.yaml").read_text())
    transformer = yaml.safe_load((root / "transformer.yaml").read_text())
    gru.pop("hidden_size")
    transformer.pop("model_type")
    transformer.pop("transformer")
    transformer["output_dir"] = gru["output_dir"]
    assert transformer == gru

    reference = yaml.safe_load((root / "gru-grouped-validation.yaml").read_text())
    comparison = yaml.safe_load((root / "transformer-grouped-validation.yaml").read_text())
    comparison["base_model_config"] = reference["base_model_config"]
    comparison["output_dir"] = reference["output_dir"]
    assert comparison == reference
