import copy
from pathlib import Path
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


def test_patch_factory_preserves_raw_context_and_exposes_patch_token_capacity() -> None:
    """Build the small patch recipe while keeping checkpoint features at raw epoch resolution."""
    factory = ModelFactory({"model_type": "patch_transformer", "context_epochs": 30,
                            "patch_size": 3, "transformer": {"dropout": 0.2}})
    model = factory.build(6, 4)
    config = factory.checkpoint_config(6, 4)

    assert model.input_size == 6 and model.max_context == 30
    assert model.patch_size == 3 and model.backbone.max_context == 10
    assert config["input_size"] == 6 and config["max_context"] == 30
    assert config["patch_size"] == 3
    assert sum(parameter.numel() for parameter in model.parameters()) == 2756
    assert factory.checkpoint_filename == "model_checkpoint.pt"
    assert factory.prediction_column == "prediction"


def test_patch_checkpoint_roundtrip_preserves_metadata_and_logits(tmp_path: Path) -> None:
    """Load portable patch weights through the same CPU inference interface.

    Args:
        tmp_path (Path): Temporary checkpoint directory.
    """
    factory = ModelFactory({"model_type": "patch_transformer", "context_epochs": 3,
                            "patch_size": 3, "transformer": {"d_model": 8, "nhead": 2,
                            "feedforward_dim": 16, "dropout": 0.0}})
    model = factory.build(6, 4).eval()
    checkpoint = {"model_type": factory.model_type,
                  "model_config": factory.checkpoint_config(6, 4),
                  "state_dict": copy.deepcopy(model.state_dict()), "context_epochs": 3}
    path = tmp_path / "patch.pt"
    torch.save(checkpoint, path)
    restored = ModelFactory.from_checkpoint(torch.load(path, weights_only=True))
    features = torch.randn(2, 3, 6)
    lengths = torch.tensor([1, 3])

    torch.testing.assert_close(
        restored(features, lengths), model(features, lengths), atol=0, rtol=0,
    )
    assert not restored.training
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())


@pytest.mark.parametrize("patch_size", [True, 0, -1, 1.5, 4])
def test_patch_factory_rejects_invalid_patch_sizes(patch_size: Any) -> None:
    """Reject unsupported patch lengths before constructing a model.

    Args:
        patch_size (Any): Invalid noninteger, nonpositive, or excessive patch length.
    """
    with pytest.raises(ValueError, match="patch_size"):
        ModelFactory({"model_type": "patch_transformer", "context_epochs": 3,
                      "patch_size": patch_size})


@pytest.mark.parametrize("model_type", ["gru", "transformer"])
def test_patch_size_cannot_be_silently_ignored_by_legacy_models(model_type: str) -> None:
    """Reject patch settings that do not belong to the selected architecture.

    Args:
        model_type (str): Legacy sequence architecture.
    """
    with pytest.raises(ValueError, match="patch_size"):
        ModelFactory(_settings(model_type) | {"patch_size": 1})


@pytest.mark.parametrize("mutation", ["missing_patch_size", "unknown_setting"])
def test_patch_checkpoint_rejects_ambiguous_or_unused_constructor_fields(mutation: str) -> None:
    """Reject patch metadata that cannot define the trained architecture exactly.

    Args:
        mutation (str): Invalid constructor metadata change.
    """
    arguments = {"input_size": 6, "num_classes": 4, "max_context": 3, "patch_size": 3}
    if mutation == "missing_patch_size":
        arguments.pop("patch_size")
    else:
        arguments["unused_setting"] = 1
    with pytest.raises(ValueError):
        ModelFactory.from_checkpoint({"model_type": "patch_transformer", "model_config": arguments,
                                      "context_epochs": 3, "state_dict": {}})
