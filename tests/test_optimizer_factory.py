from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("torch")
import torch  # noqa: E402
from torch import nn  # noqa: E402

from sleep_classifier.optimizer_adapter import OptimizerAdapter  # noqa: E402
from sleep_classifier.optimizer_factory import OptimizerFactory  # noqa: E402
from sleep_classifier.optimizer_settings import OptimizerSettings  # noqa: E402


def _model() -> nn.Module:
    """Create linear, normalization, and embedding parameters with distinct ownership."""
    model = nn.Module()
    model.projection = nn.Linear(3, 3)
    model.normalization = nn.LayerNorm(3)
    model.position = nn.Embedding(4, 3)
    return model


def test_default_factory_preserves_adam_constructor_and_parameter_order() -> None:
    """Keep the original Adam defaults and canonical parameter order unchanged."""
    model = _model()
    actual = OptimizerFactory(None).create(model, 0.001)
    expected = torch.optim.Adam(model.parameters(), lr=0.001)

    assert type(actual) is torch.optim.Adam
    assert actual.defaults == expected.defaults
    assert actual.param_groups == expected.param_groups
    assert OptimizerSettings(None).values == {
        "name": "adam", "weight_decay": 0.0, "exclude_bias_and_norm": False,
    }


def test_adamw_exempts_biases_and_layernorm_and_decays_positions() -> None:
    """Separate bias and normalization parameters while retaining position decay."""
    model = _model()
    model.attention = nn.MultiheadAttention(4, 2)
    model.recurrent = nn.GRU(3, 3)
    factory = OptimizerFactory({"name": "adamw", "weight_decay": 0.01})
    descriptions = {group["name"]: group for group in factory.describe_groups(model)}

    assert descriptions["decay"]["weight_decay"] == 0.01
    assert descriptions["no_decay"]["weight_decay"] == 0.0
    assert "position.weight" in descriptions["decay"]["parameter_names"]
    assert "attention.in_proj_bias" in descriptions["no_decay"]["parameter_names"]
    assert "recurrent.bias_ih_l0" in descriptions["no_decay"]["parameter_names"]
    assert {"projection.bias", "normalization.weight", "normalization.bias"}.issubset(
        descriptions["no_decay"]["parameter_names"],
    )
    optimizer = factory.create(model, 0.001)
    actual_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(actual_ids) == len(set(actual_ids)) == len(list(model.parameters()))
    assert sum(group["parameter_count"] for group in descriptions.values()) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_shared_parameter_is_exempt_once_and_frozen_parameters_are_absent() -> None:
    """Apply the exemption of any owner and omit frozen parameters from AdamW."""
    model = _model()
    model.normalization.weight = model.projection.weight
    model.position.weight.requires_grad_(False)
    factory = OptimizerFactory({"name": "adamw", "weight_decay": 0.01})
    descriptions = {group["name"]: group for group in factory.describe_groups(model)}
    names = [name for group in descriptions.values() for name in group["parameter_names"]]

    assert "projection.weight" in descriptions["no_decay"]["parameter_names"]
    assert "position.weight" not in names
    assert len(names) == len(set(names)) == 3


def test_decay_exclusions_can_be_disabled_explicitly() -> None:
    """Apply the configured decay to every trainable parameter when exemptions are disabled."""
    descriptions = OptimizerFactory({
        "name": "adamw", "weight_decay": 0.02, "exclude_bias_and_norm": False,
    }).describe_groups(_model())
    assert len(descriptions) == 1
    assert descriptions[0]["name"] == "decay"
    assert descriptions[0]["weight_decay"] == 0.02
    assert "normalization.weight" in descriptions[0]["parameter_names"]


def test_adamw_step_decays_only_the_declared_group() -> None:
    """Shrink only eligible parameters when gradients contain explicit zero tensors."""
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    optimizer = OptimizerFactory({"name": "adamw", "weight_decay": 0.1}).create(model, 0.01)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    for name, parameter in model.named_parameters():
        expected = 0.999 if name in ("projection.weight", "position.weight") else 1.0
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected), rtol=0, atol=0)


def test_adamw_provenance_records_effective_groups(tmp_path: Path) -> None:
    """Expose the selected optimizer and actual decay groups in saved metadata.

    Args:
        tmp_path (Path): Temporary unused optimizer artifact root.
    """
    model = _model()
    factory = OptimizerFactory({"name": "adamw", "weight_decay": 0.01})
    adapter = OptimizerAdapter.create(factory.create(model, 0.001), model, 10, tmp_path)
    assert adapter.provenance() == {
        "base_optimizer": "AdamW", "updates": 0,
        "param_groups": factory.describe_groups(model),
    }


@pytest.mark.parametrize("settings", [
    [], "adamw", {"name": "sgd"}, {"momentum": 0.5},
    {"name": "adamw", "weight_decay": -0.01},
    {"name": "adamw", "weight_decay": float("nan")},
    {"name": "adamw", "weight_decay": float("inf")},
    {"name": "adamw", "weight_decay": True},
    {"name": "adamw", "weight_decay": "0.01"},
    {"name": "adamw", "exclude_bias_and_norm": 1},
    {"name": "adam", "weight_decay": 0.01},
    {"name": "adam", "exclude_bias_and_norm": True},
])
def test_optimizer_settings_reject_invalid_or_unused_values(settings: Any) -> None:
    """Reject unsupported optimizers and ambiguous decay settings.

    Args:
        settings (Any): Invalid configuration candidate.
    """
    with pytest.raises(ValueError):
        OptimizerSettings(settings)


@pytest.mark.parametrize("learning_rate", [0, -1, True, "0.001", float("nan"), float("inf")])
def test_optimizer_factory_rejects_invalid_rate(learning_rate: Any) -> None:
    """Reject invalid update rates before constructing an optimizer.

    Args:
        learning_rate (Any): Invalid update-rate candidate.
    """
    with pytest.raises(ValueError, match="learning_rate"):
        OptimizerFactory(None).create(_model(), learning_rate)


def test_training_config_rejects_adamw_with_opal() -> None:
    """Block the unverified combination before data or model preparation."""
    pytest.importorskip("sklearn")
    from sleep_classifier.neural_training import TrainingConfig

    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "configs/transformer-patch.yaml").read_text())
    settings["optimizer"] = {"name": "adamw", "weight_decay": 0.01}
    settings["optimizer_agent"] = {
        "backend": "keep", "goal": "Do not change", "decisions": 1, "every": 50,
    }
    with pytest.raises(ValueError, match="OPAL"):
        TrainingConfig(settings)


def test_adapter_rejects_adamw_agent_combination(tmp_path: Path) -> None:
    """Reject AdamW agent wrapping even when callers bypass TrainingConfig.

    Args:
        tmp_path (Path): Temporary unused artifact directory.
    """
    model = _model()
    optimizer = OptimizerFactory({"name": "adamw", "weight_decay": 0.01}).create(model, 0.001)
    agent = {"backend": "keep", "goal": "Do not change", "decisions": 1, "every": 50}
    with pytest.raises(ValueError, match="OPAL"):
        OptimizerAdapter.create(optimizer, model, 100, tmp_path, agent)
