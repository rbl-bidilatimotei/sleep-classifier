import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sleep_classifier.optimizer_control import ZeroDecayControl, main  # noqa: E402
from sleep_classifier.optimizer_factory import OptimizerFactory  # noqa: E402


def _settings() -> dict[str, Any]:
    """Build a small causal patch recipe with active dropout."""
    return {
        "model_type": "patch_transformer", "context_epochs": 6, "patch_size": 3,
        "transformer": {"d_model": 8, "nhead": 2, "num_layers": 1,
                        "feedforward_dim": 16, "dropout": 0.2, "max_context": 6},
        "learning_rate": 0.001, "gradient_clip": 1.0,
        "optimizer": {"name": "adamw", "weight_decay": 0.01,
                      "exclude_bias_and_norm": True},
    }


def test_zero_decay_control_matches_updates_and_preserves_rng() -> None:
    """Match grouped updates, gradients, losses, and moments without consuming caller randomness."""
    settings = _settings()
    before_settings = copy.deepcopy(settings)
    random_state = torch.get_rng_state().clone()
    result = ZeroDecayControl(settings, "cpu", steps=4).run()

    assert result["passed"] and result["exact_match"]
    assert result["synthetic_only"] and not result["participant_data_read"]
    assert result["dropout_enabled"]
    assert all(result[f"max_{kind}_difference"] == 0.0
               for kind in ("loss", "gradient", "parameter", "optimizer_state"))
    assert [group["name"] for group in result["groups"]] == ["decay", "no_decay"]
    assert all(group["weight_decay"] == 0.0 for group in result["groups"])
    assert torch.equal(random_state, torch.get_rng_state())
    assert settings == before_settings


def _create_wrong_control(
    self: OptimizerFactory, model: torch.nn.Module, learning_rate: float,
) -> torch.optim.Optimizer:
    """Inject decay into the supposedly zero-decay candidate to verify failure detection.

    Args:
        self (OptimizerFactory): Selected optimizer configuration.
        model (torch.nn.Module): Synthetic model parameters.
        learning_rate (float): Update rate used by both control paths.
    """
    if self.settings.name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def test_zero_decay_control_detects_nonzero_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a control whose candidate silently applies weight decay.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped optimizer factory replacement.
    """
    monkeypatch.setattr(OptimizerFactory, "create", _create_wrong_control)
    result = ZeroDecayControl(_settings(), "cpu", steps=2).run()

    assert not result["passed"] and not result["exact_match"]
    assert result["max_parameter_difference"] > 0.0


@pytest.mark.parametrize("steps", [0, 101, True, 1.5])
def test_zero_decay_control_rejects_unbounded_steps(steps: Any) -> None:
    """Reject invalid step counts before any optimizer update.

    Args:
        steps (Any): Invalid synthetic update limit.
    """
    with pytest.raises(ValueError, match="control steps"):
        ZeroDecayControl(_settings(), "cpu", steps=steps)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_zero_decay_control_rejects_invalid_seed(seed: Any) -> None:
    """Require a nonnegative integer seed before changing random states.

    Args:
        seed (Any): Invalid synthetic seed.
    """
    with pytest.raises(ValueError, match="control seed"):
        ZeroDecayControl(_settings(), "cpu", seed=seed)


def test_control_cli_saves_a_synthetic_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the zero-decay check through a reproducible script command.

    Args:
        tmp_path (Path): Temporary report output directory.
        monkeypatch (pytest.MonkeyPatch): Scoped command argument replacement.
    """
    config = Path(__file__).resolve().parents[1] / "configs/transformer-patch-adamw.yaml"
    output = tmp_path / "control.json"
    monkeypatch.setattr(sys, "argv", [
        "optimizer_control", "--config", str(config), "--device", "cpu",
        "--steps", "2", "--output", str(output),
    ])
    main()
    result = json.loads(output.read_text())

    assert result["passed"] and result["synthetic_only"]
    assert result["device"] == "cpu" and result["steps"] == 2


def test_control_cli_cannot_overwrite_its_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a report path that would destroy the input configuration.

    Args:
        tmp_path (Path): Temporary configuration directory.
        monkeypatch (pytest.MonkeyPatch): Scoped command argument replacement.
    """
    config = tmp_path / "recipe.yaml"
    config.write_text("unchanged configuration")
    monkeypatch.setattr(sys, "argv", [
        "optimizer_control", "--config", str(config), "--output", str(config),
    ])

    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert config.read_text() == "unchanged configuration"
