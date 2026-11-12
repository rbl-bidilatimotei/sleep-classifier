from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from sleep_classifier.lr_schedule import PlateauSchedule, PlateauSettings  # noqa: E402
from sleep_classifier.optimizer_adapter import OptimizerAdapter  # noqa: E402


def _schedule(controls: dict[str, Any] | None = None) -> PlateauSchedule:
    """Create an isolated scheduler around an untrained synthetic Adam parameter.

    Args:
        controls (dict[str, Any] | None): Optional validated schedule overrides.
    """
    model = torch.nn.Linear(1, 1)
    optimizer = OptimizerAdapter.create(
        torch.optim.Adam(model.parameters(), lr=0.001), model, 100, Path("unused"),
    )
    return PlateauSchedule(PlateauSettings({"kind": "reduce_on_plateau"} | (controls or {})),
                           optimizer)


def test_plateau_patience_three_requires_four_nonimproving_evaluations() -> None:
    """Apply a reduction only after PyTorch's documented patience boundary."""
    schedule = _schedule()
    observations = [schedule.advance({"validation_macro_f1": 0.5}) for _ in range(5)]

    assert [row["scheduler_reduced"] for row in observations] == [False] * 4 + [True]
    assert observations[-1]["learning_rates_used"] == [0.001]
    assert observations[-1]["learning_rates_next_epoch"] == pytest.approx([0.0003])
    assert schedule.advance({"validation_macro_f1": 0.6})["learning_rates_used"] == [0.0003]


def test_plateau_cooldown_and_minimum_rate_are_respected() -> None:
    """Reset bad-epoch counting during cooldown and stop reductions at the configured floor."""
    schedule = _schedule({"patience": 0, "cooldown": 1, "min_lr": 0.00009})
    rates = [schedule.advance({"validation_macro_f1": 0.5})["learning_rates_next_epoch"][0]
             for _ in range(8)]

    assert rates == pytest.approx([0.001, 0.0003, 0.0003, 0.00009,
                                   0.00009, 0.00009, 0.00009, 0.00009])


def test_plateau_threshold_and_loss_monitor_use_the_configured_goal() -> None:
    """Support absolute score thresholds and the inverse goal of validation loss."""
    schedule = _schedule({"patience": 0, "threshold": 0.01})
    schedule.advance({"validation_macro_f1": 0.5})
    assert schedule.advance({"validation_macro_f1": 0.505})["scheduler_reduced"]
    lower = _schedule({"monitor": "validation_loss", "mode": "min", "patience": 0})
    lower.advance({"validation_loss": 0.5})
    assert not lower.advance({"validation_loss": 0.4})["scheduler_reduced"]
    assert lower.advance({"validation_loss": 0.5})["scheduler_reduced"]


@pytest.mark.parametrize("controls", [
    {}, [], {"kind": "unknown"}, {"kind": "reduce_on_plateau", "unknown": 1},
    {"kind": "reduce_on_plateau", "factor": 1},
    {"kind": "reduce_on_plateau", "factor": 0},
    {"kind": "reduce_on_plateau", "factor": True},
    {"kind": "reduce_on_plateau", "factor": float("nan")},
    {"kind": "reduce_on_plateau", "patience": True},
    {"kind": "reduce_on_plateau", "patience": -1},
    {"kind": "reduce_on_plateau", "cooldown": 1.0},
    {"kind": "reduce_on_plateau", "min_lr": -1},
    {"kind": "reduce_on_plateau", "eps": float("inf")},
    {"kind": "reduce_on_plateau", "threshold": True},
    {"kind": "reduce_on_plateau", "threshold_mode": "unknown"},
    {"kind": "reduce_on_plateau", "monitor": "validation_loss", "mode": "max"},
    {"kind": "reduce_on_plateau", "monitor": ["validation_macro_f1"]},
])
def test_plateau_settings_reject_invalid_controls(controls: Any) -> None:
    """Reject malformed or unsupported controls before optimizer construction.

    Args:
        controls (Any): Invalid scheduler input to reject.
    """
    with pytest.raises(ValueError):
        PlateauSettings(controls)


def test_plateau_none_is_disabled_and_constructor_does_not_change_rng() -> None:
    """Keep optional absence inert and construction free of random-state consumption."""
    disabled = PlateauSettings(None)
    assert not disabled.enabled and disabled.values == {}
    model = torch.nn.Linear(1, 1)
    optimizer = OptimizerAdapter.create(torch.optim.Adam(model.parameters(), lr=0.001),
                                        model, 100, Path("unused"))
    state = torch.get_rng_state().clone()
    schedule = PlateauSchedule(PlateauSettings({"kind": "reduce_on_plateau"}), optimizer)
    assert torch.equal(state, torch.get_rng_state())
    assert schedule.optimizer.learning_rates() == [0.001]
    with pytest.raises(ValueError, match="enabled"):
        PlateauSchedule(disabled, optimizer)
    with pytest.raises(ValueError, match="min_lr"):
        _schedule({"min_lr": 0.01})


@pytest.mark.parametrize("score", [None, float("nan"), True])
def test_plateau_rejects_missing_or_nonfinite_metric(score: Any) -> None:
    """Fail rather than reducing rates from invalid observations.

    Args:
        score (Any): Invalid monitored score.
    """
    with pytest.raises(ValueError, match="finite"):
        _schedule().advance({"validation_macro_f1": score})
