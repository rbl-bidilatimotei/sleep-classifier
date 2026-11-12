import math
from typing import Any

import torch

from sleep_classifier.optimizer_adapter import OptimizerAdapter


class PlateauSettings:
    """Validate optional metric-driven learning-rate controls."""

    def __init__(self, values: dict[str, Any] | None) -> None:
        """Resolve supported scheduler controls without constructing an optimizer.

        Args:
            values (dict[str, Any] | None): Optional plateau schedule configuration.
        """
        self.enabled = values is not None
        self.monitor = "validation_macro_f1"
        self.mode = "max"
        self.values: dict[str, Any] = {}
        if not self.enabled:
            return
        if not isinstance(values, dict):
            raise ValueError("lr_scheduler must be a mapping or None")
        defaults = {
            "kind": "reduce_on_plateau", "monitor": "validation_macro_f1", "mode": "max",
            "factor": 0.3, "patience": 3, "threshold": 0.0001, "threshold_mode": "abs",
            "cooldown": 0, "min_lr": 0.00003, "eps": 1e-8,
        }
        if set(values) - set(defaults):
            raise ValueError("Unknown lr_scheduler settings")
        if values.get("kind") != "reduce_on_plateau":
            raise ValueError("lr_scheduler.kind must be reduce_on_plateau")
        self.values = defaults | values
        self._validate()
        self.monitor = self.values["monitor"]
        self.mode = self.values["mode"]


    def _validate(self) -> None:
        """Reject inconsistent metric goals and invalid numeric controls."""
        goals = {"validation_macro_f1": "max", "validation_loss": "min"}
        if (not isinstance(self.values["monitor"], str) or self.values["monitor"] not in goals
                or self.values["mode"] != goals[self.values["monitor"]]):
            raise ValueError("Scheduler mode must match the validation metric goal")
        if self.values["threshold_mode"] not in ("abs", "rel"):
            raise ValueError("Scheduler threshold_mode must be abs or rel")
        for name in ("patience", "cooldown"):
            if type(self.values[name]) is not int or self.values[name] < 0:
                raise ValueError(f"Scheduler {name} must be a nonnegative integer")
        for name in ("factor", "threshold", "min_lr", "eps"):
            value = self.values[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Scheduler {name} must be nonnegative and finite")
        if not 0 < self.values["factor"] < 1:
            raise ValueError("Scheduler factor must be strictly between zero and one")


class PlateauSchedule:
    """Advance a plain Adam schedule after a completed validation observation."""

    def __init__(self, settings: PlateauSettings, optimizer: OptimizerAdapter) -> None:
        """Construct the PyTorch scheduler without changing learning rates or random state.

        Args:
            settings (PlateauSettings): Enabled and validated plateau controls.
            optimizer (OptimizerAdapter): Plain Adam update interface.
        """
        if not settings.enabled:
            raise ValueError("PlateauSchedule requires enabled scheduler settings")
        if optimizer.settings.values or not isinstance(optimizer.optimizer, torch.optim.Adam):
            raise ValueError("Plateau schedules currently require plain Adam")
        if settings.values["min_lr"] > min(optimizer.learning_rates()):
            raise ValueError("Scheduler min_lr cannot exceed the initial learning rate")
        self.settings = settings
        self.optimizer = optimizer
        options = {key: value for key, value in settings.values.items()
                   if key not in ("kind", "monitor")}
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.optimizer, **options,
        )


    def advance(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Observe validation and return the rates used and planned for the next epoch.

        Args:
            summary (dict[str, Any]): Completed epoch scores containing the monitored metric.
        """
        metric = summary.get(self.settings.monitor)
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(metric)):
            raise ValueError("Scheduler requires a finite monitored validation score")
        used = self.optimizer.learning_rates()
        self.scheduler.step(float(metric))
        upcoming = self.optimizer.learning_rates()
        return {
            "learning_rates_used": used, "learning_rates_next_epoch": upcoming,
            "scheduler_reduced": any(after < before
                                     for before, after in zip(used, upcoming, strict=True)),
        }
