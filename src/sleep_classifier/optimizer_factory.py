import math
from typing import Any

import torch
from torch import nn

from sleep_classifier.optimizer_settings import OptimizerSettings


class OptimizerFactory:
    """Construct optimizers with deterministic, auditable parameter groups."""

    def __init__(self, settings: dict[str, Any] | None) -> None:
        """Validate the selected optimizer settings.

        Args:
            settings (dict[str, Any] | None): Optional optimizer configuration.
        """
        self.settings = OptimizerSettings(settings)


    @staticmethod
    def _exempt_parameter_ids(model: nn.Module) -> set[int]:
        """Identify biases and LayerNorm parameters, including shared aliases.

        Args:
            model (nn.Module): Model whose parameter ownership determines exemptions.
        """
        exempt: set[int] = set()
        for module in model.modules():
            for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
                if "bias" in name.split("_") or isinstance(module, nn.LayerNorm):
                    exempt.add(id(parameter))
        return exempt


    def _parameter_groups(self, model: nn.Module) -> list[dict[str, Any]]:
        """Assign each trainable AdamW parameter once in canonical name order.

        Args:
            model (nn.Module): Model supplying the optimizer parameters.
        """
        exempt = self._exempt_parameter_ids(model) if self.settings.exclude_bias_and_norm else set()
        groups: dict[str, dict[str, Any]] = {
            "decay": {"name": "decay", "weight_decay": self.settings.weight_decay,
                      "params": [], "parameter_names": [], "parameter_count": 0},
            "no_decay": {"name": "no_decay", "weight_decay": 0.0,
                         "params": [], "parameter_names": [], "parameter_count": 0},
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = groups["no_decay" if id(parameter) in exempt else "decay"]
            group["params"].append(parameter)
            group["parameter_names"].append(name)
            group["parameter_count"] += parameter.numel()

        return [group for group in groups.values() if group["params"]]


    def describe_groups(self, model: nn.Module) -> list[dict[str, Any]]:
        """Describe the effective decay groups without constructing an optimizer.

        Args:
            model (nn.Module): Model whose named parameters define the groups.
        """
        if self.settings.name == "adam":
            parameters = list(model.named_parameters())
            return [{
                "name": "all", "weight_decay": 0.0,
                "parameter_names": [name for name, _ in parameters],
                "parameter_count": sum(parameter.numel() for _, parameter in parameters),
            }]

        return [{key: value for key, value in group.items() if key != "params"}
                for group in self._parameter_groups(model)]


    def create(self, model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        """Build the original Adam path or explicit AdamW decay groups.

        Args:
            model (nn.Module): Initialized model supplying the optimizer parameters.
            learning_rate (float): Positive finite update rate shared by all groups.
        """
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("Optimizer learning_rate must be finite and positive")
        if self.settings.name == "adam":
            return torch.optim.Adam(model.parameters(), lr=float(learning_rate))

        return torch.optim.AdamW(self._parameter_groups(model), lr=float(learning_rate))
