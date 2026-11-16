import math
from typing import Any


class OptimizerSettings:
    """Validate the optimizer and its explicit decoupled decay policy."""

    def __init__(self, values: dict[str, Any] | None) -> None:
        """Resolve defaults while preserving the original Adam recipe.

        Args:
            values (dict[str, Any] | None): Optional optimizer configuration.
        """
        if values is not None and not isinstance(values, dict):
            raise ValueError("optimizer must be a mapping or null")
        supplied = dict(values or {})
        if set(supplied) - {"name", "weight_decay", "exclude_bias_and_norm"}:
            raise ValueError("Unknown optimizer settings")

        self.name = supplied.get("name", "adam")
        if self.name not in ("adam", "adamw"):
            raise ValueError("optimizer name must be adam or adamw")
        self.weight_decay = supplied.get("weight_decay", 0.0)
        if (isinstance(self.weight_decay, bool)
                or not isinstance(self.weight_decay, (int, float))
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("optimizer weight_decay must be finite and nonnegative")
        self.weight_decay = float(self.weight_decay)
        self.exclude_bias_and_norm = supplied.get("exclude_bias_and_norm", self.name == "adamw")
        if type(self.exclude_bias_and_norm) is not bool:
            raise ValueError("optimizer exclude_bias_and_norm must be a boolean")
        if self.name == "adam" and (self.weight_decay or self.exclude_bias_and_norm):
            raise ValueError("Adam supports only zero decay without parameter exclusions")

        self.values = {
            "name": self.name,
            "weight_decay": self.weight_decay,
            "exclude_bias_and_norm": self.exclude_bias_and_norm,
        }
