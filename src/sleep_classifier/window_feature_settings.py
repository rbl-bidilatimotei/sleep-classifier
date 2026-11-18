import math
from typing import Any


class WindowFeatureSettings:
    """Validate optional deterministic channels derived inside each supplied window."""

    def __init__(self, values: dict[str, Any] | None) -> None:
        """Resolve the heart-rate diagnostic without enabling it for existing recipes.

        Args:
            values (dict[str, Any] | None): Optional window-local feature configuration.
        """
        self.enabled = values is not None
        self.values: dict[str, Any] | None = None
        if not self.enabled:
            return
        required = {"name", "history_epochs", "min_history_epochs"}
        if (not isinstance(values, dict) or not required <= set(values)
                or set(values) - required - {"masked_channels"}):
            raise ValueError("window_features must contain the documented settings")
        if values["name"] != "heart_rate_deviation":
            raise ValueError("window_features name must be heart_rate_deviation")
        history = values["history_epochs"]
        minimum = values["min_history_epochs"]
        if type(history) is not int or history <= 0:
            raise ValueError("window_features history_epochs must be a positive integer")
        if type(minimum) is not int or not 1 <= minimum <= history:
            raise ValueError("window_features min_history_epochs must be in [1, history_epochs]")
        if "masked_channels" in values:
            masked = values["masked_channels"]
            approved = {"heart_rate_delta_past", "heart_history_available"}
            if (not isinstance(masked, list) or not masked
                    or any(not isinstance(name, str) or name not in approved for name in masked)
                    or len(set(masked)) != len(masked)):
                raise ValueError(
                    "window_features masked_channels must list unique derived channels"
                )
        self.values = dict(values)
        if "masked_channels" in self.values:
            self.values["masked_channels"] = list(self.values["masked_channels"])


    @property
    def masked_channels(self) -> list[str]:
        """Return a copy of the selected zero-masked derived channels."""
        if self.values is None:
            return []
        return list(self.values.get("masked_channels", []))


    def validate_context(self, context_epochs: int) -> None:
        """Require the supplied raw context to contain the declared previous-history bound.

        Args:
            context_epochs (int): Maximum raw epochs passed to the sequence model.
        """
        if type(context_epochs) is not int or context_epochs <= 0:
            raise ValueError("context_epochs must be a positive integer")
        if self.enabled and self.values["history_epochs"] >= context_epochs:
            raise ValueError("Window feature history must be shorter than context_epochs")


    def model_features(self, raw_features: list[str]) -> list[str]:
        """Describe effective model channels without changing the stored raw feature schema.

        Args:
            raw_features (list[str]): Ordered prepared input feature names.
        """
        if (not isinstance(raw_features, list)
                or any(not isinstance(name, str) or not name for name in raw_features)
                or len(set(raw_features)) != len(raw_features)):
            raise ValueError("Raw feature names must be unique nonempty strings")
        derived = ["heart_rate_delta_past", "heart_history_available"]
        if self.enabled:
            if set(derived) & set(raw_features):
                raise ValueError("Derived window channels must not already exist in raw features")
            return [*raw_features, *derived]
        return list(raw_features)


    @staticmethod
    def validate_state(state: dict[str, Any]) -> None:
        """Reject incomplete, nonfinite, or unsupported normalization metadata.

        Args:
            state (dict[str, Any]): Proposed fitted-state mapping.
        """
        keys = {"schema_version", "heart_rate_scale", "delta_mean", "delta_scale",
                "available_training_targets", "scaling_fit"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("Window feature normalization state has an invalid schema")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("Window feature normalization requires schema version one")
        if (state["scaling_fit"] != "training_targets_only"
                or type(state["available_training_targets"]) is not int
                or state["available_training_targets"] <= 0):
            raise ValueError("Window scaling must use available training targets only")
        for name in ("heart_rate_scale", "delta_mean", "delta_scale"):
            value = state[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError("Window normalization values must be finite numbers")
        if state["heart_rate_scale"] <= 0 or state["delta_scale"] <= 0:
            raise ValueError("Window normalization scales must be positive")
