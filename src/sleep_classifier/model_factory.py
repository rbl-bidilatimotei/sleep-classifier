import math
from typing import Any

from torch import nn

from sleep_classifier.gru_model import CausalGRUClassifier


class ModelFactory:
    """Validate architecture settings and construct compatible sequence classifiers."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Store the architecture selected by a training recipe.

        Args:
            settings (dict[str, Any]): Training settings and optional architecture settings.
        """
        self.settings = settings
        self.model_type = settings.get("model_type", "gru")

        if self.model_type not in ("gru", "transformer"):
            raise ValueError("model_type must be gru or transformer")

        self.checkpoint_filename = (
            "gru_checkpoint.pt" if self.model_type == "gru" else "model_checkpoint.pt"
        )
        self.prediction_column = "gru_prediction" if self.model_type == "gru" else "prediction"

        if self.model_type == "gru":
            if "transformer" in settings:
                raise ValueError("transformer settings require model_type transformer")
            self.architecture = {"hidden_size": settings["hidden_size"]}
            self._positive_integer("hidden_size", self.architecture["hidden_size"])
        else:
            if "hidden_size" in settings:
                raise ValueError("hidden_size is a GRU setting; use transformer.d_model")
            self.architecture = self._transformer_settings()


    @staticmethod
    def _positive_integer(name: str, value: Any) -> None:
        """Reject noninteger or nonpositive architecture dimensions.

        Args:
            name (str): Setting name used in the error message.
            value (Any): Proposed architecture dimension.
        """
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


    def _transformer_settings(self) -> dict[str, Any]:
        """Validate attention dimensions and the supported history length."""
        supplied = self.settings.get("transformer", {})
        if not isinstance(supplied, dict):
            raise ValueError("transformer must be a settings mapping")

        defaults = {
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "feedforward_dim": 32,
            "dropout": 0.1,
            "max_context": self.settings.get("context_epochs", 10),
        }
        if set(supplied) - set(defaults):
            raise ValueError("Unknown transformer settings")

        values = defaults | supplied
        self._positive_integer("context_epochs", self.settings.get("context_epochs", 10))
        for name in ("d_model", "nhead", "num_layers", "feedforward_dim", "max_context"):
            self._positive_integer(name, values[name])

        if values["d_model"] % values["nhead"]:
            raise ValueError("d_model must be divisible by nhead")
        dropout = values["dropout"]
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")
        if values["max_context"] < self.settings.get("context_epochs", 10):
            raise ValueError("max_context must cover context_epochs")

        return values


    def checkpoint_config(self, input_size: int, num_classes: int) -> dict[str, Any]:
        """Return constructor arguments without coupling checkpoints to the training loop.

        Args:
            input_size (int): Number of prepared features per epoch.
            num_classes (int): Number of output sleep-stage classes.
        """
        self._positive_integer("input_size", input_size)
        self._positive_integer("num_classes", num_classes)

        return {"input_size": input_size, "num_classes": num_classes} | self.architecture


    def build(self, input_size: int, num_classes: int) -> nn.Module:
        """Construct a fresh model with the configured architecture.

        Args:
            input_size (int): Number of prepared features per epoch.
            num_classes (int): Number of output sleep-stage classes.
        """
        arguments = self.checkpoint_config(input_size, num_classes)
        if self.model_type == "gru":
            return CausalGRUClassifier(**arguments)

        from sleep_classifier.transformer_model import CausalTransformerClassifier

        return CausalTransformerClassifier(**arguments)


    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> nn.Module:
        """Restore a current or legacy model on CPU for inference.

        Args:
            checkpoint (dict[str, Any]): Weights and constructor settings loaded from a checkpoint.
        """
        model_type = checkpoint.get("model_type", "gru")
        arguments = dict(checkpoint["model_config"])
        input_size = arguments.pop("input_size")
        num_classes = arguments.pop("num_classes")

        if model_type == "gru":
            if set(arguments) != {"hidden_size"}:
                raise ValueError("Unexpected GRU checkpoint constructor settings")
            settings = {"model_type": model_type} | arguments
        elif model_type == "transformer":
            settings = {
                "model_type": model_type,
                "context_epochs": checkpoint["context_epochs"],
                "transformer": arguments,
            }
        else:
            raise ValueError("Unsupported checkpoint model_type")

        model = cls(settings).build(input_size, num_classes)
        model.load_state_dict(checkpoint["state_dict"])

        return model.eval()
