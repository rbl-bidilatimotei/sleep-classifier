import math
from typing import Any

from torch import nn

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.gru_model import CausalGRUClassifier
from sleep_classifier.window_feature_settings import WindowFeatureSettings


class ModelFactory:
    """Validate architecture settings and construct compatible sequence classifiers."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Store the architecture selected by a training recipe.

        Args:
            settings (dict[str, Any]): Training settings and optional architecture settings.
        """
        self.settings = settings
        self.model_type = settings.get("model_type", "gru")
        self.window_features = WindowFeatureSettings(settings.get("window_features"))
        if self.window_features.enabled:
            self.window_features.validate_context(settings.get("context_epochs", 10))

        if self.model_type not in ("gru", "transformer", "patch_transformer"):
            raise ValueError("model_type must be gru, transformer, or patch_transformer")
        if "patch_size" in settings and self.model_type != "patch_transformer":
            raise ValueError("patch_size requires model_type patch_transformer")

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
            if self.model_type == "patch_transformer":
                patch_size = settings.get("patch_size", 3)
                self._positive_integer("patch_size", patch_size)
                if patch_size > settings.get("context_epochs", 10):
                    raise ValueError("patch_size must not exceed context_epochs")
                self.architecture["patch_size"] = patch_size


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

        arguments = {"input_size": input_size, "num_classes": num_classes} | self.architecture
        if self.window_features.enabled:
            if input_size != len(FEATURE_COLUMNS):
                raise ValueError("Window-local heart rate requires the original six-feature input")
            arguments["window_features"] = dict(self.window_features.values)
        return arguments


    def model_features(self, raw_features: list[str]) -> list[str]:
        """Describe the effective backbone channels while preserving the external raw schema.

        Args:
            raw_features (list[str]): Ordered original prepared feature names.
        """
        return self.window_features.model_features(raw_features)


    def build(
        self, input_size: int, num_classes: int,
        window_feature_state: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Construct a classifier with optional fitted window-local augmentation.

        Args:
            input_size (int): Number of original prepared features per epoch.
            num_classes (int): Number of output sleep-stage classes.
            window_feature_state (dict[str, Any] | None): Training-fitted transform normalization.
        """
        arguments = self.checkpoint_config(input_size, num_classes)
        if not self.window_features.enabled:
            if window_feature_state is not None:
                raise ValueError("Window feature state requires an enabled window feature recipe")
            return self._construct_backbone(arguments)

        if window_feature_state is None:
            raise ValueError("Window features require training-fitted normalization state")
        from sleep_classifier.window_feature_model import WindowFeatureClassifier
        from sleep_classifier.window_heart_rate import WindowLocalHeartRate

        transform = WindowLocalHeartRate(self.window_features.values, window_feature_state)
        arguments.pop("window_features")
        arguments["input_size"] = len(self.model_features(list(FEATURE_COLUMNS)))
        backbone = self._construct_backbone(arguments)
        return WindowFeatureClassifier(transform, backbone)


    def _construct_backbone(self, arguments: dict[str, Any]) -> nn.Module:
        """Build the selected architecture with explicit effective input dimensions.

        Args:
            arguments (dict[str, Any]): Validated constructor settings for the backbone.
        """
        if self.model_type == "gru":
            return CausalGRUClassifier(**arguments)
        if self.model_type == "patch_transformer":
            from sleep_classifier.patch_transformer_model import CausalPatchTransformerClassifier

            return CausalPatchTransformerClassifier(**arguments)

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
        window_features = arguments.pop("window_features", None)

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
        elif model_type == "patch_transformer":
            if "patch_size" not in arguments:
                raise ValueError("Patch Transformer checkpoints require patch_size")
            settings = {
                "model_type": model_type,
                "context_epochs": checkpoint["context_epochs"],
                "patch_size": arguments.pop("patch_size"),
                "transformer": arguments,
            }
        else:
            raise ValueError("Unsupported checkpoint model_type")

        if window_features is not None:
            settings["window_features"] = window_features
            settings["context_epochs"] = checkpoint["context_epochs"]
        factory = cls(settings)
        state = checkpoint.get("window_feature_state")
        if factory.window_features.enabled:
            cls._validate_stored_window_state(checkpoint, state)
        model = factory.build(input_size, num_classes, window_feature_state=state)
        model.load_state_dict(checkpoint["state_dict"])
        if factory.window_features.enabled:
            cls._validate_loaded_window_state(model, state)

        return model.eval()


    @staticmethod
    def _validate_stored_window_state(
        checkpoint: dict[str, Any], state: dict[str, Any] | None,
    ) -> None:
        """Require consistent saved raw and derived-feature normalization.

        Args:
            checkpoint (dict[str, Any]): Checkpoint metadata and model weights.
            state (dict[str, Any] | None): Stored training-only delta normalization.
        """
        if state is None:
            raise ValueError("Window-feature checkpoints require fitted normalization state")
        WindowFeatureSettings.validate_state(state)
        if "scaler_scale" in checkpoint:
            scales = checkpoint["scaler_scale"]
            index = FEATURE_COLUMNS.index("heart_rate_mean")
            if (not isinstance(scales, list) or len(scales) != len(FEATURE_COLUMNS)
                    or not math.isclose(float(scales[index]), float(state["heart_rate_scale"]),
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("Stored heart-rate scales disagree")
        expected = WindowFeatureSettings(
            checkpoint["model_config"]["window_features"]
        ).model_features(list(FEATURE_COLUMNS))
        if "model_features" in checkpoint and checkpoint["model_features"] != expected:
            raise ValueError("Stored effective model features differ from the transform recipe")


    @staticmethod
    def _validate_loaded_window_state(model: nn.Module, state: dict[str, Any]) -> None:
        """Reject loaded buffers that contradict the declared fitted transform state.

        Args:
            model (nn.Module): Restored classifier with its window-local transform.
            state (dict[str, Any]): Validated normalization metadata stored in the checkpoint.
        """
        for name in ("heart_rate_scale", "delta_mean", "delta_scale"):
            actual = getattr(model.transform, name)
            expected = actual.new_tensor(state[name])
            if not actual.equal(expected):
                raise ValueError("Stored window normalization buffers differ from their metadata")
