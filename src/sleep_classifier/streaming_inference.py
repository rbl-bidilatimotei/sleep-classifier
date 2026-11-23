import hashlib
import math
from collections import deque
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.motion_feature_input import MOTION_FEATURE


class StreamingEpochPredictor:
    """Predict from one stream of completed, causally prepared epochs."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        """Restore the model and its fixed preprocessing from a checkpoint.

        Args:
            checkpoint_path (Path): Saved sequence classifier checkpoint.
            device (str): Explicit inference device, either cpu, mps, or cuda.
        """
        self.device = self._resolve_device(device)
        with checkpoint_path.open("rb") as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)

        self.feature_columns = self._checkpoint_features(checkpoint)
        context = checkpoint.get("context_epochs")
        if type(context) is not int or context <= 0:
            raise ValueError("Checkpoint context_epochs must be a positive integer")
        self.context_epochs = context
        self._mean = self._scaler_vector(checkpoint, "scaler_mean")
        self._scale = self._scaler_vector(checkpoint, "scaler_scale")
        if (self._scale <= 0).any():
            raise ValueError("Checkpoint scaler_scale values must be positive")

        self.model = ModelFactory.from_checkpoint(checkpoint).to(self.device).eval()
        self._history: deque[np.ndarray] = deque(maxlen=self.context_epochs)
        self._previous: tuple[str, int, float] | None = None


    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Validate the requested compute device without falling back.

        Args:
            device (str): Explicit compute device name.
        """
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("Streaming inference requires an explicit cpu, mps, or cuda device")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("The requested mps device is unavailable")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("The requested cuda device is unavailable")
        return torch.device(device)


    @staticmethod
    def _checkpoint_features(checkpoint: dict[str, Any]) -> tuple[str, ...]:
        """Check the raw feature order and classifier dimensions.

        Args:
            checkpoint (dict[str, Any]): Saved weights and metadata.
        """
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_config"), dict):
            raise ValueError("Checkpoint must contain model_config metadata")
        constructor = checkpoint["model_config"]
        columns = FEATURE_COLUMNS
        if constructor.get("motion_features") is not None:
            columns = (*columns, MOTION_FEATURE)
        if checkpoint.get("features") != list(columns):
            raise ValueError("Checkpoint features differ from the declared feature schema")
        if constructor.get("input_size") != len(columns) or constructor.get("num_classes") != 4:
            raise ValueError("Checkpoint dimensions must match the feature schema and four classes")
        return columns


    def _scaler_vector(self, checkpoint: dict[str, Any], name: str) -> np.ndarray:
        """Read a finite scaler vector in checkpoint feature order.

        Args:
            checkpoint (dict[str, Any]): Saved weights and metadata.
            name (str): Name of the scaler vector to restore.
        """
        values = checkpoint.get(name)
        if not isinstance(values, list) or len(values) != len(self.feature_columns):
            raise ValueError(f"Checkpoint {name} must contain one value per feature")
        return np.asarray([self._finite_number(value, name) for value in values], dtype=np.float64)


    @staticmethod
    def _finite_number(value: float, name: str) -> float:
        """Convert a finite real number for scaling or timestamp comparison.

        Args:
            value (float): Raw numeric value.
            name (str): Field name used in the validation message.
        """
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        return number


    def _standardize(self, features: dict[str, float]) -> np.ndarray:
        """Apply the saved scaler before converting inputs to model precision.

        Args:
            features (dict[str, float]): Completed epoch summaries in original units.
        """
        if not isinstance(features, dict) or features.keys() != set(self.feature_columns):
            raise ValueError("Epoch feature keys must match the checkpoint feature schema")
        values = np.asarray(
            [self._finite_number(features[name], name) for name in self.feature_columns],
            dtype=np.float64,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = ((values - self._mean) / self._scale).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled epoch features must be finite in float32")
        return scaled


    def _reset_reason(self, participant_id: str, segment_id: int, time: float) -> str | None:
        """Check the new epoch against the current history boundary.

        Args:
            participant_id (str): Participant who supplied the epoch.
            segment_id (int): Nonnegative segment identifier within the participant.
            time (float): Epoch timestamp in seconds.
        """
        if self._previous is None:
            return None
        previous_participant, previous_segment, previous_time = self._previous
        if participant_id != previous_participant:
            return "participant_change"
        if time <= previous_time:
            raise ValueError("Epoch time must strictly increase within the active participant")
        if segment_id != previous_segment:
            return "segment_change"
        if not math.isclose(time - previous_time, 30.0, rel_tol=0.0, abs_tol=1e-8):
            return "time_gap"
        return None


    def reset(self) -> None:
        """Clear retained epochs and permit a new session to start at any time."""
        self._history.clear()
        self._previous = None


    def predict_epoch(
        self, participant_id: str, segment_id: int, time: float, features: dict[str, float],
    ) -> dict[str, Any]:
        """Predict one completed epoch and retain its standardized features.

        Args:
            participant_id (str): Nonempty participant identifier.
            segment_id (int): Nonnegative segment identifier within the participant.
            time (float): Epoch timestamp in seconds.
            features (dict[str, float]): Causally prepared and imputed summaries in original units.

        Returns:
            dict[str, Any]: Epoch identity, class, probabilities, history size, and reset reason.
        """
        if not isinstance(participant_id, str) or not participant_id.strip():
            raise ValueError("participant_id must be a nonempty string")
        if isinstance(segment_id, bool) or not isinstance(segment_id, Integral) or segment_id < 0:
            raise ValueError("segment_id must be a nonnegative integer")
        time = self._finite_number(time, "time")
        reason = self._reset_reason(participant_id, segment_id, time)
        history = deque(() if reason else self._history, maxlen=self.context_epochs)
        history.append(self._standardize(features))
        prediction, probabilities = self._predict(history)

        self._history = history
        self._previous = (participant_id, segment_id, time)
        return {
            "participant_id": participant_id,
            "time": time,
            "prediction": prediction,
            "probabilities": probabilities,
            "history_epochs": len(history),
            "reset_reason": reason,
        }


    def _predict(self, history: deque[np.ndarray]) -> tuple[int, list[float]]:
        """Evaluate the full retained window in the model's inference mode.

        Args:
            history (deque[np.ndarray]): Standardized rows in time order.
        """
        features = torch.from_numpy(np.stack(history)).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(history)], dtype=torch.int64, device="cpu")
        with torch.inference_mode():
            probabilities = self.model(features, lengths).softmax(dim=1)[0].cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Model probabilities must be finite")
        return int(probabilities.argmax().item()), probabilities.tolist()
