import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.window_feature_settings import WindowFeatureSettings


class WindowLocalHeartRate(nn.Module):
    """Append standardized past-relative heart rate and unscaled availability flags."""

    def __init__(self, settings: dict[str, Any], state: dict[str, Any]) -> None:
        """Store validated training-only normalization for the deterministic feature channels.

        Args:
            settings (dict[str, Any]): Enabled heart-rate window configuration.
            state (dict[str, Any]): Normalization fitted on available training target deltas.
        """
        super().__init__()
        validated = WindowFeatureSettings(settings)
        if not validated.enabled:
            raise ValueError("Window-local heart-rate augmentation requires enabled settings")
        self.settings = validated.values
        WindowFeatureSettings.validate_state(state)
        self._normalization_state = dict(state)
        self.heart_index = FEATURE_COLUMNS.index("heart_rate_mean")
        self.register_buffer(
            "heart_rate_scale", torch.tensor(state["heart_rate_scale"], dtype=torch.float32),
        )
        self.register_buffer("delta_mean", torch.tensor(state["delta_mean"], dtype=torch.float32))
        self.register_buffer("delta_scale", torch.tensor(state["delta_scale"], dtype=torch.float32))


    @property
    def normalization_state(self) -> dict[str, Any]:
        """Return a copy of the fitted normalization metadata for checkpoint provenance."""
        return dict(self._normalization_state)


    @classmethod
    def fit(
        cls, samples: Iterable[Mapping[str, Any]], heart_rate_scale: float,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit one available last-target deviation per supplied training window.

        Args:
            samples (Iterable[Mapping[str, Any]]): Training windows containing scaled raw features.
            heart_rate_scale (float): Training-fitted scale of the raw mean heart-rate channel.
            settings (dict[str, Any]): Enabled window-local history definition.
        """
        configured = WindowFeatureSettings(settings)
        if not configured.enabled:
            raise ValueError("Fitting window-local heart rate requires enabled settings")
        if (isinstance(heart_rate_scale, bool) or not isinstance(heart_rate_scale, (int, float))
                or not math.isfinite(heart_rate_scale) or heart_rate_scale <= 0):
            raise ValueError("heart_rate_scale must be finite and positive")
        deltas: list[float] = []
        for sample in samples:
            features = sample["features"]
            if (not isinstance(features, torch.Tensor) or features.ndim != 2
                    or features.shape[1] != len(FEATURE_COLUMNS) or len(features) == 0
                    or not features.is_floating_point()
                    or not torch.isfinite(features).all()):
                raise ValueError("Training windows must contain finite raw-feature matrices")
            previous = min(len(features) - 1, configured.values["history_epochs"])
            if previous < configured.values["min_history_epochs"]:
                continue
            heart = features[:, FEATURE_COLUMNS.index("heart_rate_mean")].detach().cpu().numpy()
            heart = heart.astype(np.float64)
            deviation = heart[-1] - heart[-previous - 1:-1].mean()
            deltas.append(float(deviation * heart_rate_scale))
        if not deltas:
            raise ValueError("No training targets have sufficient window-local heart-rate history")
        scale = float(np.std(deltas, ddof=0))
        state = {
            "schema_version": 1, "heart_rate_scale": float(heart_rate_scale),
            "delta_mean": float(np.mean(deltas)), "delta_scale": scale if scale > 0 else 1.0,
            "available_training_targets": len(deltas), "scaling_fit": "training_targets_only",
        }
        WindowFeatureSettings.validate_state(state)
        return state


    def _valid_positions(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Validate a right-padded raw batch and return its valid-position mask.

        Args:
            features (torch.Tensor): Floating raw input batch with six channels.
            lengths (torch.Tensor): Number of valid epochs in each sequence.
        """
        if (features.ndim != 3 or features.shape[2] != len(FEATURE_COLUMNS)
                or features.shape[0] == 0 or features.shape[1] == 0
                or not features.is_floating_point()):
            raise ValueError("Window features require a nonempty floating B x T x 6 batch")
        if (lengths.ndim != 1 or len(lengths) != features.shape[0]
                or lengths.dtype not in (torch.uint8, torch.int8, torch.int16,
                                         torch.int32, torch.int64)
                or (lengths < 1).any() or (lengths > features.shape[1]).any()):
            raise ValueError("Window lengths must contain valid positive integer counts")
        positions = torch.arange(features.shape[1], device=features.device)
        valid = positions.unsqueeze(0) < lengths.to(features.device).unsqueeze(1)
        if not torch.isfinite(features.masked_fill(~valid.unsqueeze(2), 0)).all():
            raise ValueError("Valid raw input positions must contain finite values")
        return valid


    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Derive each channel from previous valid positions inside the supplied window only.

        Args:
            features (torch.Tensor): Training-scaled raw B x T x 6 histories.
            lengths (torch.Tensor): Valid sequence lengths before right padding.
        """
        valid = self._valid_positions(features, lengths)
        base = features.masked_fill(~valid.unsqueeze(2), 0)
        heart = base[:, :, self.heart_index]
        positions = torch.arange(features.shape[1], device=features.device)
        starts = (positions - self.settings["history_epochs"]).clamp(min=0)
        counts = positions - starts
        prefix = torch.cat([torch.zeros_like(heart[:, :1]), heart.cumsum(dim=1)], dim=1)
        past_sum = prefix[:, positions] - prefix[:, starts]
        past_mean = past_sum / counts.clamp(min=1).to(features.dtype)
        available = valid & (counts.unsqueeze(0) >= self.settings["min_history_epochs"])
        raw_delta = (heart - past_mean) * self.heart_rate_scale.to(features.dtype)
        standardized = (raw_delta - self.delta_mean.to(features.dtype)) / self.delta_scale.to(
            features.dtype
        )
        standardized = standardized.masked_fill(~available, 0)
        channels = {
            "heart_rate_delta_past": standardized,
            "heart_history_available": available.to(features.dtype),
        }
        for name in self.settings.get("masked_channels", []):
            channels[name] = channels[name].masked_fill(torch.ones_like(available), 0)
        return torch.cat([
            base, channels["heart_rate_delta_past"].unsqueeze(2),
            channels["heart_history_available"].unsqueeze(2),
        ], dim=2)
