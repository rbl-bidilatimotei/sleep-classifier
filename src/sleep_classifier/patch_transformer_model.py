import math

import torch
from torch import nn

from sleep_classifier.transformer_model import CausalTransformerClassifier


class CausalPatchTransformerClassifier(nn.Module):
    """Classify completed history using jointly embedded chronological feature patches."""

    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        feedforward_dim: int = 32,
        dropout: float = 0.1,
        max_context: int = 30,
        num_classes: int = 4,
        patch_size: int = 3,
    ) -> None:
        """Compose a causal encoder over nonoverlapping patches of completed epochs.

        Args:
            input_size (int): Number of normalized sensor features per raw epoch.
            d_model (int): Width of each patch representation.
            nhead (int): Number of attention heads dividing d_model.
            num_layers (int): Number of causal encoder layers.
            feedforward_dim (int): Width of each feedforward hidden layer.
            dropout (float): Residual and feedforward training dropout probability.
            max_context (int): Maximum raw epoch count, before patching.
            num_classes (int): Number of output sleep stages.
            patch_size (int): Consecutive raw epochs jointly embedded in each token.
        """
        super().__init__()
        for name, value in (("input_size", input_size), ("max_context", max_context),
                            ("patch_size", patch_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if patch_size > max_context:
            raise ValueError("patch_size cannot exceed max_context")

        self.input_size = input_size
        self.max_context = max_context
        self.patch_size = patch_size
        self.backbone = CausalTransformerClassifier(
            input_size=input_size * patch_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, feedforward_dim=feedforward_dim, dropout=dropout,
            max_context=math.ceil(max_context / patch_size), num_classes=num_classes,
        )


    def _validate_inputs(self, features: torch.Tensor, lengths: torch.Tensor) -> None:
        """Require raw right-padded histories and one valid length for each sample.

        Args:
            features (torch.Tensor): Floating tensor with batch, raw epoch, and feature axes.
            lengths (torch.Tensor): Integer valid raw epoch counts for the batch.
        """
        if features.ndim != 3 or features.shape[2] != self.input_size:
            raise ValueError("features must have shape (batch, epochs, input_size)")
        if features.shape[0] == 0 or not 1 <= features.shape[1] <= self.max_context:
            raise ValueError("features require a nonempty batch and supported context length")
        if not features.is_floating_point():
            raise ValueError("features must have a floating dtype")
        if lengths.ndim != 1 or len(lengths) != features.shape[0]:
            raise ValueError("lengths must contain one value per sequence")
        if lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("lengths must contain integers")
        if bool(((lengths < 1) | (lengths > features.shape[1])).any()):
            raise ValueError("lengths must be between one and the padded sequence length")


    def patch_features(
        self, features: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-align valid histories and flatten ordered features within each patch.

        Args:
            features (torch.Tensor): Right-padded normalized raw features.
            lengths (torch.Tensor): Valid raw epoch counts, excluding right padding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Right-padded patch features and valid patch counts.
        """
        self._validate_inputs(features, lengths)
        lengths = lengths.to(features.device, dtype=torch.long)
        patch_lengths = (lengths + self.patch_size - 1) // self.patch_size
        token_count = math.ceil(features.shape[1] / self.patch_size)
        front_padding = patch_lengths * self.patch_size - lengths
        positions = torch.arange(token_count * self.patch_size, device=features.device)
        raw_indices = positions.unsqueeze(0) - front_padding.unsqueeze(1)
        valid = (raw_indices >= 0) & (raw_indices < lengths.unsqueeze(1))
        indices = raw_indices.clamp(min=0, max=features.shape[1] - 1)
        gathered = features.gather(1, indices.unsqueeze(2).expand(-1, -1, self.input_size))

        # Mask before projection so arbitrary right padding, including NaN, has no effect
        gathered = gathered.masked_fill(~valid.unsqueeze(2), 0)
        patches = gathered.reshape(
            features.shape[0], token_count, self.patch_size * self.input_size,
        )
        return patches, patch_lengths


    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return stage logits for the final completed epoch using only its valid history.

        Args:
            features (torch.Tensor): Right-padded raw normalized feature histories.
            lengths (torch.Tensor): Valid raw epoch counts for each sample.
        """
        patches, patch_lengths = self.patch_features(features, lengths)
        return self.backbone(patches, patch_lengths)
