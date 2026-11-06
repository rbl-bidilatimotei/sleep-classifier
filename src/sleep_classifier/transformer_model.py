import math

import torch
from torch import nn


class CausalTransformerClassifier(nn.Module):
    """Classify the final valid epoch with causal self-attention."""

    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        feedforward_dim: int = 32,
        dropout: float = 0.1,
        max_context: int = 10,
        num_classes: int = 4,
    ) -> None:
        """Build a small Transformer with learned positions and a linear head.

        Args:
            input_size (int): Number of sensor features per epoch.
            d_model (int): Width of each attention representation.
            nhead (int): Number of attention heads dividing d_model.
            num_layers (int): Number of causal encoder layers.
            feedforward_dim (int): Width of each feedforward hidden layer.
            dropout (float): Residual and feedforward dropout probability during training.
            max_context (int): Maximum number of completed epochs per sequence.
            num_classes (int): Number of output stages.
        """
        super().__init__()
        self._validate_settings(
            input_size, d_model, nhead, num_layers, feedforward_dim,
            dropout, max_context, num_classes,
        )
        self.input_size = input_size
        self.max_context = max_context
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Embedding(max_context, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # MPS attention kernels do not support attention-weight dropout
        encoder_layer.self_attn.dropout = 0.0
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self._initialize_encoder_layers()


    @staticmethod
    def _validate_settings(
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        max_context: int,
        num_classes: int,
    ) -> None:
        """Reject invalid architecture dimensions and dropout probabilities.

        Args:
            input_size (int): Number of sensor features per epoch.
            d_model (int): Attention representation width.
            nhead (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            feedforward_dim (int): Feedforward hidden width.
            dropout (float): Training dropout probability.
            max_context (int): Maximum sequence length.
            num_classes (int): Number of output stages.
        """
        dimensions = {
            "input_size": input_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "feedforward_dim": feedforward_dim,
            "max_context": max_context,
            "num_classes": num_classes,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")


    def _initialize_encoder_layers(self) -> None:
        """Initialize cloned encoder matrices independently across layers."""
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)


    def _validate_inputs(self, features: torch.Tensor, lengths: torch.Tensor) -> None:
        """Check batch shapes and valid sequence lengths before attention.

        Args:
            features (torch.Tensor): Floating features with batch, epoch, and feature axes.
            lengths (torch.Tensor): One integer length per sequence.
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


    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return logits for the final valid epoch without attending to future epochs.

        Args:
            features (torch.Tensor): Right-padded features with batch, epoch, and feature axes.
            lengths (torch.Tensor): Valid epoch counts with shape (batch,).

        Returns:
            torch.Tensor: Stage logits with shape (batch, num_classes).
        """
        self._validate_inputs(features, lengths)
        lengths = lengths.to(features.device, dtype=torch.long)
        positions = torch.arange(features.shape[1], device=features.device)
        hidden = self.input_projection(features) + self.position_embedding(positions)

        # True entries exclude future epochs and padded keys from attention
        future_mask = positions.unsqueeze(0) > positions.unsqueeze(1)
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.encoder(
            hidden, mask=future_mask, src_key_padding_mask=padding_mask,
        )
        final = encoded[torch.arange(features.shape[0], device=features.device), lengths - 1]

        return self.classifier(final)
