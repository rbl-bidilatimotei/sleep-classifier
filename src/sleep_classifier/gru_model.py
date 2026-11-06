import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CausalGRUClassifier(nn.Module):
    """Classify the final epoch in each causal feature sequence."""

    def __init__(self, input_size: int = 6, hidden_size: int = 32, num_classes: int = 4) -> None:
        """
        Initialize the CausalGRUClassifier.

        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of hidden units in the GRU.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return class logits while ignoring right-padding.

        Args:
            features (torch.Tensor): Input feature sequences of shape (batch_size, seq_len,
                input_size).
            lengths (torch.Tensor): Lengths of each sequence in the batch.

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)

        return self.classifier(hidden[-1])
