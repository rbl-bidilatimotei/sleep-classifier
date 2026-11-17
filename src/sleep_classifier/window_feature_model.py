from torch import Tensor, nn


class WindowFeatureClassifier(nn.Module):
    """Apply deterministic window-local features before the configured sequence classifier."""

    def __init__(self, transform: nn.Module, backbone: nn.Module) -> None:
        """Store the fitted deterministic transform and the trainable classifier.

        Args:
            transform (nn.Module): Window-local augmentation with stored training-only scaling.
            backbone (nn.Module): Classifier accepting the augmented feature channels.
        """
        super().__init__()
        self.transform = transform
        self.backbone = backbone


    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        """Predict from the original feature windows without accessing observations outside them.

        Args:
            features (Tensor): Right-padded batch of normalized original feature windows.
            lengths (Tensor): Number of valid epochs in each window.
        """
        return self.backbone(self.transform(features, lengths), lengths)
