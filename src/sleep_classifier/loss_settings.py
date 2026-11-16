import math
from typing import Any


class LossSettings:
    """Validate training-label smoothing while keeping evaluation targets unchanged."""

    def __init__(self, values: dict[str, Any] | None) -> None:
        """Resolve the supported cross-entropy objective and its smoothing amount.

        Args:
            values (dict[str, Any] | None): Optional training loss configuration.
        """
        if values is not None and not isinstance(values, dict):
            raise ValueError("loss must be a mapping or null")
        supplied = dict(values or {})
        if set(supplied) - {"name", "label_smoothing"}:
            raise ValueError("Unknown loss settings")
        if supplied.get("name", "cross_entropy") != "cross_entropy":
            raise ValueError("loss name must be cross_entropy")

        smoothing = supplied.get("label_smoothing", 0.0)
        if (isinstance(smoothing, bool) or not isinstance(smoothing, (int, float))
                or not math.isfinite(smoothing) or not 0 <= smoothing <= 1):
            raise ValueError("loss label_smoothing must be finite and in [0, 1]")
        self.values = {"name": "cross_entropy", "label_smoothing": float(smoothing)}


    def provenance(self) -> dict[str, dict[str, str | float]]:
        """Describe the training objective and independent hard-label evaluation loss."""
        return {
            "training": dict(self.values),
            "evaluation": {"name": "cross_entropy", "label_smoothing": 0.0},
        }
