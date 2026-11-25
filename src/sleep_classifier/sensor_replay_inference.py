import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.bundle_loader import BundlePredictor
from sleep_classifier.sensor_replay_inputs import RAW_CHANNELS
from sleep_classifier.sensor_replay_report import SensorReplayReport
from sleep_classifier.streaming_inference import StreamingEpochPredictor

LOGGER = logging.getLogger(__name__)


class FrozenSensorStream:
    """Replay chronological summaries with fixed weights and saved preprocessing."""

    def __init__(self, checkpoint: dict[str, Any], digest: str, device: str) -> None:
        """Restore one research checkpoint with an empty, bounded inference history.

        Args:
            checkpoint (dict[str, Any]): Verified inference fields from the frozen research fold.
            digest (str): Original checkpoint SHA-256 from the passed chronological audit.
            device (str): Explicit validated backend without fallback.
        """
        self.predictor = StreamingEpochPredictor.from_checkpoint(checkpoint, digest, device)
        if tuple(self.predictor.feature_columns) != RAW_CHANNELS:
            raise ValueError("Frozen replay requires the working seven-channel recipe")
        self.guard = BundlePredictor(self.predictor)
        self.before = self.guard.integrity()


    def predict(self, frame: pd.DataFrame, label: str) -> pd.DataFrame:
        """Predict only selected raw channels and preserve one stream's sensor reset metadata.

        Args:
            frame (pd.DataFrame): Chronological held-out summaries and explicit segment identifiers.
            label (str): Fold and policy name for progress logging.
        """
        columns = ["participant_id", "time", "segment_id", *RAW_CHANNELS]
        projected = frame.loc[:, columns]
        self.predictor.reset()
        records = []
        for index, row in enumerate(projected.to_dict("records"), start=1):
            response = self.predictor.predict_epoch(
                str(row["participant_id"]), int(row["segment_id"]), float(row["time"]),
                {name: float(row[name]) for name in RAW_CHANNELS},
            )
            probabilities = response.pop("probabilities")
            response.update({f"probability_{stage}": value
                             for stage, value in enumerate(probabilities)})
            records.append(response)
            if index % 1000 == 0 or index == len(frame):
                LOGGER.info("Frozen replay %s: %d/%d epochs", label, index, len(frame))
        return pd.DataFrame(records)


    def integrity(self) -> dict[str, Any]:
        """Verify unchanged weights, evaluation mode, and absent gradients after both replays."""
        after = self.guard.integrity()
        result = {"weights_unchanged": after["state_sha256"] == self.before["state_sha256"],
                  "evaluation_mode": after["evaluation_mode"],
                  "gradients_created": after["gradients_created"],
                  "state_sha256": after["state_sha256"], "training_performed": False}
        if (not result["weights_unchanged"] or not result["evaluation_mode"]
                or result["gradients_created"]):
            raise ValueError("Frozen replay changed model state or created gradients")
        return result


    @staticmethod
    def control_parity(
        predictions: pd.DataFrame, saved: pd.DataFrame, absolute_tolerance: float,
    ) -> dict[str, Any]:
        """Require the legacy-reset stream to reproduce saved identities and probabilities.

        Args:
            predictions (pd.DataFrame): Fresh legacy-reset stream predictions without target labels.
            saved (pd.DataFrame): Verified original chronological control predictions.
            absolute_tolerance (float): Frozen absolute probability agreement threshold.
        """
        keys = ["participant_id", "time"]
        if (isinstance(absolute_tolerance, bool)
                or not isinstance(absolute_tolerance, (float, int))
                or not math.isfinite(absolute_tolerance) or not 0 < absolute_tolerance <= 1e-4):
            raise ValueError("Control tolerance must be finite, positive, and at most 1e-4")
        SensorReplayReport.build(predictions, saved)
        if not np.array_equal(predictions[keys].to_numpy(), saved[keys].to_numpy()):
            raise ValueError("Control replay identities differ from the saved chronological audit")
        columns = [f"probability_{stage}" for stage in range(4)]
        difference = float(np.abs(predictions[columns].to_numpy()-saved[columns].to_numpy()).max())
        disagreements = int(predictions.prediction.ne(saved.prediction).sum())
        histories = bool(np.array_equal(predictions.history_epochs, saved.history_epochs))
        reasons = np.array_equal(predictions.reset_reason.fillna("").to_numpy(dtype=str),
                                 saved.reset_reason.fillna("").to_numpy(dtype=str))
        if difference > absolute_tolerance or disagreements or not histories or not reasons:
            raise ValueError("Legacy-reset control differs from the saved chronological audit")
        return {"status": "passed", "epochs": len(saved),
                "maximum_probability_difference": difference, "class_disagreements": disagreements,
                "history_lengths_match": histories, "reset_reasons_match": bool(reasons)}
