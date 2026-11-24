import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.checkpoint_audit import CheckpointReplayAudit

LOGGER = logging.getLogger(__name__)


class EvaluationBundleParity:
    """Verify packaged predictions against every saved validation target before publication."""

    def __init__(self, fold_dir: Path, bundle: EvaluationBundle) -> None:
        """Bind original development inputs and the already verified self-contained bundle.

        Args:
            fold_dir (Path): Selected fold with prepared validation summaries.
            bundle (EvaluationBundle): Staged sanitized bundle loaded without source caches.
        """
        self.fold_dir = fold_dir
        self.bundle = bundle


    def build(self) -> dict[str, Any]:
        """Replay the complete selected validation fold without fitting or changing weights."""
        _, _, frame = CheckpointReplayAudit(self.fold_dir, "mps")._load()
        saved = pd.read_csv(self.fold_dir / "results/validation_predictions.csv",
                            dtype={"participant_id": str})
        saved = saved.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)
        keys = ["participant_id", "time", "target"]
        if (len(frame) == 0
                or not np.array_equal(frame[keys].to_numpy(), saved[keys].to_numpy())):
            raise ValueError("Export parity requires the original validation target identities")
        results = self._predict(frame)
        probabilities = np.asarray([row["probabilities"] for row in results])
        expected = saved[[f"probability_{index}" for index in range(4)]].to_numpy()
        if (not np.isfinite(expected).all() or (expected < 0).any()
                or not np.allclose(expected.sum(1), 1, atol=1e-6, rtol=0)
                or not np.array_equal(expected.argmax(1), saved.prediction.to_numpy())):
            raise ValueError("Saved validation probabilities must be finite and consistent")
        difference = float(np.abs(probabilities - expected).max())
        disagreements = int((probabilities.argmax(1) != saved.prediction.to_numpy()).sum())
        lengths = np.minimum(frame.groupby(["participant_id", "segment_id"]).cumcount() + 1, 30)
        reasons = self._reset_reasons(frame)
        if (not np.isfinite(probabilities).all() or difference > 2e-6 or disagreements
                or [row["reset_reason"] for row in results] != reasons
                or not np.array_equal([row["history_epochs"] for row in results], lengths)):
            raise ValueError(f"Bundle parity failed: max_difference={difference}, "
                             f"class_disagreements={disagreements}")
        return {"status": "passed", "validation_epochs": len(frame),
                "maximum_probability_difference": difference, "prediction_disagreements": 0,
                "history_lengths_match": True, "reset_reasons_match": True,
                "history_start_epochs": int((lengths == 1).sum()),
                "full_context_epochs": int((lengths == 30).sum()),
                "weights_unchanged": True, "gradients_created": False,
                "training_performed": False, "test_observations_read": False}


    @staticmethod
    def _reset_reasons(frame: pd.DataFrame) -> list[str | None]:
        """Derive expected resets from chronological metadata without using target labels.

        Args:
            frame (pd.DataFrame): Eligible validation identities and original segment boundaries.
        """
        reasons: list[str | None] = []
        previous: tuple[str, int, float] | None = None
        for current in frame[["participant_id", "segment_id", "time"]].itertuples(
            index=False, name=None,
        ):
            reason = None
            if previous is not None:
                if current[0] != previous[0]:
                    reason = "participant_change"
                elif current[1] != previous[1]:
                    reason = "segment_change"
                elif not np.isclose(current[2] - previous[2], 30, atol=1e-8, rtol=0):
                    reason = "time_gap"
            reasons.append(reason)
            previous = current
        return reasons


    def _predict(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Supply one label-free completed epoch at a time to the packaged runtime.

        Args:
            frame (pd.DataFrame): Ordered eligible development validation summaries.
        """
        predictor = self.bundle.predictor()
        before = predictor.integrity()
        columns = self.bundle.manifest["model"]["raw_features"]
        results = []
        for index, row in enumerate(frame.to_dict("records"), start=1):
            results.append(predictor.predict({
                "participant_id": str(row["participant_id"]), "segment_id": int(row["segment_id"]),
                "time": float(row["time"]),
                "features": {name: float(row[name]) for name in columns},
            }))
            if index % 1000 == 0:
                LOGGER.info("Packaged parity %s: %d/%d epochs",
                            self.fold_dir.name, index, len(frame))
        after = predictor.integrity()
        if before != after or not after["evaluation_mode"] or after["gradients_created"]:
            raise ValueError("Packaged replay changed model state or created gradients")
        return results
