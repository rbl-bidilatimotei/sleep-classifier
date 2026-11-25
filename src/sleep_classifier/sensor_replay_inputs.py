import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.motion_feature_input import MOTION_FEATURE, MotionFeatureInput

RAW_CHANNELS = (*FEATURE_COLUMNS, MOTION_FEATURE)
SENSOR_COLUMNS = ("participant_id", "time", "segment_id", "eligible", *RAW_CHANNELS)


class SensorReplayInputs:
    """Project sensor-only candidates and reconstruct a separate labeled research control."""

    @staticmethod
    def candidate(
        rows: pd.DataFrame, participants: list[str], imputation_mean: float,
    ) -> pd.DataFrame:
        """Select sensor eligibility and apply only the saved training motion replacement.

        Args:
            rows (pd.DataFrame): Verified preparation audit rows, including unused diagnostics.
            participants (list[str]): This frozen fold's held-out development identifiers.
            imputation_mean (float): Saved training-only motion imputation mean.
        """
        if (isinstance(imputation_mean, bool) or not isinstance(imputation_mean, (float, int))
                or not math.isfinite(imputation_mean)):
            raise ValueError("Replay requires a finite saved motion imputation mean")
        if not rows.columns.is_unique or not set(SENSOR_COLUMNS) <= set(rows):
            raise ValueError("Sensor replay rows lack required channels")
        frame = rows.loc[rows.participant_id.isin(participants), list(SENSOR_COLUMNS)].copy()
        if not all(isinstance(value, (bool, np.bool_)) for value in frame.eligible.dropna()):
            raise ValueError("Sensor eligibility requires booleans or absent reference-only rows")
        frame = frame.loc[frame.eligible.eq(True)].drop(columns="eligible")
        if set(frame.participant_id) != set(participants):
            raise ValueError("Every held-out participant requires sensor-eligible epochs")
        frame[MOTION_FEATURE] = frame[MOTION_FEATURE].fillna(imputation_mean)
        values = frame[["time", "segment_id", *RAW_CHANNELS]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or frame.duplicated(["participant_id", "time"]).any():
            raise ValueError("Replay identities and imputed values must be finite and unique")
        segments = frame.segment_id.to_numpy(float)
        if (segments < 0).any() or (segments % 1 != 0).any():
            raise ValueError("Sensor segment identifiers must be nonnegative integers")
        frame["segment_id"] = frame.segment_id.astype(int)
        return frame.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)


    @staticmethod
    def reference(
        fold_dir: Path, participants: list[str], checkpoint: dict[str, Any],
    ) -> pd.DataFrame:
        """Restore saved research eligibility and motion values without fitting transforms.

        Args:
            fold_dir (Path): Verified frozen fold containing original development preparation.
            participants (list[str]): Held-out participant identifiers from the frozen fold plan.
            checkpoint (dict[str, Any]): Verified saved weights and fitted preprocessing metadata.
        """
        frames = []
        for person in participants:
            path = fold_dir / "features/validation" / f"{person}.csv"
            frame = pd.read_csv(path)
            frame = frame.loc[frame.eligible & frame.target.ge(0)].copy()
            frame["participant_id"] = person
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        recipe_path = fold_dir / "gru.yaml"
        recipe = yaml.safe_load(recipe_path.read_text())
        frame = MotionFeatureInput(recipe["motion_features"], recipe_path).transform(
            frame, checkpoint["motion_features"],
        )
        return frame.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)
