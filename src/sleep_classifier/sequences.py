from operator import itemgetter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sleep_classifier.features import FEATURE_COLUMNS


class CausalSequenceDataset(Dataset):
    """Create causal, participant-local windows from eligible scored epochs."""

    def __init__(self, frame: pd.DataFrame, scaler, context_epochs: int = 10) -> None:
        """Validate rows and scale each retained history window."""
        if (
            isinstance(context_epochs, bool)
            or not isinstance(context_epochs, (int, np.integer))
            or context_epochs <= 0
        ):
            raise ValueError("context_epochs must be a positive integer")
        self.context_epochs = context_epochs
        self._validate(frame)
        ordered = frame.sort_values(["participant_id", "time"], kind="stable").reset_index(
            drop=True
        )
        scaled_features = scaler.transform(ordered.loc[:, FEATURE_COLUMNS])
        self.samples = self._make_samples(ordered, scaled_features)
        self.samples.sort(key=itemgetter("participant_id", "time"))

    def _validate(self, frame: pd.DataFrame) -> None:
        required = set(FEATURE_COLUMNS) | {
            "participant_id",
            "segment_id",
            "time",
            "target",
            "eligible",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Frame is missing columns: {sorted(missing)}")
        features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        times = frame["time"].to_numpy(dtype=float)
        targets = frame["target"].to_numpy(dtype=float)
        segments = frame["segment_id"].to_numpy(dtype=float)
        if not np.isfinite(features).all() or not np.isfinite(times).all():
            raise ValueError("Features and timestamps must be finite")
        if not np.isfinite(targets).all() or not np.isin(targets, [0, 1, 2, 3]).all():
            raise ValueError("Targets must be integers from 0 through 3")
        if (
            not np.isfinite(segments).all()
            or (segments < 0).any()
            or not np.equal(segments, np.floor(segments)).all()
        ):
            raise ValueError("segment_id must contain non-negative integers")
        missing_participant = frame["participant_id"].isna().any()
        empty_participant = (frame["participant_id"].astype(str) == "").any()
        if missing_participant or empty_participant:
            raise ValueError("participant_id must be present")
        if not frame["eligible"].eq(True).all():
            raise ValueError("All rows must be eligible")
        if frame.duplicated(["participant_id", "time"]).any():
            raise ValueError("Timestamps must be unique within each participant")
        self._validate_contiguous_segments(frame)

    def _validate_contiguous_segments(self, frame: pd.DataFrame) -> None:
        for _, group in frame.groupby(["participant_id", "segment_id"], sort=False):
            times = np.sort(group["time"].to_numpy(dtype=float))
            if len(times) > 1 and not np.allclose(np.diff(times), 30.0, rtol=0, atol=1e-8):
                raise ValueError(
                    "Timestamps within each segment must be contiguous 30-second epochs"
                )

    def _make_samples(self, frame: pd.DataFrame, scaled_features: np.ndarray) -> list[dict]:
        samples = []
        for indexes in frame.groupby(["participant_id", "segment_id"], sort=False).indices.values():
            group = frame.iloc[indexes].reset_index(drop=True)
            group_features = scaled_features[indexes]
            for end in range(len(group)):
                start = max(0, end - self.context_epochs + 1)
                samples.append(
                    {
                        "features": torch.tensor(
                            group_features[start : end + 1], dtype=torch.float32
                        ),
                        "target": torch.tensor(int(group.iloc[end]["target"]), dtype=torch.long),
                        "participant_id": str(group.iloc[end]["participant_id"]),
                        "time": float(group.iloc[end]["time"]),
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class SequenceBatchCollator:
    """Right-pad variable-length sequence examples into a batch."""

    def __call__(self, samples: list[dict]) -> dict:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        lengths = torch.tensor([len(sample["features"]) for sample in samples], dtype=torch.int64)
        feature_size = samples[0]["features"].shape[1]
        padded = torch.zeros((len(samples), int(lengths.max()), feature_size), dtype=torch.float32)
        for index, sample in enumerate(samples):
            padded[index, : lengths[index]] = sample["features"]
        return {
            "features": padded,
            "lengths": lengths,
            "targets": torch.stack([sample["target"] for sample in samples]),
            "participant_ids": [sample["participant_id"] for sample in samples],
            "times": torch.tensor([sample["time"] for sample in samples], dtype=torch.float64),
        }
