from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from sleep_classifier.baseline import PreparedFeatureDataset
from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.sequences import CausalSequenceDataset, SequenceBatchCollator


class CheckpointReplayAudit:
    """Replay a selected checkpoint without fitting preprocessing or updating weights."""

    def __init__(self, fold_dir: Path, device: str = "cpu") -> None:
        """Store a prepared fold and an explicit inference device.

        Args:
            fold_dir (Path): Fold directory containing features and results.
            device (str): CPU, MPS, or CUDA device used for read-only inference.
        """
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("Replay requires an explicit cpu, mps, or cuda device")
        self.fold_dir = fold_dir.resolve()
        self.device = torch.device(device)


    def _load(self) -> tuple[nn.Module, CausalSequenceDataset, pd.DataFrame]:
        """Restore stored scaling and the selected model on validated fold observations."""
        checkpoint = torch.load(
            self.fold_dir / "results/model_checkpoint.pt", map_location="cpu", weights_only=True,
        )
        if checkpoint["features"] != list(FEATURE_COLUMNS):
            raise ValueError("Checkpoint features differ from the prepared feature schema")
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=float)
        scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=float)
        scaler.n_features_in_ = len(FEATURE_COLUMNS)
        scaler.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        if (scaler.mean_.shape != (len(FEATURE_COLUMNS),)
                or scaler.scale_.shape != scaler.mean_.shape
                or not np.isfinite(scaler.mean_).all()
                or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any()):
            raise ValueError("Stored scaler metadata is invalid")

        loaded = PreparedFeatureDataset(self.fold_dir / "features").load()
        frame = loaded["partitions"]["validation"].sort_values(
            ["participant_id", "time"], kind="stable",
        ).reset_index(drop=True)
        data = CausalSequenceDataset(frame, scaler, checkpoint["context_epochs"])
        model = ModelFactory.from_checkpoint(checkpoint).to(self.device).eval()

        return model, data, frame


    def _predict(self, model: nn.Module, data: CausalSequenceDataset) -> np.ndarray:
        """Evaluate every valid target in order with stochastic training disabled.

        Args:
            model (nn.Module): Restored evaluation-mode classifier.
            data (CausalSequenceDataset): Validation windows using stored scaling.
        """
        loader = DataLoader(data, batch_size=64, shuffle=False, collate_fn=SequenceBatchCollator())
        probabilities: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch["features"].to(self.device), batch["lengths"])
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.concatenate(probabilities)


    def build(self) -> dict[str, Any]:
        """Compare two inference passes with each other and the saved predictions."""
        model, data, frame = self._load()
        before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        first = self._predict(model, data)
        second = self._predict(model, data)
        saved = pd.read_csv(
            self.fold_dir / "results/validation_predictions.csv", dtype={"participant_id": str},
        ).sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)
        keys = ["participant_id", "time", "target"]
        if (len(saved) != len(frame)
                or not np.array_equal(saved.participant_id.to_numpy(dtype=str),
                                      frame.participant_id.to_numpy(dtype=str))
                or not np.array_equal(saved[keys[1:]].to_numpy(), frame[keys[1:]].to_numpy())):
            raise ValueError("Saved prediction targets differ from replay inputs")
        expected = saved[[f"probability_{stage}" for stage in range(4)]].to_numpy()
        targets = frame.target.to_numpy(dtype=int)
        unchanged = all(torch.equal(before[name], value.detach().cpu())
                        for name, value in model.state_dict().items())

        return {
            "fold": self.fold_dir.name, "device": str(self.device), "epochs": len(frame),
            "evaluation_mode": not model.training,
            "repeat_max_abs_probability_difference": float(np.abs(first - second).max()),
            "saved_max_abs_probability_difference": float(np.abs(first - expected).max()),
            "matches_saved_atol_2e_6": bool(np.allclose(first, expected, atol=2e-6, rtol=0)),
            "weights_unchanged": unchanged,
            "gradients_created": any(
                parameter.grad is not None for parameter in model.parameters()
            ),
            "validation_nll": float(-np.log(np.maximum(
                first[np.arange(len(targets)), targets], 1e-12,
            )).mean()),
        }
