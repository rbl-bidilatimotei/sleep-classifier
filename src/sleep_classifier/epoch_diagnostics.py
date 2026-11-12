import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


class EpochDiagnosticsSettings:
    """Validate optional evidence settings without changing the training recipe."""

    def __init__(self, values: dict[str, Any] | None) -> None:
        """Resolve evidence controls and reject unsupported settings.

        Args:
            values (dict[str, Any] | None): Optional diagnostics configuration.
        """
        if values is not None and not isinstance(values, dict):
            raise ValueError("diagnostics must be a mapping")
        settings = values or {}
        allowed = {"enabled", "save_predictions", "evaluate_training", "checkpoint_epochs",
                   "high_confidence_threshold"}
        if set(settings) - allowed:
            raise ValueError("Unknown diagnostics settings")
        for name in ("enabled", "save_predictions", "evaluate_training"):
            if type(settings.get(name, name != "enabled")) is not bool:
                raise ValueError(f"diagnostics.{name} must be a boolean")
        epochs = settings.get("checkpoint_epochs", [])
        if (not isinstance(epochs, list)
                or any(type(epoch) is not int or epoch <= 0 for epoch in epochs)
                or len(set(epochs)) != len(epochs)):
            raise ValueError("checkpoint_epochs must contain unique positive integers")
        threshold = settings.get("high_confidence_threshold", 0.8)
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 < threshold < 1):
            raise ValueError("high_confidence_threshold must be between zero and one")

        self.enabled = settings.get("enabled", False)
        self.save_predictions = settings.get("save_predictions", True)
        self.evaluate_training = settings.get("evaluate_training", True)
        self.checkpoint_epochs: list[int] = list(epochs)
        self.high_confidence_threshold = float(threshold)


class EpochDiagnostics:
    """Store incremental evidence from existing validation outputs without optimization."""

    def __init__(
        self, settings: EpochDiagnosticsSettings, output_dir: Path,
        checkpoint_metadata: dict[str, Any], gradient_clip: float,
    ) -> None:
        """Store evidence controls, checkpoint metadata, and the clipping threshold.

        Args:
            settings (EpochDiagnosticsSettings): Validated recording controls.
            output_dir (Path): Model output directory containing diagnostic evidence.
            checkpoint_metadata (dict[str, Any]): Portable checkpoint fields excluding weights.
            gradient_clip (float): Maximum norm applied by the training recipe.
        """
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive and finite")
        self.settings = settings
        self.directory = output_dir / "diagnostics"
        self.checkpoint_metadata = checkpoint_metadata
        self.gradient_clip = gradient_clip
        self.gradient_norms: list[float] = []


    def begin_epoch(self) -> None:
        """Reset gradient observations before the next training-data pass."""
        self.gradient_norms.clear()


    def record_gradient(self, norm: float) -> None:
        """Retain one pre-clipping norm without modifying any gradients.

        Args:
            norm (float): Global gradient norm returned by clip_grad_norm_.
        """
        if not math.isfinite(norm) or norm < 0:
            raise ValueError("Gradient norms must be nonnegative and finite")
        self.gradient_norms.append(float(norm))


    @staticmethod
    def _prediction_table(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
        """Validate alignment keys and derive per-target probability-loss evidence.

        Args:
            frame (pd.DataFrame): Ordered validation rows with participant, time, and target.
            probabilities (np.ndarray): Ordered four-class softmax probabilities.
        """
        keys = ["participant_id", "time", "target"]
        if not set(keys).issubset(frame.columns) or frame.empty:
            raise ValueError("Diagnostics require nonempty participant, time, and target columns")
        if (frame[keys].isna().any().any()
                or frame.duplicated(["participant_id", "time"]).any()
                or not frame.target.isin(range(4)).all()):
            raise ValueError("Diagnostic target keys must be valid and unique")
        probabilities = np.asarray(probabilities, dtype=float)
        if (probabilities.shape != (len(frame), 4)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Diagnostics require aligned finite four-class probabilities")

        table = frame[keys].reset_index(drop=True).copy()
        table["participant_id"] = table.participant_id.astype(str)
        table["prediction"] = probabilities.argmax(axis=1)
        for label in range(4):
            table[f"probability_{label}"] = probabilities[:, label]
        table["true_class_probability"] = probabilities[
            np.arange(len(table)), table.target.to_numpy(dtype=int),
        ]
        table["confidence"] = probabilities.max(axis=1)
        table["nll"] = -np.log(np.maximum(table.true_class_probability, 1e-12))
        return table


    @staticmethod
    def _group_summary(table: pd.DataFrame, total: int) -> dict[str, Any]:
        """Summarize one disjoint group and its contribution to pooled NLL.

        Args:
            table (pd.DataFrame): Selected validation rows, possibly empty.
            total (int): Number of targets in the complete validation fold.
        """
        count = len(table)
        loss_sum = float(table.nll.sum())
        return {
            "count": count, "nll": float(table.nll.mean()) if count else None,
            "nll_sum": loss_sum, "nll_contribution": loss_sum / total,
            "accuracy": float((table.prediction == table.target).mean()) if count else None,
        }


    def _classification_summary(
        self, table: pd.DataFrame,
    ) -> tuple[list[list[int]], dict[str, dict[str, Any]]]:
        """Calculate fixed-order confusion counts and stage precision, recall, and F1.

        Args:
            table (pd.DataFrame): Validated targets, predictions, and probability losses.
        """
        matrix = np.zeros((4, 4), dtype=int)
        np.add.at(matrix, (table.target.to_numpy(dtype=int),
                           table.prediction.to_numpy(dtype=int)), 1)
        classes: dict[str, dict[str, Any]] = {}
        for label in range(4):
            true_positive = int(matrix[label, label])
            support, predicted = int(matrix[label].sum()), int(matrix[:, label].sum())
            classes[str(label)] = self._group_summary(table[table.target == label], len(table)) | {
                "precision": true_positive / predicted if predicted else 0.0,
                "recall": true_positive / support if support else 0.0,
                "f1": 2 * true_positive / (support + predicted) if support + predicted else 0.0,
                "support": support,
            }
        return matrix.tolist(), classes


    def _validation_summary(self, table: pd.DataFrame) -> dict[str, Any]:
        """Describe losses and confidence across classes and held-out participants.

        Args:
            table (pd.DataFrame): Validated target predictions and probability losses.
        """
        correct = table.prediction == table.target
        matrix, classes = self._classification_summary(table)
        result = self._group_summary(table, len(table))
        result.update(
            mean_confidence_correct=float(table.loc[correct, "confidence"].mean())
            if correct.any() else None,
            mean_confidence_wrong=float(table.loc[~correct, "confidence"].mean())
            if (~correct).any() else None,
            high_confidence_error_fraction=float((
                ~correct & (table.confidence >= self.settings.high_confidence_threshold)
            ).mean()),
            high_confidence_threshold=self.settings.high_confidence_threshold,
            classes=classes, confusion_matrix=matrix,
            participants={str(participant): self._group_summary(rows, len(table))
                          for participant, rows in table.groupby("participant_id", sort=True)},
        )
        return result


    def _optimizer_summary(self, learning_rates: list[float]) -> dict[str, Any]:
        """Record actual learning rates and pre-clipping gradient statistics.

        Args:
            learning_rates (list[float]): Current learning rate for every optimizer group.
        """
        if (not learning_rates or any(not math.isfinite(rate) or rate < 0
                                     for rate in learning_rates)):
            raise ValueError("Learning rates must be nonnegative and finite")
        norms = self.gradient_norms
        return {
            "learning_rates": learning_rates, "gradient_clip": self.gradient_clip,
            "gradient_steps": len(norms),
            "gradient_norm_pre_clip_mean": float(np.mean(norms)) if norms else None,
            "gradient_norm_pre_clip_max": max(norms) if norms else None,
            "clipped_step_fraction": float(np.mean(np.asarray(norms) > self.gradient_clip))
            if norms else None,
        }


    def _artifact(self, path: Path) -> dict[str, str]:
        """Describe a local artifact with a portable relative path and content hash.

        Args:
            path (Path): Completed evidence file below the diagnostics directory.
        """
        return {"path": path.relative_to(self.directory).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


    def _save_predictions(self, epoch: int, table: pd.DataFrame) -> dict[str, str]:
        """Persist compressed target-level validation evidence without overwriting files.

        Args:
            epoch (int): Completed optimization epoch.
            table (pd.DataFrame): Validated target-level predictions and loss values.
        """
        path = self.directory / "predictions" / f"epoch-{epoch:03d}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_name(path.name + ".tmp")
        table.to_csv(temporary, index=False, compression={"method": "gzip", "mtime": 0})
        temporary.replace(path)
        return self._artifact(path)


    def _save_checkpoint(self, epoch: int, model: nn.Module) -> dict[str, str]:
        """Store a portable intermediate checkpoint without changing model tensors.

        Args:
            epoch (int): Completed optimization epoch.
            model (nn.Module): Current model whose weights require inspection.
        """
        path = self.directory / "checkpoints" / f"epoch-{epoch:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        state = {name: tensor.detach().cpu().clone()
                 for name, tensor in model.state_dict().items()}
        temporary = path.with_suffix(".tmp")
        torch.save(self.checkpoint_metadata | {"state_dict": state, "epoch": epoch}, temporary)
        temporary.replace(path)
        return self._artifact(path)


    def record_epoch(
        self, epoch: int, frame: pd.DataFrame, probabilities: np.ndarray,
        training_nll: float | None, learning_rates: list[float], model: nn.Module,
        validation_nll: float, train_loss: float, validation_macro_f1: float,
    ) -> Path:
        """Save one completed epoch's evidence using existing validation probabilities.

        Args:
            epoch (int): Completed optimization epoch.
            frame (pd.DataFrame): Ordered validation targets with participant and time keys.
            probabilities (np.ndarray): Four-class validation probabilities in target order.
            training_nll (float | None): Optional end-of-epoch training loss in evaluation mode.
            learning_rates (list[float]): Actual optimizer parameter-group learning rates.
            model (nn.Module): Current model used only for optional checkpoint storage.
            validation_nll (float): Validation criterion loss from the existing forward pass.
            train_loss (float): Stochastic training loss averaged during parameter updates.
            validation_macro_f1 (float): Existing checkpoint-selection score.
        """
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("Epoch must be a positive integer")
        scalars = [validation_nll, train_loss, validation_macro_f1]
        if training_nll is not None:
            scalars.append(training_nll)
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("Diagnostic losses and scores must be finite")
        path = self.directory / "epochs" / f"epoch-{epoch:03d}.json"
        if path.exists():
            raise FileExistsError(path)
        table = self._prediction_table(frame, probabilities)
        record = {
            "schema_version": 1, "epoch": epoch,
            "validation": self._validation_summary(table),
            "validation_criterion_nll": validation_nll,
            "validation_macro_f1": validation_macro_f1,
            "training": {"evaluation_nll": training_nll, "stochastic_nll": train_loss},
            "optimizer": self._optimizer_summary(learning_rates), "files": {},
        }
        if self.settings.save_predictions:
            record["files"]["validation_predictions"] = self._save_predictions(epoch, table)
        if epoch in self.settings.checkpoint_epochs:
            record["files"]["checkpoint"] = self._save_checkpoint(epoch, model)

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        return path
