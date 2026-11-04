import argparse
import copy
import json
import logging
import math
import platform
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from sleep_classifier.baseline import CLASS_LABELS, PreparedFeatureDataset
from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.gru_model import CausalGRUClassifier
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.optimizer_adapter import AgentOptimizerSettings, OptimizerAdapter
from sleep_classifier.sequences import CausalSequenceDataset, SequenceBatchCollator

LOGGER = logging.getLogger(__name__)


class TrainingConfig:
    """Validate and hold GRU training settings."""

    def __init__(self, values: dict) -> None:
        self.values = values
        AgentOptimizerSettings(values.get("optimizer_agent"))
        integer_settings = (
            "seed",
            "epochs",
            "batch_size",
            "context_epochs",
            "hidden_size",
            "patience",
        )
        for name in integer_settings:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if (name == "seed" and value < 0) or (name != "seed" and value <= 0):
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gradient_clip"):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if values.get("device", "auto") not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")

    def device(self) -> torch.device:
        """Select CUDA, MPS, then CPU for automatic device selection."""
        requested = self.values.get("device", "auto")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        return torch.device(requested)


class EarlyStopping:
    """Track and restore the model state with the best validation score."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = -1.0
        self.best_epoch = 0
        self.stale_epochs = 0
        self.best_state = None

    def update(self, model, epoch: int, score: float) -> bool:
        """Record a new best state or report whether patience is exhausted."""
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience

    def restore(self, model) -> None:
        """Load the best recorded state into the model."""
        if self.best_state is None:
            raise RuntimeError("Training did not produce a best model state")
        model.load_state_dict(self.best_state)


class GRUExperiment:
    """Train and evaluate a causal GRU on prepared development participants."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.settings: dict = {}
        self.config: TrainingConfig | None = None
        self.prepared_dir = Path()
        self.output_dir = Path()

    def run_check(self) -> dict:
        """Validate configured inputs and model shapes without optimization."""
        context = self._prepare()
        collator = SequenceBatchCollator()
        batch_size = min(int(self.settings["batch_size"]), len(context["train_data"]))
        batch = collator([context["train_data"][index] for index in range(batch_size)])
        with torch.no_grad():
            logits = context["model"](batch["features"].to(context["device"]), batch["lengths"])
        result = {
            "train_rows": len(context["dataset"]["partitions"]["train"]),
            "validation_rows": len(context["validation_frame"]),
            "train_sequences": len(context["train_data"]),
            "validation_sequences": len(context["validation_data"]),
            "input_shape": list(batch["features"].shape),
            "logit_shape": list(logits.shape),
            "scaler_mean": context["scaler"].mean_.tolist(),
            "device": str(context["device"]),
            "parameter_count": sum(
                parameter.numel() for parameter in context["model"].parameters()
            ),
        }
        LOGGER.info("GRU check passed: %s", json.dumps(result, sort_keys=True))
        return result

    def run(self, allow_live_agent: bool = False) -> Path:
        """Train with explicit control over live optimizer-agent calls.

        Args:
            allow_live_agent (bool): Permit the configured live OPAL backend.
        """
        started = perf_counter()
        context = self._prepare()
        train_loader, validation_loader = self._build_loaders(context)
        optimizer = self._create_optimizer(context["model"], len(train_loader), allow_live_agent)
        criterion = nn.CrossEntropyLoss()
        LOGGER.info(
            "Training GRU: train_sequences=%d validation_sequences=%d device=%s",
            len(context["train_data"]),
            len(context["validation_data"]),
            context["device"],
        )
        try:
            best_epoch, history = self._fit(
                context["model"],
                train_loader,
                validation_loader,
                optimizer,
                criterion,
                context["device"],
                context["validation_frame"],
            )
        finally:
            optimizer.close()
        context["optimizer_provenance"] = optimizer.provenance()
        metrics_path = self._finish(context, validation_loader, criterion, best_epoch, history)
        LOGGER.info(
            "GRU results saved: path=%s elapsed_seconds=%.2f",
            self.output_dir,
            perf_counter() - started,
        )
        return metrics_path

    def _create_optimizer(
        self, model: nn.Module, batches: int, allow_live_agent: bool
    ) -> OptimizerAdapter:
        """Build Adam or the explicitly selected OPAL wrapper.

        Args:
            model (nn.Module): Initialized training model.
            batches (int): Training batches per epoch.
            allow_live_agent (bool): Permit live agent construction.
        """
        settings = self.settings.get("optimizer_agent")
        if settings:
            if settings["backend"] == "codex" and not allow_live_agent:
                raise RuntimeError("Live OPAL requires explicit --allow-live-agent opt-in")
            self.output_dir.mkdir(parents=True, exist_ok=False)

        adam = torch.optim.Adam(model.parameters(), lr=float(self.settings["learning_rate"]))
        return OptimizerAdapter.create(
            adam,
            model,
            batches * int(self.settings["epochs"]),
            self.output_dir,
            settings=settings,
            allow_live_agent=allow_live_agent,
        )

    def _build_loaders(self, context: dict) -> tuple[DataLoader, DataLoader]:
        """Create deterministic train and ordered validation loaders."""
        generator = torch.Generator().manual_seed(int(self.settings["seed"]))
        collator = SequenceBatchCollator()
        train_loader = DataLoader(
            context["train_data"],
            batch_size=int(self.settings["batch_size"]),
            shuffle=True,
            num_workers=0,
            collate_fn=collator,
            generator=generator,
        )
        validation_loader = DataLoader(
            context["validation_data"],
            batch_size=int(self.settings["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        return train_loader, validation_loader

    def _fit(self, model, train_loader, validation_loader, optimizer, criterion, device, frame):
        """Run epochs and restore the checkpoint with the best validation macro-F1."""
        stopping = EarlyStopping(int(self.settings["patience"]))
        history = []
        max_epochs = int(self.settings["epochs"])
        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                float(self.settings["gradient_clip"]),
            )
            validation_loss, targets, predictions, _ = self._evaluate(
                model, validation_loader, criterion, device
            )
            self._assert_target_alignment(targets, frame)
            summary = self._epoch_summary(
                epoch, train_loss, validation_loss, targets, predictions, frame
            )
            history.append(summary)
            stop = stopping.update(model, epoch, summary["validation_macro_f1"])
            if stop:
                LOGGER.info(
                    "Early stopping at epoch %d; best validation macro-F1 epoch=%d",
                    epoch,
                    stopping.best_epoch,
                )
                break
        stopping.restore(model)
        return stopping.best_epoch, history

    def _epoch_summary(
        self, epoch, train_loss, validation_loss, targets, predictions, frame
    ) -> dict:
        """Calculate and log the scalar metrics used by early stopping."""
        metrics = self._metrics(predictions, targets, frame)
        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": metrics["macro_f1"],
            "validation_accuracy": metrics["accuracy"],
        }
        LOGGER.info(
            "Epoch %d/%d train_loss=%.5f validation_loss=%.5f validation_macro_f1=%.5f "
            "validation_accuracy=%.5f",
            epoch,
            int(self.settings["epochs"]),
            train_loss,
            validation_loss,
            metrics["macro_f1"],
            metrics["accuracy"],
        )
        return summary

    def _finish(self, context, validation_loader, criterion, best_epoch, history) -> Path:
        """Evaluate the restored model and persist validation artifacts.

        Args:
            context (dict): Prepared inputs, model, scaler, and optimizer provenance.
            validation_loader (DataLoader): Held-out participant batches.
            criterion (nn.Module): Validation loss function.
            best_epoch (int): Selected checkpoint epoch.
            history (list[dict]): Recorded epoch losses and scores.
        """
        model = context["model"]
        device = context["device"]
        _, targets, predictions, probabilities = self._evaluate(
            model, validation_loader, criterion, device
        )
        validation_frame = context["validation_frame"]
        self._assert_target_alignment(targets, validation_frame)
        metrics = self._metrics(predictions, targets, validation_frame)
        metrics["best_epoch"] = best_epoch
        metrics["history"] = history
        metrics["config"] = self.settings
        metrics["features"] = list(FEATURE_COLUMNS)
        metrics["provenance"] = context["dataset"]["provenance"]
        metrics["device"] = str(device)
        metrics["optimizer"] = context.get("optimizer_provenance", {})
        metrics["packages"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        }
        predictions_table = self._prediction_table(validation_frame, predictions, probabilities)
        self._save(model, context["scaler"], metrics, predictions_table)
        return self.output_dir / "metrics.json"

    def _assert_target_alignment(self, targets: np.ndarray, frame: pd.DataFrame) -> None:
        """Require sequence evaluation order to match prepared row order."""
        expected_targets = frame["target"].to_numpy(dtype=int)
        if not np.array_equal(targets, expected_targets):
            raise ValueError("Validation sequence targets do not align with validation rows")

    def _prepare(self):
        """Load development data, fit training-only scaling, and build model."""
        self.settings = yaml.safe_load(self.config_path.read_text())
        self.config = TrainingConfig(self.settings)
        prepare_path = self.config_path.parent / self.settings["prepare_config"]
        prepare_settings = yaml.safe_load(prepare_path.read_text())
        self.prepared_dir = (prepare_path.parent / prepare_settings["output_dir"]).resolve()
        self.output_dir = (self.config_path.parent / self.settings["output_dir"]).resolve()
        dataset = PreparedFeatureDataset(self.prepared_dir).load()
        train = dataset["partitions"]["train"]
        validation = (
            dataset["partitions"]["validation"]
            .sort_values(["participant_id", "time"], kind="stable")
            .reset_index(drop=True)
        )
        dataset["partitions"]["validation"] = validation
        scaler = StandardScaler().fit(train.loc[:, FEATURE_COLUMNS])
        train_data = CausalSequenceDataset(train, scaler, int(self.settings["context_epochs"]))
        validation_data = CausalSequenceDataset(
            validation, scaler, int(self.settings["context_epochs"])
        )
        self._seed(int(self.settings["seed"]))
        device = self.config.device()
        model = CausalGRUClassifier(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=int(self.settings["hidden_size"]),
            num_classes=len(CLASS_LABELS),
        ).to(device)
        LOGGER.info(
            "Loaded prepared rows: train_rows=%d validation_rows=%d train_sequences=%d "
            "validation_sequences=%d scaler_fit=train_only",
            len(train),
            len(validation),
            len(train_data),
            len(validation_data),
        )
        return {
            "dataset": dataset,
            "scaler": scaler,
            "train_data": train_data,
            "validation_data": validation_data,
            "validation_frame": validation,
            "model": model,
            "device": device,
        }

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _train_epoch(
        self, model, loader, optimizer, criterion, device, gradient_clip: float
    ) -> float:
        """Apply one training epoch with the configured optimizer.

        Args:
            model (nn.Module): Model updated by this epoch.
            loader (DataLoader): Training batches.
            optimizer (OptimizerAdapter): Adam or OPAL update interface.
            criterion (nn.Module): Training loss function.
            device (torch.device): Device for tensors.
            gradient_clip (float): Maximum gradient norm.
        """
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features, batch["lengths"]), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step(loss)
            total_loss += float(loss.item()) * len(targets)
            total_count += len(targets)
        return total_loss / total_count

    def _evaluate(self, model, loader, criterion, device):
        model.eval()
        losses = 0.0
        count = 0
        targets_all, predictions_all, probabilities_all = [], [], []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(device)
                targets = batch["targets"].to(device)
                logits = model(features, batch["lengths"])
                losses += float(criterion(logits, targets).item()) * len(targets)
                count += len(targets)
                targets_all.extend(targets.cpu().numpy().tolist())
                predictions_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                probabilities_all.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
        return (
            losses / count,
            np.asarray(targets_all),
            np.asarray(predictions_all),
            np.asarray(probabilities_all),
        )

    def _metrics(self, predictions: np.ndarray, targets: np.ndarray, frame: pd.DataFrame) -> dict:
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=CLASS_LABELS, zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(np.mean(f1)),
            "classes": {
                str(label): {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(CLASS_LABELS)
            },
            "confusion_matrix": confusion_matrix(
                targets, predictions, labels=CLASS_LABELS
            ).tolist(),
            "coverage": self._coverage(frame),
            "participants": self._participant_metrics(frame, predictions),
        }

    def _coverage(self, frame: pd.DataFrame) -> dict:
        scored = frame.groupby("participant_id")["scored_epochs"].first()
        return {
            "scored_epochs": int(scored.sum()),
            "eligible_epochs": int(
                frame.groupby("participant_id")["eligible_epochs"].first().sum()
            ),
            "label_bearing_epochs": int(scored.sum()),
            "fraction": float(len(frame) / scored.sum()) if scored.sum() else 0.0,
        }

    def _participant_metrics(self, frame: pd.DataFrame, predictions: np.ndarray) -> dict:
        results = {}
        for participant_id, indexes in frame.groupby("participant_id").indices.items():
            selected = np.asarray(indexes, dtype=int)
            targets = frame.iloc[selected]["target"].to_numpy(dtype=int)
            _, _, f1, _ = precision_recall_fscore_support(
                targets, predictions[selected], labels=CLASS_LABELS, zero_division=0
            )
            row = frame.iloc[selected]
            scored = int(row["scored_epochs"].iloc[0])
            results[str(participant_id)] = {
                "scored_epochs": scored,
                "eligible_epochs": int(row["eligible_epochs"].iloc[0]),
                "label_bearing_epochs": scored,
                "coverage": float(len(selected) / scored) if scored else 0.0,
                "accuracy": float(accuracy_score(targets, predictions[selected])),
                "macro_f1": float(np.mean(f1)),
            }
        return results

    def _prediction_table(self, frame: pd.DataFrame, predictions, probabilities) -> pd.DataFrame:
        columns = ["participant_id"] + (["time"] if "time" in frame.columns else []) + ["target"]
        table = frame.loc[:, columns].reset_index(drop=True).copy()
        table["gru_prediction"] = predictions
        for label in CLASS_LABELS:
            table[f"probability_{label}"] = probabilities[:, label]
        return table

    def _save(self, model, scaler, metrics: dict, predictions: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": int(self.settings["hidden_size"]),
            "num_classes": len(CLASS_LABELS),
        }
        checkpoint = {
            "state_dict": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "model_config": config,
            "features": list(FEATURE_COLUMNS),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "context_epochs": int(self.settings["context_epochs"]),
            "training_config": self.settings,
        }
        torch.save(checkpoint, self.output_dir / "gru_checkpoint.pt")
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        predictions.to_csv(self.output_dir / "validation_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or check the causal GRU experiment")
    parser.add_argument("--config", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--train", action="store_true", help="Confirm the requested model fit")
    operation.add_argument(
        "--check", action="store_true", help="Validate loading and forward pass only"
    )
    parser.add_argument(
        "--allow-live-agent",
        action="store_true",
        help="Permit the configured live OPAL agent during explicit training",
    )
    args = parser.parse_args()
    if args.allow_live_agent and not args.train:
        parser.error("--allow-live-agent requires --train")
    configure_logging()
    experiment = GRUExperiment(args.config)
    if args.check:
        experiment.run_check()
    else:
        LOGGER.info("GRU metrics saved: %s", experiment.run(allow_live_agent=args.allow_live_agent))


if __name__ == "__main__":
    main()
