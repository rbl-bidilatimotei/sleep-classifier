import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sleep_classifier.baseline import PreparedFeatureDataset
from sleep_classifier.paired_architecture_report import PairedArchitectureReport

EPOCH_KEYS = ["participant_id", "time", "target"]
STAGE_NAMES = ("wake", "light", "deep", "REM")


class TemporalSliceReport:
    """Compare saved predictions by available history and true stage transitions."""

    def __init__(
        self, reference: pd.DataFrame, candidate: pd.DataFrame,
        validation_rows: pd.DataFrame, context_epochs: int = 30,
        short_history_epochs: int = 2,
    ) -> None:
        """Store matched predictions and the prepared segment boundaries.

        Args:
            reference (pd.DataFrame): Reference epoch keys and prediction column.
            candidate (pd.DataFrame): Candidate epoch keys and prediction column.
            validation_rows (pd.DataFrame): Eligible epoch keys and segment_id column.
            context_epochs (int): Maximum available history used by both models.
            short_history_epochs (int): Upper bound of the short-history slice.
        """
        if (type(context_epochs) is not int or type(short_history_epochs) is not int
                or not 0 < short_history_epochs < context_epochs - 1):
            raise ValueError("History bounds must leave a nonempty intermediate bin")

        self.context_epochs = context_epochs
        self.short_history_epochs = short_history_epochs
        self.reference = self._validate_frame(reference, "prediction")
        self.candidate = self._validate_frame(candidate, "prediction")
        self.rows = self._validate_frame(validation_rows, "segment_id")
        if (not self.reference[EPOCH_KEYS].equals(self.candidate[EPOCH_KEYS])
                or not self.reference[EPOCH_KEYS].equals(self.rows[EPOCH_KEYS])):
            raise ValueError("Prediction and metadata epoch populations must match exactly")

        self._validate_segments()


    @staticmethod
    def _validate_frame(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
        """Validate and sort one unique epoch table.

        Args:
            frame (pd.DataFrame): Prediction or eligible metadata table.
            value_column (str): Required prediction or segment column.
        """
        required = EPOCH_KEYS + [value_column]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Epoch table lacks required columns: {required}")

        result = frame[required].copy()
        if result.empty or result.isna().any().any():
            raise ValueError("Epoch tables must be nonempty and complete")
        result["participant_id"] = result.participant_id.astype(str)
        numeric = result[["time", "target", value_column]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Epoch values must be finite")
        if not result.target.isin(range(4)).all():
            raise ValueError("Targets must use stage codes 0..3")
        values = numeric[:, 2]
        if value_column == "prediction" and not result.prediction.isin(range(4)).all():
            raise ValueError("Predictions must use stage codes 0..3")
        if value_column == "segment_id" and ((values < 0).any() or (values % 1 != 0).any()):
            raise ValueError("Segment identifiers must be nonnegative integers")
        if result.duplicated(["participant_id", "time"]).any():
            raise ValueError("Epoch keys must be unique")

        result["time"] = result.time.astype(float)
        result["target"] = result.target.astype(int)
        result[value_column] = result[value_column].astype(int)
        return result.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)


    def _validate_segments(self) -> None:
        """Reject noncontiguous or interleaved prepared segments."""
        previous = self.rows.groupby("participant_id", sort=False).segment_id.shift()
        starts = self.rows.loc[self.rows.segment_id.ne(previous)]
        if starts.duplicated(["participant_id", "segment_id"]).any():
            raise ValueError("Segment identifiers must form consecutive blocks")

        groups = self.rows.groupby(["participant_id", "segment_id"], sort=False)
        differences = groups.time.diff().dropna()
        if not np.isclose(differences.to_numpy(), 30.0, rtol=0, atol=1e-8).all():
            raise ValueError("Epochs within each segment must be contiguous at 30 seconds")


    @classmethod
    def from_grouped(
        cls, reference_summary: Path, candidate_summary: Path, short_history_epochs: int = 2,
    ) -> "TemporalSliceReport":
        """Load verified grouped predictions and immutable prepared segment metadata.

        Args:
            reference_summary (Path): Completed reference grouped summary.
            candidate_summary (Path): Completed candidate grouped summary.
            short_history_epochs (int): Upper bound of the short-history slice.
        """
        PairedArchitectureReport(reference_summary, candidate_summary).build()
        reference = cls._load_predictions(reference_summary)
        candidate = cls._load_predictions(candidate_summary)
        rows, context = cls._load_metadata(candidate_summary)
        return cls(reference, candidate, rows, context, short_history_epochs)


    @staticmethod
    def _load_predictions(summary_path: Path) -> pd.DataFrame:
        """Read previously verified prediction artifacts for all grouped folds.

        Args:
            summary_path (Path): Completed grouped summary path.
        """
        summary = json.loads(summary_path.read_text())
        frames = []
        for result in summary["results"]:
            metrics_path = Path(result["metrics_path"])
            if not metrics_path.is_absolute():
                metrics_path = summary_path.parent / metrics_path
            frames.append(PairedArchitectureReport._read_predictions(metrics_path.parent))
        return pd.concat(frames, ignore_index=True)


    @staticmethod
    def _verify_features(root: Path, feature_dir: Path, hashes: dict[str, str]) -> None:
        """Verify every prepared feature table and its provenance against the manifest.

        Args:
            root (Path): Grouped experiment root.
            feature_dir (Path): Fold feature directory.
            hashes (dict[str, str]): Recorded derived artifact digests.
        """
        paths = list(feature_dir.rglob("*.csv")) + [feature_dir / "provenance.json"]
        prefix = str(feature_dir.relative_to(root)) + "/"
        expected = {key for key in hashes if key.startswith(prefix)}
        observed = {str(path.relative_to(root)) for path in paths}
        if expected != observed:
            raise ValueError("Prepared feature inventory differs from the manifest")
        for path in paths:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Prepared feature paths must remain inside the grouped root")
            key = str(path.relative_to(root))
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key]:
                raise ValueError(f"Prepared feature checksum differs: {key}")


    @classmethod
    def _load_metadata(cls, summary_path: Path) -> tuple[pd.DataFrame, int]:
        """Load validation metadata after verifying its preparation manifest.

        Args:
            summary_path (Path): Completed candidate grouped summary path.
        """
        root = summary_path.resolve().parent
        summary = json.loads(summary_path.read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["plan"] != summary["plan"]:
            raise ValueError("Preparation manifest and result plans differ")

        rows = []
        contexts = set()
        for fold in summary["plan"]["folds"]:
            feature_dir = root / fold["name"] / "features"
            if not feature_dir.resolve().is_relative_to(root):
                raise ValueError("Fold feature directory must remain inside the grouped root")
            cls._verify_features(root, feature_dir, manifest["derived_hashes"])
            rows.append(PreparedFeatureDataset(feature_dir).load()["partitions"]["validation"])
            result = next(item for item in summary["results"] if item["fold"] == fold["name"])
            metrics_path = Path(result["metrics_path"])
            if not metrics_path.is_absolute():
                metrics_path = root / metrics_path
            contexts.add(json.loads(metrics_path.read_text())["config"]["context_epochs"])

        if len(contexts) != 1:
            raise ValueError("All folds must use the same context length")
        return pd.concat(rows, ignore_index=True), contexts.pop()


    @staticmethod
    def _metrics(frame: pd.DataFrame) -> dict[str, Any] | None:
        """Compute four-class scores for one nonempty prediction slice.

        Args:
            frame (pd.DataFrame): Matched target and prediction values.
        """
        if frame.empty:
            return None
        precision, recall, f1, support = precision_recall_fscore_support(
            frame.target, frame.prediction, labels=range(4), zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(frame.target, frame.prediction)),
            "macro_f1": float(f1.mean()),
            "classes": {
                name: {"precision": float(precision[index]), "recall": float(recall[index]),
                       "f1": float(f1[index]), "support": int(support[index])}
                for index, name in enumerate(STAGE_NAMES)
            },
        }


    def build(self) -> dict[str, Any]:
        """Return paired history bins and the overlapping true-transition slice."""
        groups = self.rows.groupby(["participant_id", "segment_id"], sort=False)
        history = (groups.cumcount() + 1).clip(upper=self.context_epochs)
        transition = groups.target.shift().notna() & self.rows.target.ne(groups.target.shift())
        short = self.short_history_epochs
        context = self.context_epochs
        masks = {
            f"history_1_{short}": history <= short,
            f"history_{short + 1}_{context - 1}": (history > short) & (history < context),
            f"history_{context}": history == context,
            "stage_transition": transition,
        }
        slices = []
        for name, mask in masks.items():
            reference = self._metrics(self.reference.loc[mask])
            candidate = self._metrics(self.candidate.loc[mask])
            slices.append({
                "name": name, "targets": int(mask.sum()),
                "reference": reference, "candidate": candidate,
                "macro_f1_difference": None if reference is None else (
                    candidate["macro_f1"] - reference["macro_f1"]
                ),
            })

        return {
            "context_epochs": context, "short_history_epochs": short,
            "posthoc": True, "independent_outer_test": False,
            "definitions": {
                "history": "Contiguous eligible epochs within each segment, capped at context",
                "stage_transition": "True stage changes within the same contiguous segment",
                "interpretation": "History bins are disjoint; transitions overlap; inspect support",
            },
            "slices": slices,
        }
