from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sleep_classifier.logging_utils import configure_logging

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)


class EpochEvidenceReport:
    """Inspect incremental experiment evidence without loading or changing models."""

    def __init__(self, output_dir: Path) -> None:
        """Locate one grouped experiment's local evidence directory.

        Args:
            output_dir (Path): Grouped output directory containing planned fold subdirectories.
        """
        self.output_dir = output_dir.resolve()


    def _fold_names(self) -> list[str]:
        """Read planned fold names or discover existing fold directories."""
        manifest = self.output_dir / "manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            names = [fold["name"] for fold in saved["plan"]["folds"]]
        else:
            names = sorted(path.name for path in self.output_dir.glob("fold-*") if path.is_dir())
        if (any(not isinstance(name, str) or re.fullmatch(r"fold-\d+", name) is None
                for name in names) or len(set(names)) != len(names)):
            raise ValueError("Evidence requires unique numbered fold names")
        return names


    @staticmethod
    def _number(value: Any, name: str, optional: bool = False) -> float | None:
        """Validate a finite scalar while preserving explicitly absent observations.

        Args:
            value (Any): Decoded scalar value.
            name (str): Field name used for validation errors.
            optional (bool): Permit None when evidence was not collected.
        """
        if value is None and optional:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"Invalid evidence scalar: {name}")
        return float(value)


    @staticmethod
    def _linked_path(directory: Path, artifact: dict[str, str]) -> Path:
        """Resolve a relative evidence file and require its recorded content hash.

        Args:
            directory (Path): Diagnostics root used for portable artifact paths.
            artifact (dict[str, str]): Relative path and expected SHA-256 digest.
        """
        if not isinstance(artifact, dict):
            raise ValueError("Evidence artifact metadata must be a mapping")
        raw = artifact.get("path")
        digest = artifact.get("sha256")
        if (not isinstance(raw, str) or not raw or Path(raw).is_absolute()
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("Evidence artifacts require relative paths and SHA-256 hashes")
        path = (directory / raw).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Evidence artifact path escapes the diagnostics directory")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Evidence artifact is missing or its hash differs")
        return path


    def _read_epochs(self, fold: str) -> list[dict[str, Any]]:
        """Validate completed epoch records and every linked artifact.

        Args:
            fold (str): Planned numbered fold name.
        """
        directory = self.output_dir / fold / "results/diagnostics"
        records: list[dict[str, Any]] = []
        for path in sorted((directory / "epochs").glob("epoch-*.json")):
            record = json.loads(path.read_text())
            epoch = record.get("epoch")
            count = record.get("validation", {}).get("count")
            if (type(record.get("schema_version")) is not int
                    or record["schema_version"] != 1 or type(epoch) is not int or epoch <= 0
                    or type(count) is not int or count <= 0):
                raise ValueError("Evidence requires version 1 and positive epoch/count values")
            if path.name != f"epoch-{epoch:03d}.json":
                raise ValueError("Evidence epoch filename differs from its recorded number")
            if not isinstance(record.get("files"), dict):
                raise ValueError("Evidence files must be a mapping")
            for artifact in record["files"].values():
                self._linked_path(directory, artifact)
            self._flat_epoch(record)
            records.append(record)
        if len({record["epoch"] for record in records}) != len(records):
            raise ValueError("Evidence epoch numbers must be unique")
        return sorted(records, key=lambda record: record["epoch"])


    def _group_summaries(
        self, groups: dict[str, Any], total: int, name: str,
    ) -> dict[str, Any]:
        """Validate disjoint group counts and their pooled probability-loss contributions.

        Args:
            groups (dict[str, Any]): Raw class or participant summaries from the recorder.
            total (int): Number of scored validation targets.
            name (str): Grouping name used in validation errors.
        """
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"Evidence {name} must be a nonempty summary mapping")
        count_sum = 0
        for group in groups.values():
            count = group["count"]
            if type(count) is not int or count < 0:
                raise ValueError(f"Evidence {name} counts must be nonnegative integers")
            mean = self._number(group["nll"], "group_nll", optional=count == 0)
            loss_sum = self._number(group["nll_sum"], "group_nll_sum")
            contribution = self._number(group["nll_contribution"], "group_nll_contribution")
            if (loss_sum < 0 or contribution < 0 or (mean is not None and mean < 0)
                    or not np.isclose(loss_sum, (mean or 0) * count, atol=1e-8)
                    or not np.isclose(contribution, loss_sum / total, atol=1e-8)):
                raise ValueError(f"Evidence {name} has inconsistent probability-loss summaries")
            for key in ("precision", "recall", "f1"):
                if key in group:
                    value = self._number(group[key], f"group_{key}")
                    if not 0 <= value <= 1:
                        raise ValueError("Class scores must be between zero and one")
            if "support" in group and group["support"] != count:
                raise ValueError("Class support differs from its recorded target count")
            count_sum += count
        if count_sum != total:
            raise ValueError(f"Evidence {name} does not cover the recorded validation count")
        return groups


    def _flat_epoch(self, record: dict[str, Any]) -> dict[str, Any]:
        """Extract measured scalar columns without replacing absent values with zero.

        Args:
            record (dict[str, Any]): Validated epoch evidence record.
        """
        optimizer = record["optimizer"]
        validation = record["validation"]
        rates = optimizer["learning_rates"]
        if not isinstance(rates, list) or not rates:
            raise ValueError("Evidence learning_rates must be a nonempty list")
        rates = [self._number(rate, "learning_rate") for rate in rates]
        if any(rate < 0 for rate in rates):
            raise ValueError("Evidence learning rates cannot be negative")
        return {
            "epoch": record["epoch"],
            "validation_nll": self._number(record["validation"]["nll"], "validation_nll"),
            "validation_criterion_nll": self._number(
                record["validation_criterion_nll"], "validation_criterion_nll",
            ),
            "validation_macro_f1": self._number(
                record["validation_macro_f1"], "validation_macro_f1",
            ),
            "training_evaluation_nll": self._number(
                record["training"]["evaluation_nll"], "training_evaluation_nll", optional=True,
            ),
            "training_stochastic_nll": self._number(
                record["training"]["stochastic_nll"], "training_stochastic_nll",
            ),
            "learning_rates": rates,
            "validation_classes": self._group_summaries(
                validation["classes"], validation["count"], "classes",
            ),
            "validation_participants": self._group_summaries(
                validation["participants"], validation["count"], "participants",
            ),
            "mean_confidence_wrong": self._number(
                validation["mean_confidence_wrong"], "mean_confidence_wrong", optional=True,
            ),
            "high_confidence_error_fraction": self._number(
                validation["high_confidence_error_fraction"], "high_confidence_error_fraction",
            ),
            "clipped_step_fraction": self._number(
                optimizer["clipped_step_fraction"], "clipped_step_fraction", optional=True,
            ),
            "gradient_norm_pre_clip_mean": self._number(
                optimizer["gradient_norm_pre_clip_mean"], "gradient_norm_mean", optional=True,
            ),
            "gradient_norm_pre_clip_max": self._number(
                optimizer["gradient_norm_pre_clip_max"], "gradient_norm_max", optional=True,
            ),
        }


    def build(self) -> dict[str, Any]:
        """Read pending or partial evidence without requiring a completed training run."""
        names = self._fold_names()
        summary_path = self.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        folds: list[dict[str, Any]] = []
        for name in names:
            records = self._read_epochs(name)
            metrics_path = self.output_dir / name / "results/metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            status = "complete" if metrics else "partial" if records else "prepared"
            if summary.get("failed_fold") == name:
                status = "failed"
            folds.append({
                "fold": name, "status": status,
                "epochs": [self._flat_epoch(record) for record in records],
                "best_epoch": metrics.get("best_epoch"),
            })
        status = "not_prepared" if not names else "partial" if any(
            fold["epochs"] for fold in folds
        ) else "prepared"
        if summary.get("status") in {"running", "failed", "complete"}:
            status = summary["status"]
        elif folds and all(fold["status"] == "complete" for fold in folds):
            status = "complete"
        return {"schema_version": 1, "status": status, "folds": folds}


    def plot_curves(self) -> Figure:
        """Plot losses, selection scores, gradients, and learning rates in separate panels."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        folds = [fold for fold in self.build()["folds"] if fold["epochs"]]
        if not folds:
            raise ValueError("No completed epoch evidence is available to plot")
        figure, axes = plt.subplots(len(folds), 4, figsize=(18, 4 * len(folds)), squeeze=False)
        for row, fold in enumerate(folds):
            frame = pd.DataFrame(fold["epochs"])
            for column, label in (("training_stochastic_nll", "Train, stochastic"),
                                  ("training_evaluation_nll", "Train, evaluation mode"),
                                  ("validation_nll", "Validation")):
                axes[row, 0].plot(frame.epoch, frame[column], label=label)
            axes[row, 1].plot(frame.epoch, frame.validation_macro_f1, label="Validation macro-F1")
            for column, label in (("gradient_norm_pre_clip_mean", "Mean"),
                                  ("gradient_norm_pre_clip_max", "Maximum")):
                axes[row, 2].plot(frame.epoch, frame[column], label=label)
            count = max(len(rates) for rates in frame.learning_rates)
            for index in range(count):
                rates = [values[index] if index < len(values) else np.nan
                         for values in frame.learning_rates]
                axes[row, 3].plot(frame.epoch, rates, label=f"Group {index}")
            for axis, title in zip(axes[row], ("NLL", "Macro-F1", "Pre-clip gradient norm", "LR")):
                axis.set(title=f"{fold['fold']}: {title}", xlabel="Optimization epoch")
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                axis.legend(fontsize=8)
                axis.grid(alpha=0.2)
        figure.tight_layout()
        return figure


    def _prediction_frame(self, fold: str, epoch: int) -> pd.DataFrame:
        """Recompute target losses from a verified epoch probability artifact.

        Args:
            fold (str): Planned numbered fold name.
            epoch (int): Completed optimization epoch to inspect.
        """
        if fold not in self._fold_names():
            raise ValueError("Requested fold is not part of this experiment")
        records = {record["epoch"]: record for record in self._read_epochs(fold)}
        if epoch not in records:
            raise ValueError(f"No completed evidence exists for epoch {epoch}")
        record = records[epoch]
        artifact = record["files"].get("validation_predictions")
        if artifact is None:
            raise ValueError("Requested epoch did not save target-level predictions")
        path = self._linked_path(self.output_dir / fold / "results/diagnostics", artifact)
        frame = pd.read_csv(path, dtype={"participant_id": str})
        keys = ["participant_id", "time", "target"]
        probabilities = [f"probability_{label}" for label in range(4)]
        if not set(keys + probabilities + ["nll"]).issubset(frame.columns) or frame.empty:
            raise ValueError("Epoch predictions lack required targets or probability columns")
        if (frame[keys].isna().any().any()
                or frame.duplicated(["participant_id", "time"]).any()
                or not frame.target.isin(range(4)).all()
                or len(frame) != record["validation"]["count"]):
            raise ValueError("Epoch prediction population is invalid or has changed")
        values = frame[probabilities].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Epoch probabilities must be finite normalized four-class values")
        losses = -np.log(np.maximum(values[np.arange(len(frame)), frame.target.astype(int)], 1e-12))
        if (not np.allclose(losses, frame.nll, atol=1e-7, rtol=0)
                or not np.isclose(losses.mean(), record["validation"]["nll"], atol=1e-7, rtol=0)):
            raise ValueError("Saved NLL differs from the epoch probabilities")
        frame["nll"] = losses
        return frame.sort_values(keys, kind="stable").reset_index(drop=True)


    @staticmethod
    def _group_changes(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        """Decompose the pooled loss difference into disjoint signed group contributions.

        Args:
            frame (pd.DataFrame): Aligned before/after target losses.
            column (str): Participant or true-class grouping column.
        """
        result: list[dict[str, Any]] = []
        for value, rows in frame.groupby(column, sort=True):
            result.append({
                column: str(value) if column == "participant_id" else int(value),
                "count": len(rows), "mean_nll_before": float(rows.nll_before.mean()),
                "mean_nll_after": float(rows.nll_after.mean()),
                "contribution_delta": float((rows.nll_after - rows.nll_before).sum() / len(frame)),
            })
        return result


    def compare_epochs(self, fold: str, before: int, after: int) -> dict[str, Any]:
        """Locate the people, classes, and targets contributing to a measured loss change.

        Args:
            fold (str): Planned numbered fold name.
            before (int): Earlier completed optimization epoch.
            after (int): Later completed optimization epoch.
        """
        if type(before) is not int or type(after) is not int or not 0 < before < after:
            raise ValueError("Comparison requires positive before < after epoch numbers")
        first = self._prediction_frame(fold, before)
        second = self._prediction_frame(fold, after)
        keys = ["participant_id", "time", "target"]
        if not first[keys].equals(second[keys]):
            raise ValueError("Epoch comparison requires identical target populations")
        changes = first[keys].copy()
        changes["nll_before"] = first.nll
        changes["nll_after"] = second.nll
        changes["nll_difference"] = second.nll - first.nll
        largest = changes.iloc[np.argsort(-np.abs(changes.nll_difference.to_numpy()))[:20]]
        return {
            "fold": fold, "before_epoch": before, "after_epoch": after,
            "count": len(changes), "nll_difference": float(changes.nll_difference.mean()),
            "participants": self._group_changes(changes, "participant_id"),
            "classes": self._group_changes(changes, "target"),
            "largest_targets": largest.to_dict("records"),
        }


def main() -> None:
    """Write local evidence summaries and an optional matched-epoch comparison."""
    parser = argparse.ArgumentParser(description="Inspect recorded epoch evidence without training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", action="store_true", required=True)
    parser.add_argument("--compare-fold")
    parser.add_argument("--before-epoch", type=int)
    parser.add_argument("--after-epoch", type=int)
    args = parser.parse_args()
    requested = [args.compare_fold is not None, args.before_epoch is not None,
                 args.after_epoch is not None]
    if any(requested) and not all(requested):
        parser.error("Epoch comparison requires --compare-fold, --before-epoch, and --after-epoch")
    configure_logging()
    report = EpochEvidenceReport(args.output_dir)
    result = report.build()
    if result["status"] == "not_prepared":
        LOGGER.info("Evidence report not written: status=not_prepared path=%s", args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "evidence-summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(fold["epochs"] for fold in result["folds"]):
        import matplotlib.pyplot as plt

        figure = report.plot_curves()
        figure.savefig(args.output_dir / "evidence-curves.png", dpi=150)
        plt.close(figure)
    if all(requested):
        comparison = report.compare_epochs(args.compare_fold, args.before_epoch, args.after_epoch)
        destination = args.output_dir / (
            f"{args.compare_fold}-epoch-{args.before_epoch:03d}-to-{args.after_epoch:03d}.json"
        )
        destination.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    LOGGER.info("Evidence report saved: path=%s status=%s", path, result["status"])


if __name__ == "__main__":
    main()
