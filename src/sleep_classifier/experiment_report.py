import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from sleep_classifier.logging_utils import configure_logging


class ExperimentComparisonReport:
    """Compare saved baseline metrics and export compact figures and a summary."""

    def __init__(self, reference_metrics: Path, comparison_metrics: Path, output_dir: Path):
        """Store metrics paths and the report destination."""
        self.reference_path = Path(reference_metrics)
        self.comparison_path = Path(comparison_metrics)
        self.output_dir = Path(output_dir)

    def run(self) -> Path:
        """Validate saved runs and generate a report without fitting models."""
        reference = self._read_metrics(self.reference_path)
        comparison = self._read_metrics(self.comparison_path)
        self._validate(reference, comparison)
        self._guard_output()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        matrices = self._plot_confusion(reference, comparison)
        self._plot_scores(reference, comparison)
        coefficient_path = self._plot_coefficients()
        timeline_path = self._plot_timelines()
        summary = self._summary(reference, comparison, matrices, coefficient_path, timeline_path)
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        self._write_readme(summary)
        return summary_path

    @staticmethod
    def _read_metrics(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Metrics file does not exist: {path}")
        return json.loads(path.read_text())

    def _validate(self, reference: dict, comparison: dict) -> None:
        for key in ("features", "classes"):
            if reference.get(key) != comparison.get(key):
                raise ValueError(f"Reference and comparison {key} differ")
        if reference.get("provenance", {}).get("split") != comparison.get("provenance", {}).get(
            "split"
        ):
            raise ValueError("Reference and comparison provenance split differ")
        for key in ("rule", "source_inventory"):
            first = reference.get("provenance", {}).get(key)
            second = comparison.get("provenance", {}).get(key)
            if (first is not None or second is not None) and first != second:
                raise ValueError(f"Reference and comparison provenance {key} differ")
        if self._config_for_comparison(reference.get("config", {})) != self._config_for_comparison(
            comparison.get("config", {})
        ):
            raise ValueError("Run configs may differ only in class_weight and output_dir")
        for model_name in ("majority", "logistic_regression"):
            first = reference.get("models", {}).get(model_name)
            second = comparison.get("models", {}).get(model_name)
            if first is not None and second is not None:
                if self._supports(first) != self._supports(second):
                    raise ValueError(f"Per-class supports differ for {model_name}")
                if self._coverage(first) != self._coverage(second):
                    raise ValueError(f"Coverage counts differ for {model_name}")
        if not {"majority", "logistic_regression"}.issubset(reference.get("models", {})):
            raise ValueError("Reference metrics must contain majority and logistic_regression")
        if "logistic_regression" not in comparison.get("models", {}):
            raise ValueError("Comparison metrics must contain logistic_regression")

    @staticmethod
    def _config_for_comparison(config: dict) -> dict:
        result = dict(config)
        result.pop("class_weight", None)
        result.pop("output_dir", None)
        return result

    @staticmethod
    def _supports(metrics: dict) -> list[int]:
        return [int(metrics["classes"][str(label)]["support"]) for label in range(4)]

    @staticmethod
    def _coverage(metrics: dict) -> dict:
        coverage = metrics.get("coverage", {})
        return {
            key: coverage.get(key)
            for key in ("scored_epochs", "eligible_epochs", "label_bearing_epochs")
        }

    def _guard_output(self) -> None:
        out = self.output_dir.resolve()
        for raw in (self.reference_path.resolve(), self.comparison_path.resolve()):
            if out == raw or out in raw.parents:
                raise ValueError("Output directory cannot contain or replace a raw metrics input")

    def _plot_confusion(self, reference: dict, comparison: dict) -> dict:
        models = [
            ("Majority reference", reference["models"]["majority"]),
            ("Reference\nlogistic regression", reference["models"]["logistic_regression"]),
            ("Balanced\nlogistic regression", comparison["models"]["logistic_regression"]),
        ]
        labels = reference["classes"]
        class_names = [self._class_name(label) for label in labels]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        normalized = {}
        for ax, (title, metrics) in zip(axes, models):
            counts = np.asarray(metrics["confusion_matrix"], dtype=int)
            support = counts.sum(axis=1)
            rates = np.divide(
                counts,
                support[:, None],
                out=np.zeros_like(counts, dtype=float),
                where=support[:, None] != 0,
            )
            normalized[title] = rates.tolist()
            ax.imshow(rates, vmin=0, vmax=1, cmap="Blues")
            ax.set(
                title=title,
                xlabel="Predicted class",
                ylabel="True class",
                xticks=range(len(labels)),
                yticks=range(len(labels)),
                xticklabels=class_names,
                yticklabels=class_names,
            )
            for row in range(len(labels)):
                for col in range(len(labels)):
                    color = "white" if rates[row, col] >= 0.55 else "black"
                    ax.text(
                        col,
                        row,
                        f"{rates[row, col]:.0%}\n(n={counts[row, col]})",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=color,
                    )
        fig.suptitle(
            "True-row normalized confusion matrices (percent; raw counts shown)", fontsize=9
        )
        fig.savefig(self.output_dir / "confusion_matrices.png", dpi=150)
        plt.close(fig)
        return normalized

    def _plot_scores(self, reference: dict, comparison: dict) -> None:
        pairs = [
            ("Majority", reference["models"]["majority"]),
            ("Reference LR", reference["models"]["logistic_regression"]),
            ("Balanced LR", comparison["models"]["logistic_regression"]),
        ]
        classes = reference["classes"]
        class_names = [self._class_name(label) for label in classes]
        x = np.arange(len(classes))
        width = 0.24
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
        for axis, metric in zip(axes, ("recall", "f1")):
            for index, (name, model) in enumerate(pairs):
                values = [model["classes"][str(label)][metric] for label in classes]
                axis.bar(x + (index - 1) * width, values, width, label=name)
            axis.set(
                xticks=x,
                xticklabels=class_names,
                ylim=(0, 1),
                xlabel="Sleep stage",
                ylabel=metric.title(),
            )
            axis.legend(fontsize=6)
        fig.suptitle("Per-class recall and F1", fontsize=9)
        fig.savefig(self.output_dir / "per_class_scores.png", dpi=150)
        plt.close(fig)

    def _plot_coefficients(self) -> str | None:
        paths = [
            self.reference_path.parent / "models" / "logistic_regression.joblib",
            self.comparison_path.parent / "models" / "logistic_regression.joblib",
        ]
        if not all(path.is_file() for path in paths):
            return None
        pipelines = [joblib.load(path) for path in paths]
        matrices = []
        for pipeline in pipelines:
            model = pipeline.named_steps["logisticregression"]
            matrices.append(self._coefficient_grid(np.asarray(model.coef_)))
        metrics = self._read_metrics(self.reference_path)
        features = metrics.get("features", [])
        class_names = [self._class_name(label) for label in metrics["classes"]]
        max_abs = max(float(np.max(np.abs(values))) for values in matrices) or 1.0
        fig, axes = plt.subplots(
            1, 2, figsize=(9.6, max(2.6, len(features) * 0.36)), constrained_layout=True
        )
        for axis, values, title in zip(axes, matrices, ("Reference LR", "Balanced LR")):
            axis.imshow(
                values,
                aspect="auto",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            axis.set(
                title=title,
                xticks=range(values.shape[1]),
                xticklabels=class_names,
                yticks=range(len(features)),
                yticklabels=features,
            )
            axis.set_xlabel("Sleep stage")
            axis.tick_params(axis="x", labelrotation=25, labelsize=9)
        fig.colorbar(axes[0].images[0], ax=axes, label="Standardized coefficient")
        fig.suptitle(
            "Standardized logistic coefficients (associations, not causal feature importance)",
            fontsize=9,
        )
        fig.savefig(self.output_dir / "standardized_coefficients.png", dpi=150)
        plt.close(fig)
        return "standardized_coefficients.png"

    def _plot_timelines(self) -> str | None:
        reference_csv = self._prediction_file(self.reference_path.parent)
        comparison_csv = self._prediction_file(self.comparison_path.parent)
        if comparison_csv is None:
            return None
        datasets = {"Balanced": pd.read_csv(comparison_csv)}
        if reference_csv is not None:
            datasets["Reference"] = pd.read_csv(reference_csv)
            self._validate_prediction_alignment(datasets["Reference"], datasets["Balanced"])
        participants = sorted(datasets["Balanced"]["participant_id"].astype(str).unique())
        fig, axes = plt.subplots(
            len(participants),
            1,
            figsize=(8.2, max(2.5, 1.45 * len(participants))),
            squeeze=False,
            sharex=True,
            constrained_layout=True,
        )
        for axis, participant in zip(axes[:, 0], participants):
            self._timeline_axis(axis, participant, datasets)
        times = pd.to_numeric(datasets["Balanced"]["time"], errors="raise") / 3600
        axes[0, 0].set_xlim(float(times.min()), float(times.max()) + 30 / 3600)
        cmap, norm = self._stage_colors()
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(
            scalar,
            ax=axes[:, 0],
            ticks=range(4),
            boundaries=np.arange(-0.5, 4.5),
            label="Sleep stage",
            fraction=0.025,
            pad=0.015,
        ).set_ticklabels([self._class_name(label) for label in range(4)])
        fig.savefig(self.output_dir / "participant_timelines.png", dpi=150)
        plt.close(fig)
        return "participant_timelines.png"

    @staticmethod
    def _prediction_file(directory: Path) -> Path | None:
        return next(
            (
                candidate
                for candidate in (
                    directory / "validation-predictions.csv",
                    directory / "validation_predictions.csv",
                )
                if candidate.is_file()
            ),
            None,
        )

    @staticmethod
    def _timeline_axis(axis, participant: str, datasets: dict[str, pd.DataFrame]) -> None:
        predictions = {
            name: ExperimentComparisonReport._participant_rows(frame, participant, name)
            for name, frame in datasets.items()
        }
        lanes = [("Observed", predictions["Balanced"], "target")]
        if "Reference" in predictions:
            lanes.append(
                ("Reference LR", predictions["Reference"], "logistic_regression_prediction")
            )
        lanes.append(("Balanced LR", predictions["Balanced"], "logistic_regression_prediction"))
        for row, (_, frame, column) in enumerate(lanes):
            ExperimentComparisonReport._plot_ribbon(axis, frame, column, row)
        axis.set(
            title=f"Participant {participant}",
            ylabel="Prediction lane",
            xlabel="Time (hours)",
            yticks=range(len(lanes)),
            yticklabels=[lane[0] for lane in lanes],
            ylim=(-0.5, len(lanes) - 0.5),
        )

    @staticmethod
    def _stage_colors() -> tuple[ListedColormap, BoundaryNorm]:
        cmap = ListedColormap(["#455a64", "#4c9f70", "#38598c", "#b45f8c"])
        return cmap, BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)

    @staticmethod
    def _plot_ribbon(axis, frame: pd.DataFrame, column: str, row: int) -> None:
        colors = ["#455a64", "#4c9f70", "#38598c", "#b45f8c"]
        epoch_hours = 30 / 3600
        for time, stage in zip(frame["time_hours"], frame[column]):
            if pd.isna(stage):
                continue
            axis.add_patch(
                Rectangle(
                    (float(time), row - 0.34),
                    epoch_hours,
                    0.68,
                    facecolor=colors[int(stage)],
                    edgecolor="none",
                )
            )

    @staticmethod
    def _validate_prediction_alignment(reference: pd.DataFrame, comparison: pd.DataFrame) -> None:
        required = {"participant_id", "time", "target"}
        if not required.issubset(reference.columns) or not required.issubset(comparison.columns):
            raise ValueError("Prediction CSVs must contain participant_id, time, and target")
        keys = ["participant_id", "time"]
        left = reference[keys + ["target"]].copy()
        right = comparison[keys + ["target"]].copy()
        left["participant_id"] = left["participant_id"].astype(str)
        right["participant_id"] = right["participant_id"].astype(str)
        left = left.sort_values(keys).reset_index(drop=True)
        right = right.sort_values(keys).reset_index(drop=True)
        if left.duplicated(keys).any() or right.duplicated(keys).any() or not left.equals(right):
            raise ValueError("Reference and comparison prediction rows do not align exactly")

    @staticmethod
    def _participant_rows(frame: pd.DataFrame, participant: str, model_name: str) -> pd.DataFrame:
        pred_col = "logistic_regression_prediction"
        required = {"time", "participant_id", "target", pred_col}
        if not required.issubset(frame.columns):
            missing = sorted(required - set(frame.columns))
            raise ValueError(f"{model_name} prediction CSV missing columns: {missing}")
        selected = frame.loc[frame["participant_id"].astype(str) == participant].copy()
        selected["time_hours"] = pd.to_numeric(selected["time"], errors="coerce") / 3600
        return selected.dropna(subset=["time_hours"]).sort_values("time_hours")

    @staticmethod
    def _plot_series(
        axis, frame: pd.DataFrame, column: str, label: str, color: str, width: float
    ) -> None:
        time = frame["time_hours"].to_numpy(dtype=float)
        values = frame[column].to_numpy(dtype=float)
        breaks = np.r_[True, np.diff(time) > (30 / 3600 + 1e-9)]
        segments = np.split(np.arange(len(time)), np.flatnonzero(breaks)[1:])
        for index, segment in enumerate(segments):
            if len(segment):
                axis.plot(
                    time[segment],
                    values[segment],
                    color=color,
                    linewidth=width,
                    marker="o",
                    markersize=2,
                    label=label if index == 0 else None,
                )

    @staticmethod
    def _coefficient_grid(coefficients: np.ndarray) -> np.ndarray:
        """Orient class-by-feature coefficients for feature-row heatmaps."""
        return coefficients.T

    @staticmethod
    def _class_name(label: int) -> str:
        return {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}.get(int(label), str(label))

    def _summary(
        self,
        reference: dict,
        comparison: dict,
        matrices: dict,
        coefficients: str | None,
        timelines: str | None,
    ) -> dict:
        return {
            "classes": reference["classes"],
            "features": reference["features"],
            "reference": reference["models"],
            "comparison": comparison["models"],
            "configs": {
                "reference": reference.get("config", {}),
                "comparison": comparison.get("config", {}),
            },
            "split": reference.get("provenance", {}).get("split", {}),
            "packages": {
                "reference": reference.get("packages", {}),
                "comparison": comparison.get("packages", {}),
            },
            "optimizer": {
                "reference": reference.get("optimizer"),
                "comparison": comparison.get("optimizer"),
            },
            "input_metrics_sha256": {
                "reference": self._file_sha256(self.reference_path),
                "comparison": self._file_sha256(self.comparison_path),
            },
            "validation_participants": reference.get("provenance", {})
            .get("split", {})
            .get("validation", []),
            "model_summary": self._model_summary(reference, comparison),
            "normalized_confusion_matrices": matrices,
            "figures": {
                "confusion_matrices": "confusion_matrices.png",
                "per_class_scores": "per_class_scores.png",
                "coefficients": coefficients,
                "participant_timelines": timelines,
            },
        }

    @staticmethod
    def _model_summary(reference: dict, comparison: dict) -> dict:
        models = {
            "Majority reference": reference["models"]["majority"],
            "Reference logistic regression": reference["models"]["logistic_regression"],
            "Balanced logistic regression": comparison["models"]["logistic_regression"],
        }
        result = {}
        for name, metrics in models.items():
            coverage = metrics.get("coverage", {})
            scored = int(coverage.get("scored_epochs", 0))
            retained = int(coverage.get("eligible_epochs", scored))
            label_bearing = int(coverage.get("label_bearing_epochs", scored))
            confusion = np.asarray(metrics["confusion_matrix"], dtype=float)
            support = confusion.sum(axis=1)
            recalls = np.divide(
                np.diag(confusion), support, out=np.zeros_like(support), where=support > 0
            )
            balanced_accuracy = float(recalls[support > 0].mean()) if np.any(support > 0) else 0.0
            result[name] = {
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": metrics["macro_f1"],
                "rem_recall": metrics["classes"]["3"]["recall"],
                "scored_epochs": scored,
                "retained_epochs": retained,
                "label_bearing_epochs": label_bearing,
                "coverage": float(
                    coverage.get("fraction", scored / label_bearing if label_bearing else 0)
                ),
            }
        return result

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_readme(self, summary: dict) -> None:
        validation_count = len(summary["validation_participants"])
        lines = [
            "# Experiment comparison",
            "",
            "This report compares saved validation metrics. It does not fit models.",
            f"The validation split contains {validation_count} participants. "
            "Results may vary with the held-out people.",
            "",
            "## Summary",
            "",
            "| Run | Accuracy | Balanced accuracy | Macro F1 | REM recall | "
            "Retained / scored epochs | Coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, model in summary["model_summary"].items():
            lines.append(
                f"| {name} | {model['accuracy']:.3f} | {model['balanced_accuracy']:.3f} | "
                f"{model['macro_f1']:.3f} | {model['rem_recall']:.3f} | "
                f"{model['retained_epochs']} / {model['scored_epochs']} | {model['coverage']:.1%} |"
            )
        lines.extend(
            [
                "",
                "## Figures",
                "",
                "![True-row normalized confusion matrices](confusion_matrices.png)",
                "",
                "![Per-class recall and F1](per_class_scores.png)",
                "",
            ]
        )
        if summary["figures"]["coefficients"]:
            lines.extend(
                ["![Standardized logistic coefficients](standardized_coefficients.png)", ""]
            )
        if summary["figures"]["participant_timelines"]:
            lines.extend(["![Participant timelines](participant_timelines.png)", ""])
        lines.extend(
            [
                "Logistic coefficients show standardized associations, not causal importance.",
                "",
            ]
        )
        (self.output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report a saved baseline experiment comparison")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    import logging

    logger = logging.getLogger(__name__)
    result = ExperimentComparisonReport(args.reference, args.comparison, args.output_dir).run()
    logger.info("Experiment report saved: %s", result)


if __name__ == "__main__":
    main()
