import argparse
import hashlib
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from sleep_classifier.audit_evidence import AuditEvidenceLoader
from sleep_classifier.audit_features import CausalAuditFeatures
from sleep_classifier.audit_motion_source import VerifiedMotionSource
from sleep_classifier.audit_visualization import AuditVisualization
from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
AUDIT_FEATURES = (*FEATURE_COLUMNS, "heart_rate_delta_past", "motion_vector_change_mean")


class AuditSettings:
    """Resolve the bounded audit settings without opening sensor observations."""

    def __init__(self, config_path: Path) -> None:
        """Validate paths and the two causal diagnostic definitions.

        Args:
            config_path (Path): YAML file containing frozen references and audit settings.
        """
        self.config_path = config_path.resolve()
        values = yaml.safe_load(self.config_path.read_text())
        expected = {
            "reference_summary", "comparisons", "raw_data_dir", "output_dir",
            "heart_history_epochs", "min_heart_history_epochs", "max_motion_pair_gap_seconds",
        }
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError("Audit configuration must contain exactly the documented settings")
        self.values = values
        self.reference_summary = self._path(values["reference_summary"])
        self.raw_data_dir = self._path(values["raw_data_dir"])
        self.output_dir = self._path(values["output_dir"])
        supplied = values["comparisons"]
        if not isinstance(supplied, dict) or not supplied:
            raise ValueError("Audit comparisons must be a nonempty mapping")
        self.comparisons: dict[str, dict[str, Any]] = {}
        for name, entry in supplied.items():
            if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                    or name == "reference" or not isinstance(entry, dict)
                    or set(entry) != {"summary", "comparison"}):
                raise ValueError("Comparison aliases and entries must use the documented schema")
            self.comparisons[name] = {
                "summary": self._path(entry["summary"]), "comparison": entry["comparison"],
            }
        self.features = CausalAuditFeatures(
            history_epochs=values["heart_history_epochs"],
            min_history_epochs=values["min_heart_history_epochs"],
            max_motion_pair_gap_seconds=values["max_motion_pair_gap_seconds"],
        )
        self._validate_output()


    def _path(self, value: Any) -> Path:
        """Resolve one required path relative to the configuration file.

        Args:
            value (Any): Nonempty YAML path string.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Audit paths must be nonempty strings")
        return (self.config_path.parent / value).resolve()


    def _validate_output(self) -> None:
        """Keep audit output separate from every immutable input directory."""
        protected = [self.raw_data_dir, self.reference_summary.parent]
        protected.extend(entry["summary"].parent for entry in self.comparisons.values())
        for directory in protected:
            if (self.output_dir.is_relative_to(directory)
                    or directory.is_relative_to(self.output_dir)):
                raise ValueError("Audit output must not contain or modify input directories")
        if self.config_path.is_relative_to(self.output_dir):
            raise ValueError("Audit output must not contain its configuration")


class CausalFeatureAudit:
    """Build a development-only audit snapshot and a portable evaluation view."""

    def __init__(self, config_path: Path) -> None:
        """Store the validated settings without loading participant observations.

        Args:
            config_path (Path): Audit configuration YAML file.
        """
        self.settings = AuditSettings(config_path)
        self.output_dir = self.settings.output_dir


    @staticmethod
    def _digest(path: Path) -> str:
        """Hash a small audit artifact or its configuration.

        Args:
            path (Path): File whose bytes define the audit snapshot.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()


    def plan(self) -> dict[str, Any]:
        """Describe development membership and read scope using saved metadata only."""
        summary = json.loads(self.settings.reference_summary.read_text())
        plan = summary["plan"]
        development = plan["development"]
        reserved = plan["reserved_test"]
        identifiers = development + reserved
        if (not development or len(set(identifiers)) != len(identifiers)
                or any(not isinstance(item, str) or not re.fullmatch(r"\d+", item)
                       for item in identifiers)):
            raise ValueError("Audit development and final-test membership must be disjoint")
        return {
            "development_participants": development, "reserved_test_participants": reserved,
            "development_count": len(development), "reserved_test_count": len(reserved),
            "recipes": ["reference", *self.settings.comparisons],
            "output_dir": str(self.output_dir), "model_training_performed": False,
            "test_observations_read": False,
            "raw_read_scope": "Motion observations for development participants only",
            "candidate_features_integrated": False,
        }


    def run(self) -> Path:
        """Create one audit snapshot without fitting or updating a prediction model."""
        if self.output_dir.exists():
            raise FileExistsError("Audit output exists; load it or choose a fresh directory")
        started = perf_counter()
        self.plan()
        evidence = AuditEvidenceLoader(
            self.settings.reference_summary, self.settings.comparisons,
        ).load()
        rows = self.settings.features.add_heart_history(evidence["rows"])
        source = VerifiedMotionSource(
            self.settings.raw_data_dir, evidence["participant_ids"], evidence["reserved_test_ids"],
        )
        rows = self._add_motion(rows, source)
        summary = self._summarize(rows, evidence)
        summary["elapsed_seconds_before_render"] = perf_counter() - started
        summary["raw_files_read"] = source.accessed_files
        summary["source_artifact_hashes"] = evidence["input_hashes"]
        self._write_snapshot(rows, summary)
        LOGGER.info("Audit completed without training: targets=%d elapsed_seconds=%.1f output=%s",
                    len(rows), perf_counter() - started, self.output_dir)
        return self.output_dir


    def _add_motion(self, rows: pd.DataFrame, source: VerifiedMotionSource) -> pd.DataFrame:
        """Add diagnostic motion changes one allowed participant at a time.

        Args:
            rows (pd.DataFrame): Matched development targets and causal heart-rate diagnostics.
            source (VerifiedMotionSource): Reader enforcing the development allowlist.
        """
        frames = []
        participants = sorted(rows.participant_id.unique())
        for index, participant in enumerate(participants, start=1):
            started = perf_counter()
            selected = rows.loc[rows.participant_id == participant].copy()
            motion = source.load(participant)
            values = self.settings.features.motion_summaries(motion, selected)
            frames.append(selected.merge(values, on="time", how="left", validate="one_to_one"))
            LOGGER.info("Audited motion %d/%d participant=%s samples=%d targets=%d seconds=%.1f",
                        index, len(participants), participant, len(motion), len(selected),
                        perf_counter() - started)
            del motion
        result = pd.concat(frames, ignore_index=True)
        if len(result) != len(rows):
            raise ValueError("Diagnostic features must not change the eligible target population")
        return result


    def _summarize(self, rows: pd.DataFrame, evidence: dict[str, Any]) -> dict[str, Any]:
        """Describe persistent errors and diagnostic support without selecting a model.

        Args:
            rows (pd.DataFrame): Aligned development evidence with causal diagnostic columns.
            evidence (dict[str, Any]): Verified plans, recipe reports, and input hashes.
        """
        persistent = rows.loc[rows.persistent_to_light]
        counts = [{"stage": stage, "target": target,
                   "persistent_to_light": int((persistent.target == target).sum()),
                   "reference_to_light": int(((rows.target == target)
                                               & (rows.prediction_reference == 1)).sum()),
                   "support": int((rows.target == target).sum())}
                  for target, stage in ((2, "Deep"), (3, "REM"))]
        availability = [{"feature": feature, "available_targets": int(rows[feature].notna().sum()),
                         "total_targets": len(rows)}
                        for feature in AUDIT_FEATURES]
        return {
            **self.plan(), "schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
            "participants": len(evidence["participant_ids"]),
            "targets": len(rows), "persistent_counts": counts,
            "candidate_availability": availability,
            "persistent_definition": "True Deep/REM with Light predicted by every included recipe",
            "scope": "Post-hoc development diagnostics; correlated recipes and reused participants",
            "feature_definitions": {
                "heart_rate_delta_past": "Current mean minus strictly previous eligible-epoch mean",
                "motion_vector_change_mean": (
                    "Mean axis-vector change over admissible within-epoch pairs"
                ),
                "units": "Heart rate in source BPM; motion changes in source acceleration units",
                "gap_policy": self.settings.values["max_motion_pair_gap_seconds"],
                "history_epochs": self.settings.values["heart_history_epochs"],
                "minimum_past_epochs": self.settings.values["min_heart_history_epochs"],
            },
            "training_fold_stage_summary": self._training_stage_summary(rows, evidence["plan"]),
            "matched_error_summary": self._matched_error_summary(rows),
        }


    @staticmethod
    def _training_stage_summary(rows: pd.DataFrame, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Summarize participant-stage medians using each fold's training participants only.

        Args:
            rows (pd.DataFrame): Raw development diagnostic features, without fitted scaling.
            plan (dict[str, Any]): Frozen grouped development folds.
        """
        summaries = []
        for fold in plan["folds"]:
            training = rows.loc[rows.participant_id.isin(fold["train"])]
            for feature in AUDIT_FEATURES:
                for target, group in training.groupby("target"):
                    values = group.groupby("participant_id")[feature].median().dropna()
                    summaries.append({
                        "fold": fold["name"], "target": int(target), "feature": feature,
                        "participants": len(values),
                        "available_epochs": int(group[feature].notna().sum()),
                        "median": float(values.median()) if len(values) else None,
                        "q25": float(values.quantile(.25)) if len(values) else None,
                        "q75": float(values.quantile(.75)) if len(values) else None,
                    })
        return summaries


    @staticmethod
    def _matched_error_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
        """Compare persistent and reference-correct targets within the same participant and stage.

        Args:
            rows (pd.DataFrame): Post-hoc development predictions and diagnostic features.
        """
        summaries = []
        for (participant, target), group in rows.loc[rows.target.isin([2, 3])].groupby(
            ["participant_id", "target"]
        ):
            for feature in AUDIT_FEATURES:
                errors = group.loc[group.persistent_to_light, feature].dropna()
                correct = group.loc[group.reference_correct, feature].dropna()
                summaries.append({
                    "participant_id": participant, "target": int(target), "feature": feature,
                    "persistent_count": len(errors), "correct_count": len(correct),
                    "persistent_median": float(errors.median()) if len(errors) else None,
                    "correct_median": float(correct.median()) if len(correct) else None,
                    "median_difference": float(errors.median() - correct.median())
                    if len(errors) and len(correct) else None,
                })
        return summaries


    def _write_snapshot(self, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
        """Publish data, figures, and a self-contained dashboard after successful analysis.

        Args:
            rows (pd.DataFrame): Audited eligible development rows.
            summary (dict[str, Any]): Diagnostic definitions, supports, and provenance.
        """
        import matplotlib.pyplot as plt

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".causal-audit-", dir=self.output_dir.parent) as temporary:
            directory = Path(temporary)
            rows.to_csv(directory / "rows.csv", index=False)
            encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
            (directory / "summary.json").write_text(encoded)
            view = AuditVisualization(rows, summary)
            view.write_html(directory / "dashboard.html")
            for name, figure in (("feature-distributions.png", view.plot_distributions()),
                                 ("persistent-errors.png", view.plot_persistent_errors())):
                figure.savefig(directory / name, dpi=140)
                plt.close(figure)
            manifest = {
                "schema_version": 1, "config_sha256": self._digest(self.settings.config_path),
                "outputs": {path.name: self._digest(path) for path in directory.iterdir()},
                "implementation_hashes": {
                    path.name: self._digest(path)
                    for path in [Path(__file__), *Path(__file__).parent.glob("audit_*.py"),
                                 Path(__file__).parent / "templates/causal_audit.html"]
                },
            }
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            shutil.move(str(directory), self.output_dir)


    def load_report(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read a verified snapshot without rereading raw sensor observations."""
        manifest = json.loads((self.output_dir / "manifest.json").read_text())
        expected = {"rows.csv", "summary.json", "dashboard.html",
                    "feature-distributions.png", "persistent-errors.png"}
        if (manifest.get("schema_version") != 1 or set(manifest.get("outputs", {})) != expected
                or manifest["config_sha256"] != self._digest(self.settings.config_path)):
            raise ValueError("Audit snapshot configuration or output inventory differs")
        for filename, digest in manifest["outputs"].items():
            path = self.output_dir / filename
            if path.resolve().parent != self.output_dir or self._digest(path) != digest:
                raise ValueError("Audit snapshot output checksum differs")
        summary = json.loads((self.output_dir / "summary.json").read_text())
        rows = pd.read_csv(self.output_dir / "rows.csv", dtype={"participant_id": str})
        if (len(rows) != summary["targets"] or summary["test_observations_read"] is not False
                or summary["model_training_performed"] is not False
                or not set(rows.participant_id).issubset(summary["development_participants"])
                or set(rows.participant_id) & set(summary["reserved_test_participants"])):
            raise ValueError("Audit snapshot violates its development-only scope")
        return rows, summary


def main() -> None:
    """Plan or run a development-only feature audit without a training command."""
    parser = argparse.ArgumentParser(
        description="Audit causal features and saved prediction errors",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/causal-feature-audit.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_logging()
    audit = CausalFeatureAudit(args.config)
    if args.plan:
        LOGGER.info("Audit plan: %s", json.dumps(audit.plan(), sort_keys=True))
    else:
        LOGGER.info("Audit snapshot saved: %s", audit.run())


if __name__ == "__main__":
    main()
