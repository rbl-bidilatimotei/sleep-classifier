import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.paired_architecture_report import PairedArchitectureReport
from sleep_classifier.temporal_slice_report import EPOCH_KEYS, TemporalSliceReport


class AuditEvidenceLoader:
    """Align development-only prepared features and verified saved model predictions."""

    def __init__(
        self, reference_summary: Path, comparisons: dict[str, dict[str, Any]],
    ) -> None:
        """Locate completed grouped studies without reading participant observations.

        Args:
            reference_summary (Path): Completed reference grouped summary.
            comparisons (dict[str, dict[str, Any]]): Aliases with summary paths and paired policies.
        """
        self.reference_summary = reference_summary.resolve()
        self.comparisons = {}
        for alias, settings in comparisons.items():
            if (not isinstance(alias, str) or re.fullmatch(r"[a-z][a-z0-9_]*", alias) is None
                    or alias == "reference"):
                raise ValueError("Comparison aliases must be safe unique model names")
            if not isinstance(settings, dict) or set(settings) != {"summary", "comparison"}:
                raise ValueError("Each comparison requires summary and comparison settings")
            self.comparisons[alias] = {
                "summary": Path(settings["summary"]).resolve(),
                "comparison": settings["comparison"],
            }
        self.input_hashes: dict[str, str] = {}


    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read metadata and record its exact input digest.

        Args:
            path (Path): Metadata file within a selected study.
        """
        content = path.read_bytes()
        self.input_hashes[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Audit metadata must contain JSON objects")
        return value


    @staticmethod
    def _identifiers(values: Any, name: str) -> set[str]:
        """Require unique safe participant names in one declared partition.

        Args:
            values (Any): Proposed participant identifier list.
            name (str): Partition label used in error messages.
        """
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
                       for value in values) or len(values) != len(set(values))):
            raise ValueError(f"Audit {name} participant identifiers are invalid")
        return set(values)


    def _validate_plan(self, summary: dict[str, Any]) -> None:
        """Reject reserved-test membership before any prediction or feature read.

        Args:
            summary (dict[str, Any]): Grouped summary containing its frozen fold plan.
        """
        plan = summary["plan"]
        development = self._identifiers(plan["development"], "development")
        reserved = self._identifiers(plan["reserved_test"], "reserved test")
        if development & reserved or plan.get("independent_outer_test") is not False:
            raise ValueError("Audit scope must exclude the reserved test participants")
        folds = plan["folds"]
        if not isinstance(folds, list) or len(folds) < 2:
            raise ValueError("Audit requires the complete grouped development study")
        names = []
        validations = []
        for fold in folds:
            name = fold["name"]
            if not isinstance(name, str) or re.fullmatch(r"fold-\d+", name) is None:
                raise ValueError("Audit fold names must be safe numbered directories")
            train = self._identifiers(fold["train"], "fold training")
            validation = self._identifiers(fold["validation"], "fold validation")
            if train & validation or train | validation != development:
                raise ValueError("Each audit fold must partition the development participants")
            names.append(name)
            validations.extend(validation)
        if len(names) != len(set(names)) or len(validations) != len(set(validations)):
            raise ValueError("Audit folds and held-out participants must be unique")
        if set(validations) != development:
            raise ValueError("Audit validation folds must cover development exactly once")


    @staticmethod
    def _inside(path: Path, root: Path) -> Path:
        """Reject artifact escapes and symlinks before observation access.

        Args:
            path (Path): Expected artifact location under the grouped study.
            root (Path): Resolved grouped root.
        """
        absolute = path.absolute()
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or absolute != resolved:
            raise ValueError("Audit artifacts must remain inside their declared study")
        return resolved


    def _preflight_study(self, path: Path) -> dict[str, Any]:
        """Validate metadata, fold membership, and prepared paths before observations.

        Args:
            path (Path): Completed grouped summary path.
        """
        summary = self._read_json(path)
        self._validate_plan(summary)
        root = path.parent
        manifest = self._read_json(self._inside(root / "manifest.json", root))
        if manifest["plan"] != summary["plan"]:
            raise ValueError("Audit manifest and result plans differ")
        for fold in summary["plan"]["folds"]:
            directory = self._inside(root / fold["name"] / "features", root)
            provenance_path = self._inside(directory / "provenance.json", root)
            provenance = self._read_json(provenance_path)
            relative_provenance = str(provenance_path.relative_to(root))
            if (self.input_hashes[str(provenance_path)]
                    != manifest["derived_hashes"].get(relative_provenance)):
                raise ValueError("Audit prepared provenance checksum differs")
            split = provenance["split"]
            for partition in ("train", "validation"):
                if split.get(partition) != fold[partition]:
                    raise ValueError("Prepared audit participant assignment differs from the plan")
            if set(split.get("test", [])) != set(summary["plan"]["reserved_test"]):
                raise ValueError("Prepared reserved-test metadata differs from the plan")
            allowed = {str(provenance_path.relative_to(root))}
            for partition in ("train", "validation"):
                for participant in fold[partition]:
                    file = self._inside(directory / partition / f"{participant}.csv", root)
                    allowed.add(str(file.relative_to(root)))
            prefix = str(directory.relative_to(root)) + "/"
            recorded = {key for key in manifest["derived_hashes"] if key.startswith(prefix)}
            if recorded != allowed:
                raise ValueError("Audit feature manifest contains undeclared observation paths")
            for result in summary["results"]:
                metrics = Path(result["metrics_path"])
                metrics = metrics if metrics.is_absolute() else root / metrics
                self._inside(metrics, root)
                self._inside(metrics.parent / "validation_predictions.csv", root)
        return summary


    def _snapshot_features(self, path: Path, summary: dict[str, Any]) -> None:
        """Hash only declared development feature tables after metadata preflight.

        Args:
            path (Path): Grouped summary path.
            summary (dict[str, Any]): Preflighted grouped summary.
        """
        root = path.parent
        manifest = json.loads((root / "manifest.json").read_text())
        for fold in summary["plan"]["folds"]:
            for partition in ("train", "validation"):
                for participant in fold[partition]:
                    file = root / fold["name"] / "features" / partition / f"{participant}.csv"
                    digest = hashlib.sha256(file.read_bytes()).hexdigest()
                    key = str(file.relative_to(root))
                    if digest != manifest["derived_hashes"][key]:
                        raise ValueError("Audit development feature checksum differs")
                    self.input_hashes[str(file)] = digest


    def _prediction_rows(self, path: Path, summary: dict[str, Any], alias: str) -> pd.DataFrame:
        """Read verified fold predictions and optional normalized probability evidence.

        Args:
            path (Path): Grouped summary path.
            summary (dict[str, Any]): Preflighted grouped result metadata.
            alias (str): Model name used for distinct output columns.
        """
        frames = []
        for result in summary["results"]:
            metrics = Path(result["metrics_path"])
            metrics = metrics if metrics.is_absolute() else path.parent / metrics
            self._read_json(metrics)
            prediction_path = metrics.parent / "validation_predictions.csv"
            self.input_hashes[str(prediction_path)] = hashlib.sha256(
                prediction_path.read_bytes(),
            ).hexdigest()
            frame = PairedArchitectureReport._read_predictions(metrics.parent)
            frame = frame.rename(columns={"prediction": f"prediction_{alias}"})
            frame["fold"] = result["fold"]
            probabilities = pd.read_csv(prediction_path, dtype={"participant_id": str})
            columns = [f"probability_{stage}" for stage in range(4)]
            if any(column in probabilities for column in columns):
                frame = self._attach_probabilities(frame, probabilities, columns, alias)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        if result.duplicated(["participant_id", "time"]).any():
            raise ValueError("Audit prediction epochs repeat across folds")
        return result


    @staticmethod
    def _attach_probabilities(
        frame: pd.DataFrame, raw: pd.DataFrame, columns: list[str], alias: str,
    ) -> pd.DataFrame:
        """Validate probabilities without replacing missing or invalid evidence.

        Args:
            frame (pd.DataFrame): Validated label predictions in sorted epoch order.
            raw (pd.DataFrame): Original prediction file with probability columns.
            columns (list[str]): Four ordered probability column names.
            alias (str): Model name used in derived confidence columns.
        """
        if not set(columns).issubset(raw.columns):
            raise ValueError("Audit probability evidence must include all four classes")
        ordered = raw.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)
        if (len(ordered) != len(frame)
                or not np.array_equal(ordered.participant_id.astype(str), frame.participant_id)
                or any(not np.array_equal(ordered[key].to_numpy(dtype=float), frame[key])
                       for key in ("time", "target"))):
            raise ValueError("Audit probabilities must align with their saved epoch identities")
        values = ordered[columns].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
            raise ValueError("Audit probabilities must be finite normalized distributions")
        if not np.array_equal(values.argmax(axis=1), frame[f"prediction_{alias}"].to_numpy()):
            raise ValueError("Audit probabilities disagree with the saved predictions")
        result = frame.copy()
        result[f"p_true_{alias}"] = values[np.arange(len(values)), frame.target.to_numpy(int)]
        result[f"confidence_{alias}"] = values.max(axis=1)
        return result


    @staticmethod
    def _merge_epochs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """Require exact epoch identity and fold membership before combining models.

        Args:
            left (pd.DataFrame): Existing eligible audit rows.
            right (pd.DataFrame): Next model's out-of-fold predictions.
        """
        keys = EPOCH_KEYS + ["fold"]
        merged = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError("Audit features and predictions must have identical epoch populations")
        return merged.drop(columns="_merge")


    def load(self) -> dict[str, Any]:
        """Return aligned development observations, saved comparisons, and input digests."""
        self.input_hashes = {}
        reference = self._preflight_study(self.reference_summary)
        summaries = {alias: self._preflight_study(settings["summary"])
                     for alias, settings in self.comparisons.items()}
        reports = {"reference": PairedArchitectureReport(
            self.reference_summary, self.reference_summary,
        ).build()}
        for alias, settings in self.comparisons.items():
            reports[alias] = PairedArchitectureReport(
                self.reference_summary, settings["summary"], settings["comparison"],
            ).build()
        metadata, _ = TemporalSliceReport._load_metadata(self.reference_summary)
        assignment = {participant: fold["name"] for fold in reference["plan"]["folds"]
                      for participant in fold["validation"]}
        metadata["fold"] = metadata.participant_id.map(assignment)
        rows = self._merge_epochs(metadata, self._prediction_rows(
            self.reference_summary, reference, "reference",
        ))
        self._snapshot_features(self.reference_summary, reference)
        for alias, settings in self.comparisons.items():
            rows = self._merge_epochs(rows, self._prediction_rows(
                settings["summary"], summaries[alias], alias,
            ))
            self._snapshot_features(settings["summary"], summaries[alias])
        prediction_columns = [f"prediction_{alias}" for alias in ("reference", *self.comparisons)]
        rows["reference_correct"] = rows.prediction_reference.eq(rows.target)
        rows["persistent_to_light"] = (
            rows.target.isin([2, 3]) & rows[prediction_columns].eq(1).all(axis=1)
        )
        rows = rows.sort_values(["participant_id", "time"], kind="stable").reset_index(drop=True)
        return {
            "rows": rows, "plan": reference["plan"], "reports": reports,
            "input_hashes": dict(self.input_hashes),
            "participant_ids": list(reference["plan"]["development"]),
            "reserved_test_ids": list(reference["plan"]["reserved_test"]),
            "feature_columns": list(FEATURE_COLUMNS),
        }
