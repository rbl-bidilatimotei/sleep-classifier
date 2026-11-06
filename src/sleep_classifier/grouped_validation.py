import argparse
import copy
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold

from sleep_classifier.baseline import CLASS_LABELS, PreparedFeatureDataset
from sleep_classifier.gru_training import GRUExperiment, TrainingConfig
from sleep_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


class GroupedValidationExperiment:
    """Prepare and run participant-disjoint development validation folds."""

    def __init__(self, config_path: Path) -> None:
        """Initialize the grouped validation experiment with configuration paths.

        Args:
            config_path (Path): Path to the YAML configuration file for the experiment.

        Raises:
            ValueError: If the configuration is invalid or incompatible with grouped validation.
        """
        self.config_path = config_path.resolve()
        self.settings = yaml.safe_load(self.config_path.read_text())
        self.prepare_path = (
            self.config_path.parent / self.settings["source_prepare_config"]
        ).resolve()
        self.model_path = (self.config_path.parent / self.settings["base_model_config"]).resolve()
        source_settings = yaml.safe_load(self.prepare_path.read_text())
        self.source_dir = (self.prepare_path.parent / source_settings["output_dir"]).resolve()
        self.output_dir = (self.config_path.parent / self.settings["output_dir"]).resolve()
        self.model_settings = yaml.safe_load(self.model_path.read_text())

        TrainingConfig(self.model_settings)
        if self.model_settings.get("optimizer_agent"):
            raise ValueError("Grouped validation requires the plain Adam reference recipe")

        self._validate_settings()


    def _validate_settings(self) -> None:
        """Reject invalid fold counts and split seeds before reading observations.

        Raises:
            ValueError: If the fold count or split seed is invalid, or if the output directory
                conflicts with the source directory.
        """
        folds = self.settings["folds"]
        seed = self.settings["split_seed"]

        if type(folds) is not int or folds < 2:
            raise ValueError("folds must be an integer of at least two")
        if type(seed) is not int or seed < 0:
            raise ValueError("split_seed must be a nonnegative integer")
        if self.output_dir == self.source_dir or self.source_dir in self.output_dir.parents:
            raise ValueError("Grouped outputs must be separate from source features")


    def _source_provenance(self) -> dict[str, Any]:
        """Read source metadata and validate development membership without observations.

        Returns:
            dict[str, Any]: Source provenance metadata.

        Raises:
            ValueError: If the source metadata is invalid or inconsistent.
        """
        provenance = json.loads((self.source_dir / "provenance.json").read_text())
        split = provenance["split"]
        groups = [split[name] for name in ("train", "validation", "test")]
        if any(not isinstance(group, list) for group in groups):
            raise ValueError("Source split partitions must be lists")

        identifiers = [participant for group in groups for participant in group]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Source participant partitions must be unique and disjoint")

        if any(
            not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
            for value in identifiers
        ):
            raise ValueError("Source participant identifiers are invalid")

        if not split["train"] or not split["validation"]:
            raise ValueError("Source development partitions must be nonempty")
        if provenance.get("scaling_fitted") is not False:
            raise ValueError("Source features must not contain fitted scaling")
        if provenance.get("test_observations_read") is not False:
            raise ValueError("Source preparation must exclude final test observations")

        return provenance


    def plan(self) -> dict[str, Any]:
        """Assign each development participant to one held-out fold without training.

        Returns:
            dict[str, Any]: Cross-validation plan for development participants.
        """
        provenance = self._source_provenance()
        split = provenance["split"]
        participants = sorted(split["train"] + split["validation"])
        if self.settings["folds"] > len(participants):
            raise ValueError("folds cannot exceed the development participant count")

        splitter = KFold(
            n_splits=self.settings["folds"], shuffle=True, random_state=self.settings["split_seed"]
        )
        folds = [
            {
                "name": f"fold-{index:02d}",
                "train": [participants[position] for position in training],
                "validation": [participants[position] for position in validation],
            }
            for index, (training, validation) in enumerate(splitter.split(participants), start=1)
        ]

        return {
            "folds": folds,
            "development": participants,
            "reserved_test": split["test"],
            "split_seed": self.settings["split_seed"],
            "training_seed": self.model_settings["seed"],
            "maximum_epochs": self.settings["folds"] * self.model_settings["epochs"],
            "checkpoint_selection": "best macro-F1 on each development validation fold",
            "independent_outer_test": False,
        }


    @staticmethod
    def _digest(path: Path) -> str:
        """Hash one file without interpreting its content.

        Args:
            path (Path): Configuration or derived development file.

        Returns:
            str: SHA-256 hash of the file content.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()


    def _source_files(self, provenance: dict[str, Any]) -> dict[str, Path]:
        """List only immutable configuration and development feature inputs.

        Args:
            provenance (dict[str, Any]): Validated source preparation metadata.

        Returns:
            dict[str, Path]: Mapping from logical file names to their paths.
        """
        files = {
            "experiment_config": self.config_path,
            "prepare_config": self.prepare_path,
            "model_config": self.model_path,
            "provenance": self.source_dir / "provenance.json",
        }

        for partition in ("train", "validation"):
            for participant in provenance["split"][partition]:
                files[f"{partition}/{participant}.csv"] = (
                    self.source_dir / partition / f"{participant}.csv"
                )

        return files


    def prepare(self) -> Path:
        """Copy validated development features into frozen fold-specific directories.

        Returns:
            Path: Path to the manifest JSON file.
        """
        plan = self.plan()

        dataset = PreparedFeatureDataset(self.source_dir).load()
        provenance = dataset["provenance"]

        source_files = self._source_files(provenance)
        source_hashes = {name: self._digest(path) for name, path in source_files.items()}

        self.output_dir.mkdir(parents=True, exist_ok=False)

        device = str(TrainingConfig(self.model_settings).device())
        for fold in plan["folds"]:
            self._prepare_fold(fold, provenance, device)

        manifest = {
            "plan": plan,
            "device": device,
            "source_hashes": source_hashes,
            "derived_hashes": self._derived_hashes(plan),
        }

        self._write_json(self.output_dir / "manifest.json", manifest)
        LOGGER.info(
            "Prepared %d development folds without training: %s",
            len(plan["folds"]),
            self.output_dir,
        )

        return self.output_dir / "manifest.json"


    def _prepare_fold(
        self,
        fold: dict[str, list[str]],
        provenance: dict[str, Any],
        device: str
    ) -> None:
        """Write full participant tables and corrected fold membership metadata.

        Args:
            fold (dict[str, list[str]]): Disjoint training and validation participant lists.
            provenance (dict[str, Any]): Original feature preparation metadata.
            device (str): Fixed device for every fold.
        """
        directory = self.output_dir / fold["name"]
        features = directory / "features"
        original_partition = {
            participant: partition
            for partition in ("train", "validation")
            for participant in provenance["split"][partition]
        }

        original_meta = {entry["participant_id"]: entry for entry in provenance["participants"]}
        summaries = []

        for partition in ("train", "validation"):
            (features / partition).mkdir(parents=True)
            for participant in fold[partition]:
                source = self.source_dir / original_partition[participant] / f"{participant}.csv"
                shutil.copyfile(source, features / partition / source.name)
                summaries.append(original_meta[participant] | {"partition": partition})

        derived = copy.deepcopy(provenance)
        derived["split"].update(
            train=fold["train"],
            validation=fold["validation"],
            validation_count=len(fold["validation"]),
        )
        derived["participants"] = summaries
        derived["development_fold"] = fold["name"]

        self._write_json(features / "provenance.json", derived)

        (directory / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "features"}))

        recipe = self.model_settings | {
            "prepare_config": "prepare.yaml",
            "output_dir": "results",
            "device": device,
        }

        (directory / "gru.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))


    def _derived_hashes(self, plan: dict[str, Any]) -> dict[str, str]:
        """Hash prepared files while excluding later model outputs.

        Args:
            plan (dict[str, Any]): Saved fold definitions.

        Returns:
            dict[str, str]: Hashes of derived files.
        """
        files = []
        for fold in plan["folds"]:
            directory = self.output_dir / fold["name"]
            files.extend(directory.glob("*.yaml"))
            files.extend((directory / "features").rglob("*.csv"))
            files.append(directory / "features/provenance.json")

        return {
            str(path.relative_to(self.output_dir)): self._digest(path) for path in sorted(files)
        }


    def _validate_manifest(self) -> dict[str, Any]:
        """Reject changed source files, fold assignments, or prepared fold files.

        Returns:
            dict[str, Any]: Validated manifest.
        """
        manifest = json.loads((self.output_dir / "manifest.json").read_text())
        provenance = self._source_provenance()
        expected = {
            name: self._digest(path) for name, path in self._source_files(provenance).items()
        }

        if manifest["source_hashes"] != expected or manifest["plan"] != self.plan():
            raise ValueError("Source settings or development features changed after preparation")
        if manifest["derived_hashes"] != self._derived_hashes(manifest["plan"]):
            raise ValueError("Prepared fold files changed after preparation")

        return manifest

    def run(self) -> Path:
        """Train fresh models for prepared folds and save development validation scores.

        Returns:
            Path: Path to the saved summary file.
        """
        manifest = self._validate_manifest()
        plan = manifest["plan"]
        if any((self.output_dir / fold["name"] / "results").exists() for fold in plan["folds"]):
            raise FileExistsError(
                "Fold results already exist; use a fresh grouped output directory"
            )
        if (self.output_dir / "summary.json").exists():
            raise FileExistsError("Grouped summary already exists; use a fresh output directory")

        summary = {"status": "running", "plan": plan, "results": []}
        self._save_summary(summary)

        for index, fold in enumerate(plan["folds"], start=1):
            LOGGER.info(
                "Starting development fold %d/%d: %s", index, len(plan["folds"]), fold["name"]
            )
            try:
                result = self._run_fold(fold)
            except Exception:
                summary.update(status="failed", failed_fold=fold["name"])
                self._save_summary(summary)
                raise
            summary["results"].append(result)
            self._save_summary(summary)

        try:
            aggregate = self._aggregate(summary["results"])
        except Exception:
            summary.update(status="failed", failed_stage="aggregation")
            self._save_summary(summary)
            raise

        summary.update(status="complete", aggregate=aggregate)

        return self._save_summary(summary)


    def _run_fold(self, fold: dict[str, Any]) -> dict[str, Any]:
        """Run a fresh training instance and retain its complete artifact hash.

        Args:
            fold (dict[str, Any]): One planned participant partition.

        Returns:
            dict[str, Any]: Complete fold results including metrics and artifact locations.
        """
        metrics_path = GRUExperiment(self.output_dir / fold["name"] / "gru.yaml").run()
        metrics = json.loads(metrics_path.read_text())

        return {
            "fold": fold["name"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": self._digest(metrics_path),
            **{
                key: metrics[key]
                for key in (
                    "accuracy",
                    "macro_f1",
                    "best_epoch",
                    "classes",
                    "coverage",
                    "participants",
                )
            },
        }


    def _aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Pool one validation prediction per retained development epoch.

        Args:
            results (list[dict[str, Any]]): Completed fold scores and artifact locations.

        Returns:
            dict[str, Any]: Aggregated validation results including pooled metrics and participant
                coverage.
        """
        frames = []
        participants = {}

        for result in results:
            path = Path(result["metrics_path"]).parent / "validation_predictions.csv"
            frame = pd.read_csv(path, dtype={"participant_id": str})

            if "prediction" not in frame.columns and "gru_prediction" in frame.columns:
                frame = frame.rename(columns={"gru_prediction": "prediction"})
            if "prediction" not in frame.columns:
                raise ValueError("Fold predictions require a prediction column")

            frame["fold"] = result["fold"]
            expected = set(result["participants"])

            if set(frame.participant_id) != expected:
                raise ValueError("Fold prediction participant IDs differ from the measured fold")
            if set(participants) & expected:
                raise ValueError("Validation participants repeat across folds")
            participants.update(result["participants"])
            frames.append(frame)

        if set(participants) != set(self.plan()["development"]):
            raise ValueError("Validation predictions must cover all development participants")

        pooled = pd.concat(frames, ignore_index=True)
        for column in ("target", "prediction"):
            if not pooled[column].isin(CLASS_LABELS).all():
                raise ValueError("Validation prediction labels must be in 0..3")

        for participant, metadata in participants.items():
            if int((pooled.participant_id == participant).sum()) != metadata["eligible_epochs"]:
                raise ValueError("Validation prediction row counts differ from eligible counts")
        if pooled.duplicated(["participant_id", "time"]).any():
            raise ValueError("Out-of-fold prediction epochs must be unique")

        exported = pooled
        if self.model_settings.get("model_type", "gru") == "gru":
            exported = pooled.rename(columns={"prediction": "gru_prediction"})

        exported.to_csv(self.output_dir / "development_predictions.csv", index=False)
        precision, recall, f1, support = precision_recall_fscore_support(
            pooled.target, pooled.prediction, labels=CLASS_LABELS, zero_division=0
        )
        scores = [result["macro_f1"] for result in results]

        return {
            "mean_fold_macro_f1": mean(scores),
            "sample_std_fold_macro_f1": stdev(scores),
            "pooled_accuracy": accuracy_score(pooled.target, pooled.prediction),
            "pooled_macro_f1": float(f1.mean()),
            "confusion_matrix": confusion_matrix(
                pooled.target, pooled.prediction, labels=CLASS_LABELS
            ).tolist(),
            "classes": {
                str(label): {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(CLASS_LABELS)
            },
            "participants": participants,
            "validation_participants": len(participants),
            "eligible_epochs": len(pooled),
            "independent_outer_test": False,
        }


    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        """Replace a JSON artifact atomically.

        Args:
            path (Path): Destination metadata file.
            value (dict[str, Any]): Serializable metadata.
        """
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)


    def _save_summary(self, summary: dict[str, Any]) -> Path:
        """Preserve completed fold metrics after each run.

        Args:
            summary (dict[str, Any]): Current experiment state and completed results.

        Returns:
            Path: Location of the saved summary JSON file.
        """
        path = self.output_dir / "summary.json"
        self._write_json(path, summary)
        return path


def main() -> None:
    """Plan, prepare, or explicitly train the grouped development experiment."""
    parser = argparse.ArgumentParser(description="Grouped participant development validation")
    parser.add_argument("--config", type=Path, required=True)

    operation = parser.add_mutually_exclusive_group(required=True)

    operation.add_argument("--plan", action="store_true")
    operation.add_argument("--prepare", action="store_true")
    operation.add_argument("--train", action="store_true")

    args = parser.parse_args()
    configure_logging()

    experiment = GroupedValidationExperiment(args.config)
    if args.plan:
        LOGGER.info("Grouped development plan:\n%s", json.dumps(experiment.plan(), indent=2))
    elif args.prepare:
        LOGGER.info("Prepared manifest: %s", experiment.prepare())
    else:
        LOGGER.info("Grouped development summary: %s", experiment.run())


if __name__ == "__main__":
    main()
