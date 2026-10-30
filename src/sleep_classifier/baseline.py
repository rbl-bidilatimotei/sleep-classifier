import argparse
import json
import platform
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sleep_classifier.features import FEATURE_COLUMNS

CLASS_LABELS = (0, 1, 2, 3)


class PreparedFeatureDataset:
    """Read eligible train and validation rows from prepared participant tables."""

    def __init__(self, prepared_dir: Path) -> None:
        """Store the prepared feature directory."""
        self.prepared_dir = prepared_dir

    def load(self) -> dict:
        """Load only provenance-listed train and validation participants."""
        provenance_path = self.prepared_dir / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Prepared provenance is missing: {provenance_path}")
        provenance = json.loads(provenance_path.read_text())
        feature_columns = provenance.get("feature_columns")
        if feature_columns is not None and tuple(feature_columns) != FEATURE_COLUMNS:
            raise ValueError("Prepared provenance feature columns do not match the baseline schema")
        split = provenance.get("split", {})
        train_ids = split.get("train")
        validation_ids = split.get("validation")
        if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
            raise ValueError("Provenance must contain train and validation participant lists")
        if set(train_ids) & set(validation_ids):
            raise ValueError("Train and validation participants must be disjoint")
        test_ids = set(split.get("test", []))
        if (set(train_ids) | set(validation_ids)) & test_ids:
            raise ValueError("Development participants overlap the test partition")
        self._validate_participant_ids(train_ids + validation_ids + list(test_ids))
        participant_meta = {
            (item["partition"], item["participant_id"]): item
            for item in provenance.get("participants", [])
        }
        loaded = {"provenance": provenance, "partitions": {}}
        for partition, participant_ids in (("train", train_ids), ("validation", validation_ids)):
            loaded["partitions"][partition] = self._load_partition(
                partition, participant_ids, participant_meta
            )
        self._validate_rows(loaded["partitions"])
        return loaded

    def _validate_participant_ids(self, participant_ids: list[str]) -> None:
        """Reject unsafe or duplicate participant identifiers."""
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Participant IDs must be unique across partitions")
        valid_id = re.compile(r"[A-Za-z0-9_-]+")
        if any(
            not isinstance(value, str) or valid_id.fullmatch(value) is None
            for value in participant_ids
        ):
            raise ValueError(
                "Participant IDs must contain only letters, digits, underscores, or hyphens"
            )

    def _load_partition(
        self, partition: str, participant_ids: list[str], metadata: dict
    ) -> pd.DataFrame:
        """Load and reconcile all participant tables for one split."""
        frames = []
        for participant_id in participant_ids:
            raw = self._read_participant(partition, participant_id)
            participant_meta = metadata.get((partition, participant_id))
            if participant_meta is None:
                raise ValueError(
                    f"Provenance lacks participant counts for {partition}/{participant_id}"
                )
            eligible_count = int(participant_meta["eligible_epochs"])
            scored_count = int((raw["target"] != -1).sum())
            frame = raw.loc[raw["eligible"].astype(bool) & (raw["target"] != -1)].copy()
            if len(frame) != eligible_count:
                raise ValueError(
                    f"Eligible epoch count mismatch for {partition}/{participant_id}: "
                    f"provenance={eligible_count}, table={len(frame)}"
                )
            if frame.empty:
                raise ValueError(f"No eligible rows for {partition}/{participant_id}")
            frame["participant_id"] = participant_id
            frame["eligible_epochs"] = eligible_count
            frame["scored_epochs"] = scored_count
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else self._empty()

    def _read_participant(self, partition: str, participant_id: str) -> pd.DataFrame:
        """Read one participant table and require the expected schema."""
        path = self.prepared_dir / partition / f"{participant_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared participant table is missing: {path}")
        frame = pd.read_csv(path)
        required = set(FEATURE_COLUMNS) | {"target", "eligible"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Prepared table {path} is missing columns: {sorted(missing)}")
        return frame

    def _empty(self) -> pd.DataFrame:
        """Create the expected empty partition schema."""
        return pd.DataFrame(
            columns=[
                *FEATURE_COLUMNS,
                "target",
                "participant_id",
                "eligible_epochs",
                "scored_epochs",
            ]
        )

    def _validate_rows(self, partitions: dict[str, pd.DataFrame]) -> None:
        """Reject invalid targets and non-finite feature values."""
        for partition, frame in partitions.items():
            if frame.empty:
                raise ValueError(f"Partition has no eligible labeled rows: {partition}")
            features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
            raw_targets = frame["target"].to_numpy(dtype=float)
            if not np.isfinite(features).all():
                raise ValueError(f"Feature values must be finite in {partition}")
            if not np.isfinite(raw_targets).all() or not np.isin(raw_targets, CLASS_LABELS).all():
                raise ValueError(f"Targets must be in 0..3 in {partition}")


class BaselineExperiment:
    """Fit majority and scaled logistic-regression baselines on train participants."""

    def __init__(self, config_path: Path) -> None:
        """Store the baseline configuration path."""
        self.config_path = config_path

    def run(self) -> Path:
        """Fit on training participants and evaluate validation participants."""
        settings, prepared_dir, output_dir = self._load_settings()
        dataset = PreparedFeatureDataset(prepared_dir).load()
        pipeline, majority = self._fit_models(dataset["partitions"]["train"], settings)
        return self._save_results(settings, dataset, pipeline, majority, output_dir)

    def _load_settings(self) -> tuple[dict, Path, Path]:
        """Read baseline and preparation paths from the configuration files."""
        settings = yaml.safe_load(self.config_path.read_text())
        prepare_path = self.config_path.parent / settings["prepare_config"]
        prepare_settings = yaml.safe_load(prepare_path.read_text())
        prepared_dir = (prepare_path.parent / prepare_settings["output_dir"]).resolve()
        output_dir = (self.config_path.parent / settings["output_dir"]).resolve()
        return settings, prepared_dir, output_dir

    def _fit_models(self, train: pd.DataFrame, settings: dict):
        """Fit both classifiers using training rows only."""
        x_train = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train["target"].to_numpy(dtype=int)
        seed = int(settings["seed"])
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=int(settings["max_iter"]),
                C=float(settings["inverse_regularization"]),
                random_state=seed,
            ),
        )
        pipeline.fit(x_train, y_train)
        majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
        return pipeline, majority

    def _save_results(
        self, settings: dict, dataset: dict, pipeline, majority, output_dir: Path
    ) -> Path:
        """Persist fitted estimators and validation metrics."""
        validation = dataset["partitions"]["validation"]
        x_validation = validation.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y_validation = validation["target"].to_numpy(dtype=int)
        output_dir.mkdir(parents=True, exist_ok=True)
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, models_dir / "logistic_regression.joblib")
        joblib.dump(majority, models_dir / "majority.joblib")
        result = {
            "config": settings,
            "features": list(FEATURE_COLUMNS),
            "classes": list(CLASS_LABELS),
            "packages": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "provenance": dataset["provenance"],
            "models": {
                "majority": self._metrics(majority.predict(x_validation), y_validation, validation),
                "logistic_regression": self._metrics(
                    pipeline.predict(x_validation), y_validation, validation
                ),
            },
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2) + "\n")
        return metrics_path

    def _metrics(self, predictions: np.ndarray, targets: np.ndarray, frame: pd.DataFrame) -> dict:
        """Calculate aggregate and per-participant validation metrics."""
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=CLASS_LABELS, zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(np.mean(f1)),
            "classes": {
                str(label): {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(CLASS_LABELS)
            },
            "confusion_matrix": confusion_matrix(
                targets, predictions, labels=CLASS_LABELS
            ).tolist(),
            "coverage": self._coverage(frame),
            "participants": self._participant_metrics(frame, predictions),
        }

    def _coverage(self, frame: pd.DataFrame) -> dict:
        """Report retained eligible rows over label-bearing prepared rows."""
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
        """Calculate participant metrics with participant-specific coverage."""
        results = {}
        for participant_id, indexes in frame.groupby("participant_id").indices.items():
            selected = np.asarray(indexes, dtype=int)
            targets = frame.iloc[selected]["target"].to_numpy(dtype=int)
            predicted = predictions[selected]
            _, _, f1, _ = precision_recall_fscore_support(
                targets, predicted, labels=CLASS_LABELS, zero_division=0
            )
            eligible_count = int(frame.iloc[selected]["eligible_epochs"].iloc[0])
            scored_count = int(frame.iloc[selected]["scored_epochs"].iloc[0])
            results[str(participant_id)] = {
                "scored_epochs": scored_count,
                "eligible_epochs": eligible_count,
                "label_bearing_epochs": scored_count,
                "coverage": float(len(selected) / scored_count) if scored_count else 0.0,
                "accuracy": float(accuracy_score(targets, predicted)),
                "macro_f1": float(np.mean(f1)),
            }
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit baseline sleep-stage classifiers")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", action="store_true", help="Confirm the requested model fit")
    args = parser.parse_args()
    if not args.train:
        parser.error("pass --train to explicitly fit the baseline models")
    print(BaselineExperiment(args.config).run())


if __name__ == "__main__":
    main()
