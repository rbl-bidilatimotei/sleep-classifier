import json
from pathlib import Path

import pandas as pd
import pytest

from sleep_classifier.features import FEATURE_COLUMNS

baseline = pytest.importorskip("sleep_classifier.baseline", exc_type=ImportError)
PreparedFeatureDataset = baseline.PreparedFeatureDataset
BaselineExperiment = baseline.BaselineExperiment


def _write_dataset(root: Path) -> None:
    provenance = {
        "split": {"train": ["train-1"], "validation": ["validation-1"], "test": ["test-1"]},
        "feature_columns": list(FEATURE_COLUMNS),
        "participants": [
            {"participant_id": "train-1", "partition": "train", "eligible_epochs": 2},
            {"participant_id": "validation-1", "partition": "validation", "eligible_epochs": 2},
        ],
    }
    (root / "provenance.json").write_text(json.dumps(provenance))
    values = {column: [1.0, 2.0, 3.0, 4.0] for column in FEATURE_COLUMNS}
    values.update({"target": [0, -1, 1, 2], "eligible": [True, False, True, False]})
    for partition, participant in (("train", "train-1"), ("validation", "validation-1")):
        directory = root / partition
        directory.mkdir()
        pd.DataFrame(values).to_csv(directory / f"{participant}.csv", index=False)


def test_dataset_reads_only_split_participants_and_explicit_features(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    loaded = PreparedFeatureDataset(tmp_path).load()

    assert len(loaded["partitions"]["train"]) == 2
    assert loaded["partitions"]["train"]["target"].tolist() == [0, 1]
    assert list(loaded["partitions"]["train"].loc[:, FEATURE_COLUMNS].columns) == list(
        FEATURE_COLUMNS
    )
    assert "test-1" not in loaded["partitions"]["validation"]["participant_id"].tolist()
    metrics = BaselineExperiment(tmp_path / "unused.yaml")
    assert metrics._coverage(loaded["partitions"]["validation"])["fraction"] == pytest.approx(2 / 3)


def test_dataset_rejects_non_finite_features(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    path = tmp_path / "train" / "train-1.csv"
    frame = pd.read_csv(path)
    frame.loc[0, FEATURE_COLUMNS[0]] = float("nan")
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="finite"):
        PreparedFeatureDataset(tmp_path).load()


def test_dataset_rejects_overlapping_participant_splits(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["split"]["validation"] = ["train-1"]
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="disjoint"):
        PreparedFeatureDataset(tmp_path).load()


def test_dataset_rejects_non_integer_targets_before_cast(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    path = tmp_path / "train" / "train-1.csv"
    frame = pd.read_csv(path)
    frame["target"] = frame["target"].astype(float)
    frame.loc[0, "target"] = 1.5
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Targets"):
        PreparedFeatureDataset(tmp_path).load()


def test_dataset_rejects_stale_eligible_count(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["participants"][0]["eligible_epochs"] = 3
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="count mismatch"):
        PreparedFeatureDataset(tmp_path).load()


def test_scaler_fits_training_features_only(tmp_path: Path) -> None:
    values = {column: [0.0, 2.0] for column in FEATURE_COLUMNS}
    train = pd.DataFrame(values)
    train["target"] = [0, 1]
    validation = pd.DataFrame({column: [100.0] for column in FEATURE_COLUMNS})
    validation["target"] = [0]
    experiment = BaselineExperiment(tmp_path / "unused.yaml")

    settings = {"seed": 42, "max_iter": 300, "inverse_regularization": 1.0}
    pipeline, _ = experiment._fit_models(train, settings)

    scaler = pipeline.named_steps["standardscaler"]
    assert scaler.mean_.tolist() == [1.0] * len(FEATURE_COLUMNS)
    assert validation.loc[:, FEATURE_COLUMNS].iloc[0, 0] == 100.0


def test_loader_refuses_to_hide_participant_without_eligible_epochs(tmp_path):
    _write_dataset(tmp_path)
    path = tmp_path / "validation/validation-1.csv"
    frame = pd.read_csv(path)
    frame["eligible"] = False
    frame.to_csv(path, index=False)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for row in provenance["participants"]:
        if row["partition"] == "validation":
            row["eligible_epochs"] = 0
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="No eligible rows"):
        PreparedFeatureDataset(tmp_path).load()
