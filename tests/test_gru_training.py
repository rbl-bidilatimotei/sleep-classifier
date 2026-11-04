import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.gru_model import CausalGRUClassifier  # noqa: E402
from sleep_classifier.gru_training import GRUExperiment, TrainingConfig  # noqa: E402


def _write_prepared(root: Path) -> None:
    provenance = {
        "split": {"train": ["train1"], "validation": ["valid1"], "test": ["test1"]},
        "feature_columns": list(FEATURE_COLUMNS),
        "participants": [
            {"participant_id": "train1", "partition": "train", "eligible_epochs": 2},
            {"participant_id": "valid1", "partition": "validation", "eligible_epochs": 2},
        ],
    }
    (root / "provenance.json").write_text(json.dumps(provenance))
    for partition, participant, offset in (
        ("train", "train1", 0.0),
        ("validation", "valid1", 100.0),
    ):
        directory = root / partition
        directory.mkdir()
        values = {column: [offset, offset + 2.0] for column in FEATURE_COLUMNS}
        values.update(
            {
                "target": [0, 1],
                "eligible": [True, True],
                "time": [0, 30],
                "segment_id": [0, 0],
            }
        )
        pd.DataFrame(values).to_csv(directory / f"{participant}.csv", index=False)


def _config(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_prepared(prepared)
    prepare_path = tmp_path / "prepare.yaml"
    prepare_path.write_text(yaml.safe_dump({"output_dir": "prepared"}))
    config_path = tmp_path / "gru.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "prepare_config": "prepare.yaml",
                "output_dir": "results",
                "seed": 42,
                "epochs": 2,
                "batch_size": 2,
                "context_epochs": 2,
                "hidden_size": 4,
                "learning_rate": 0.001,
                "patience": 1,
                "gradient_clip": 1.0,
                "device": "cpu",
            }
        )
    )
    return config_path


def test_check_runs_forward_without_training_or_writing_artifacts(tmp_path: Path) -> None:
    experiment = GRUExperiment(_config(tmp_path))

    result = experiment.run_check()

    assert result["train_rows"] == 2
    assert result["validation_rows"] == 2
    assert result["logit_shape"] == [2, 4]
    assert result["scaler_mean"] == [1.0] * len(FEATURE_COLUMNS)
    assert not (tmp_path / "results").exists()


def test_training_config_rejects_unknown_device_and_nonpositive_epochs() -> None:
    settings = {
        "seed": 42,
        "epochs": 1,
        "batch_size": 1,
        "context_epochs": 1,
        "hidden_size": 4,
        "patience": 1,
        "learning_rate": 0.001,
        "gradient_clip": 1.0,
        "device": "unknown",
    }
    with pytest.raises(ValueError, match="device"):
        TrainingConfig(settings)
    settings["device"] = "cpu"
    settings["epochs"] = 0
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(settings)


@pytest.mark.parametrize("value", [True, 2.0, -1])
def test_training_config_rejects_invalid_seed(value) -> None:
    settings = {
        "seed": value,
        "epochs": 1,
        "batch_size": 1,
        "context_epochs": 1,
        "hidden_size": 4,
        "patience": 1,
        "learning_rate": 0.001,
        "gradient_clip": 1.0,
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(settings)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_training_config_rejects_nonfinite_or_boolean_float_settings(value) -> None:
    settings = {
        "seed": 0,
        "epochs": 1,
        "batch_size": 1,
        "context_epochs": 1,
        "hidden_size": 4,
        "patience": 1,
        "learning_rate": value,
        "gradient_clip": 1.0,
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(settings)


def test_checkpoint_contains_plain_scaler_lists_and_model_state(tmp_path: Path) -> None:
    experiment = GRUExperiment(_config(tmp_path))
    experiment.settings = {"hidden_size": 4, "context_epochs": 2, "seed": 42}
    experiment.output_dir = tmp_path / "results"
    model = CausalGRUClassifier(input_size=len(FEATURE_COLUMNS), hidden_size=4)
    scaler = StandardScaler().fit(pd.DataFrame({column: [0.0, 2.0] for column in FEATURE_COLUMNS}))
    experiment._save(model, scaler, {"accuracy": 0.5}, pd.DataFrame({"target": [0]}))

    checkpoint = torch.load(experiment.output_dir / "gru_checkpoint.pt", weights_only=True)

    assert checkpoint["scaler_mean"] == [1.0] * len(FEATURE_COLUMNS)
    assert checkpoint["features"] == list(FEATURE_COLUMNS)
    assert checkpoint["model_config"] == {
        "input_size": len(FEATURE_COLUMNS),
        "hidden_size": 4,
        "num_classes": 4,
    }
    assert checkpoint["context_epochs"] == 2
    assert checkpoint["training_config"]["seed"] == 42
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["state_dict"].values())


def test_synthetic_training_restores_best_state_and_exports_aligned_rows(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    validation_path = tmp_path / "prepared" / "validation" / "valid1.csv"
    validation = pd.read_csv(validation_path)
    validation["time"] = [30, 0]
    validation["target"] = [0, 1]
    validation.to_csv(validation_path, index=False)

    result_path = GRUExperiment(config_path).run()

    assert result_path.is_file()
    metrics = json.loads(result_path.read_text())
    exported = pd.read_csv(result_path.parent / "validation_predictions.csv")
    checkpoint = torch.load(result_path.parent / "gru_checkpoint.pt", weights_only=True)
    assert 1 <= metrics["best_epoch"] <= len(metrics["history"]) <= 2
    assert metrics["device"] == "cpu"
    assert exported["time"].tolist() == [0, 30]
    assert exported["target"].tolist() == [1, 0]
    assert checkpoint["context_epochs"] == 2


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_priority(tmp_path, monkeypatch, cuda_available, mps_available, expected):
    from unittest.mock import Mock

    settings = yaml.safe_load(_config(tmp_path).read_text())
    settings["device"] = "auto"
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=cuda_available))
    monkeypatch.setattr(torch.backends.mps, "is_available", Mock(return_value=mps_available))

    assert TrainingConfig(settings).device().type == expected


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_explicit_unavailable_device_fails(tmp_path, monkeypatch, requested):
    from unittest.mock import Mock

    settings = yaml.safe_load(_config(tmp_path).read_text())
    settings["device"] = requested
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    monkeypatch.setattr(torch.backends.mps, "is_available", Mock(return_value=False))

    with pytest.raises(RuntimeError, match=f"{requested.upper()} was requested"):
        TrainingConfig(settings).device()


def test_explicit_cpu_ignores_available_accelerators(tmp_path, monkeypatch):
    from unittest.mock import Mock

    settings = yaml.safe_load(_config(tmp_path).read_text())
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.backends.mps, "is_available", Mock(return_value=True))

    assert TrainingConfig(settings).device().type == "cpu"
