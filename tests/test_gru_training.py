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


def test_live_pilot_is_rejected_without_opt_in_before_output_creation(tmp_path):
    path = _config(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["optimizer_agent"] = {
        "backend": "codex",
        "goal": "Reduce training loss",
        "decisions": 1,
        "every": 1,
        "model": "gpt-6.1-sol",
        "effort": "high",
        "max_tokens": 1000,
    }
    path.write_text(yaml.safe_dump(settings))
    experiment = GRUExperiment(path)
    experiment.run_check()
    with pytest.raises(RuntimeError, match="allow-live-agent"):
        experiment.run()
    assert not (tmp_path / "results").exists()


def test_actual_keep_adapter_preserves_synthetic_training(tmp_path):
    pytest.importorskip("opal")
    path = _config(tmp_path)
    original = yaml.safe_load(path.read_text())
    plain_metrics = json.loads(GRUExperiment(path).run().read_text())
    settings = original | {
        "output_dir": "keep-results",
        "optimizer_agent": {
            "backend": "keep",
            "goal": "Preserve Adam updates",
            "decisions": 1,
            "every": 1,
        },
    }
    path.write_text(yaml.safe_dump(settings))
    keep_metrics = json.loads(GRUExperiment(path).run().read_text())
    assert keep_metrics["history"] == plain_metrics["history"]
    assert keep_metrics["macro_f1"] == plain_metrics["macro_f1"]
    assert keep_metrics["optimizer"]["agent"]["backend"] == "keep"
    assert keep_metrics["optimizer"]["updates"] > 0
    plain = torch.load(tmp_path / "results/gru_checkpoint.pt", weights_only=True)
    wrapped = torch.load(tmp_path / "keep-results/gru_checkpoint.pt", weights_only=True)
    for name in plain["state_dict"]:
        torch.testing.assert_close(
            plain["state_dict"][name], wrapped["state_dict"][name], atol=0, rtol=0
        )


def test_optimizer_closes_after_training_failure(tmp_path, monkeypatch):
    from unittest.mock import Mock

    experiment = GRUExperiment(_config(tmp_path))
    adapter = Mock()
    monkeypatch.setattr(experiment, "_create_optimizer", Mock(return_value=adapter))
    monkeypatch.setattr(experiment, "_fit", Mock(side_effect=RuntimeError("synthetic failure")))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        experiment.run()
    adapter.close.assert_called_once_with()


def test_pilot_configs_preserve_the_same_model_recipe():
    root = Path(__file__).resolve().parents[1] / "configs"
    control = yaml.safe_load((root / "gru-opal-control.yaml").read_text())
    for name in ("keep", "codex"):
        recipe = yaml.safe_load((root / f"gru-opal-{name}.yaml").read_text())
        recipe.pop("optimizer_agent")
        recipe["output_dir"] = control["output_dir"]
        assert recipe == control


def test_runtime_failure_precedes_run_directory_creation(tmp_path, monkeypatch):
    from unittest.mock import Mock

    path = _config(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["optimizer_agent"] = {
        "backend": "codex",
        "goal": "Reduce training loss",
        "decisions": 1,
        "every": 1,
        "model": "gpt-6.1-sol",
        "effort": "high",
        "max_tokens": 1000,
    }
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(
        "sleep_classifier.gru_training.CodexRuntime.check",
        Mock(side_effect=RuntimeError("isolated login is missing")),
    )
    with pytest.raises(RuntimeError, match="isolated login is missing"):
        GRUExperiment(path).run(allow_live_agent=True)
    assert not (tmp_path / "results").exists()


def test_retry_configs_preserve_model_and_schedule_early_intervention():
    root = Path(__file__).resolve().parents[1] / "configs"
    control = yaml.safe_load((root / "gru-opal-retry-control.yaml").read_text())
    for name in ("keep", "codex"):
        recipe = yaml.safe_load((root / f"gru-opal-retry-{name}.yaml").read_text())
        agent = recipe.pop("optimizer_agent")
        assert agent["schedule"] == [50]
        assert agent["decisions"] == 1
        if name == "codex":
            assert agent["binary"] == "tools/opal-codex/node_modules/.bin/codex"
        recipe["output_dir"] = control["output_dir"]
        assert recipe == control


def _transformer_config(tmp_path: Path) -> Path:
    """Build a tiny Transformer recipe over synthetic prepared features.

    Args:
        tmp_path (Path): Temporary root for features and configuration.
    """
    path = _config(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings.pop("hidden_size")
    settings["model_type"] = "transformer"
    settings["transformer"] = {
        "d_model": 8, "nhead": 2, "num_layers": 1, "feedforward_dim": 16,
        "dropout": 0.0, "max_context": 2,
    }
    path.write_text(yaml.safe_dump(settings))

    return path


def test_transformer_preflight_uses_the_shared_loader_without_artifacts(tmp_path: Path) -> None:
    """Check Transformer batches without creating an optimizer or model results.

    Args:
        tmp_path (Path): Temporary root for synthetic prepared features.
    """
    experiment = GRUExperiment(_transformer_config(tmp_path))
    result = experiment.run_check()

    assert result["model_type"] == "transformer"
    assert result["input_shape"] == [2, 2, len(FEATURE_COLUMNS)]
    assert result["logit_shape"] == [2, 4]
    assert result["scaler_mean"] == [1.0] * len(FEATURE_COLUMNS)
    assert not (tmp_path / "results").exists()


def test_synthetic_transformer_training_exports_portable_model_metadata(tmp_path: Path) -> None:
    """Train the tiny Transformer through the same loop and restore its checkpoint.

    Args:
        tmp_path (Path): Temporary root for synthetic features and model artifacts.
    """
    from sleep_classifier.model_factory import ModelFactory
    from sleep_classifier.transformer_model import CausalTransformerClassifier

    path = _transformer_config(tmp_path)
    metrics_path = GRUExperiment(path).run()
    metrics = json.loads(metrics_path.read_text())
    predictions = pd.read_csv(metrics_path.parent / "validation_predictions.csv")
    checkpoint = torch.load(metrics_path.parent / "model_checkpoint.pt", weights_only=True)
    restored = ModelFactory.from_checkpoint(checkpoint)

    assert metrics["model_type"] == checkpoint["model_type"] == "transformer"
    parameter_count = sum(parameter.numel() for parameter in restored.parameters())
    assert metrics["parameter_count"] == parameter_count
    assert checkpoint["model_config"] == metrics["model_config"]
    assert checkpoint["scaler_mean"] == [1.0] * len(FEATURE_COLUMNS)
    assert "prediction" in predictions
    assert "gru_prediction" not in predictions
    assert not (metrics_path.parent / "gru_checkpoint.pt").exists()
    assert isinstance(restored, CausalTransformerClassifier)
    assert not restored.training
    assert len(predictions) == 2


@pytest.mark.parametrize("model_type", ["gru", "transformer"])
def test_shared_trainer_logs_under_an_architecture_neutral_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, model_type: str,
) -> None:
    """Identify the shared trainer independently of the selected architecture.

    Args:
        tmp_path (Path): Temporary synthetic feature and configuration directory.
        caplog (pytest.LogCaptureFixture): Captured application log records.
        model_type (str): Model architecture used for the no-gradient preflight.
    """
    import logging

    config = _config(tmp_path)
    settings = yaml.safe_load(config.read_text())
    settings["model_type"] = model_type
    if model_type == "transformer":
        settings.pop("hidden_size")
        settings["transformer"] = {"d_model": 8, "nhead": 2, "num_layers": 1,
                                   "feedforward_dim": 16, "max_context": 2, "dropout": 0}
    config.write_text(yaml.safe_dump(settings))

    with caplog.at_level(logging.INFO, logger="sleep_classifier.neural_training"):
        GRUExperiment(config).run_check()

    records = [record for record in caplog.records if "Neural model check passed" in record.message]
    assert len(records) == 1
    assert records[0].name == "sleep_classifier.neural_training"
    assert json.loads(records[0].message.split(": ", 1)[1])["model_type"] == model_type
    assert not (tmp_path / "results").exists()
