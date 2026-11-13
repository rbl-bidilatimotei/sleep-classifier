import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")
import torch  # noqa: E402

from sleep_classifier.checkpoint_audit import CheckpointReplayAudit  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.gru_training import GRUExperiment  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402


def _write_patch_recipe(root: Path) -> Path:
    """Write synthetic participant tables and a tiny instrumented patch recipe.

    Args:
        root (Path): Temporary recipe and feature directory.
    """
    prepared = root / "features"
    participants: list[dict[str, Any]] = []
    for partition, participant, offset in (("train", "train1", 0.0),
                                           ("validation", "valid1", 0.2)):
        directory = prepared / partition
        directory.mkdir(parents=True)
        values = {column: np.arange(6, dtype=float) * (index + 1) + offset
                  for index, column in enumerate(FEATURE_COLUMNS)}
        pd.DataFrame(values).assign(time=np.arange(6) * 30, target=[0, 1, 2, 3, 0, 1],
                                    eligible=True, segment_id=0).to_csv(
            directory / f"{participant}.csv", index=False,
        )
        participants.append({"participant_id": participant, "partition": partition,
                             "eligible_epochs": 6})
    (prepared / "test").mkdir()
    (prepared / "test/reserved.csv").write_text("DO NOT READ FINAL TEST OBSERVATIONS")
    (prepared / "provenance.json").write_text(json.dumps({
        "split": {"train": ["train1"], "validation": ["valid1"], "test": ["reserved"]},
        "feature_columns": list(FEATURE_COLUMNS), "participants": participants,
        "scaling_fitted": False, "test_observations_read": False,
    }))
    (root / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "features"}))
    settings = {
        "prepare_config": "prepare.yaml", "output_dir": "results",
        "model_type": "patch_transformer",
        "patch_size": 3, "seed": 42, "epochs": 2, "batch_size": 2, "context_epochs": 3,
        "learning_rate": 0.001, "patience": 2, "gradient_clip": 1.0, "device": "cpu",
        "transformer": {"d_model": 8, "nhead": 2, "num_layers": 1, "feedforward_dim": 16,
                        "dropout": 0.2, "max_context": 3},
        "diagnostics": {"enabled": True, "save_predictions": True, "evaluate_training": True,
                        "checkpoint_epochs": [1, 2]},
    }
    path = root / "patch.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_patch_preflight_keeps_raw_feature_shape_and_writes_no_results(tmp_path: Path) -> None:
    """Check the patch model without optimization or persistence.

    Args:
        tmp_path (Path): Temporary synthetic preflight directory.
    """
    config = _write_patch_recipe(tmp_path)
    result = GRUExperiment(config).run_check()

    assert result["model_type"] == "patch_transformer"
    assert result["input_shape"] == [2, 2, 6]
    assert result["logit_shape"] == [2, 4]
    assert not (tmp_path / "results").exists()


def test_tiny_synthetic_patch_training_exports_replayable_checkpoints(tmp_path: Path) -> None:
    """Exercise shared training, train-only scaling, evidence, and portable patch inference.

    Args:
        tmp_path (Path): Temporary synthetic training directory.
    """
    config = _write_patch_recipe(tmp_path)
    metrics_path = GRUExperiment(config).run()
    metrics = json.loads(metrics_path.read_text())
    results = metrics_path.parent
    checkpoint = torch.load(results / "model_checkpoint.pt", weights_only=True)
    restored = ModelFactory.from_checkpoint(checkpoint)
    predictions = pd.read_csv(results / "validation_predictions.csv")
    replay = CheckpointReplayAudit(tmp_path).build()

    assert metrics["model_type"] == "patch_transformer"
    assert metrics["features"] == list(FEATURE_COLUMNS)
    assert len(predictions) == 6 and predictions["target"].tolist() == [0, 1, 2, 3, 0, 1]
    assert restored.input_size == 6 and restored.backbone.input_size == 18
    assert checkpoint["context_epochs"] == 3
    assert checkpoint["model_config"]["patch_size"] == 3
    assert checkpoint["scaler_mean"] == [2.5 * (index + 1) for index in range(6)]
    assert replay["matches_saved_atol_2e_6"] and replay["weights_unchanged"]
    assert not replay["gradients_created"]
    intermediate = torch.load(
        results / "diagnostics/checkpoints/epoch-001.pt", weights_only=True,
    )
    assert intermediate["model_type"] == "patch_transformer"
    assert intermediate["model_config"] == checkpoint["model_config"]
    assert not ModelFactory.from_checkpoint(intermediate).training
