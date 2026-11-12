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

from sleep_classifier.epoch_diagnostics import EpochDiagnostics  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.gru_training import GRUExperiment  # noqa: E402


def _write_config(root: Path, enabled: bool = True) -> Path:
    """Create a complete synthetic Transformer recipe and participant-disjoint feature tables.

    Args:
        root (Path): Isolated directory for the synthetic experiment.
        enabled (bool): Enable diagnostics and fixed-weight training evaluation.
    """
    root.mkdir(parents=True)
    prepared = root / "prepared"
    participants: list[dict[str, Any]] = []
    for partition, identifier, offset in (("train", "train1", 0.0),
                                           ("validation", "valid1", 0.15)):
        directory = prepared / partition
        directory.mkdir(parents=True)
        indexes = np.arange(8, dtype=float)
        values = {column: np.sin(indexes + feature) + indexes * (feature + 1) / 10 + offset
                  for feature, column in enumerate(FEATURE_COLUMNS)}
        frame = pd.DataFrame(values).assign(time=indexes * 30, target=[0, 1, 2, 3] * 2,
                                             eligible=True, segment_id=0)
        frame.to_csv(directory / f"{identifier}.csv", index=False)
        participants.append({"participant_id": identifier, "partition": partition,
                             "eligible_epochs": 8})
    (prepared / "test").mkdir()
    (prepared / "test/reserved.csv").write_text("DO NOT READ FINAL TEST OBSERVATIONS")
    (prepared / "provenance.json").write_text(json.dumps({
        "split": {"train": ["train1"], "validation": ["valid1"], "test": ["reserved"]},
        "feature_columns": list(FEATURE_COLUMNS), "participants": participants,
        "scaling_fitted": False, "test_observations_read": False,
    }))
    (root / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "prepared"}))
    settings: dict[str, Any] = {
        "prepare_config": "prepare.yaml", "output_dir": "results", "model_type": "transformer",
        "seed": 42, "epochs": 3, "batch_size": 2, "context_epochs": 3,
        "learning_rate": 0.001, "patience": 3, "gradient_clip": 1.0, "device": "cpu",
        "transformer": {"d_model": 8, "nhead": 2, "num_layers": 1, "feedforward_dim": 16,
                        "dropout": 0.2, "max_context": 3},
    }
    if enabled:
        settings["diagnostics"] = {"enabled": True, "evaluate_training": True,
                                   "save_predictions": True, "checkpoint_epochs": [1, 2, 3]}
    path = root / "transformer.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def _run_synthetic(config: Path) -> tuple[dict[str, Any], torch.Tensor]:
    """Fit the tiny synthetic model and retain the final Torch random state.

    Args:
        config (Path): Synthetic CPU recipe to train.
    """
    path = GRUExperiment(config).run()
    rng = torch.get_rng_state().clone()
    return json.loads(path.read_text()), rng


def test_diagnostics_preserve_dropout_trajectory_and_selected_checkpoint(tmp_path: Path) -> None:
    """Require identical stochastic training with evidence capture disabled and enabled.

    Args:
        tmp_path (Path): Isolated source and instrumented experiment directories.
    """
    plain_config = _write_config(tmp_path / "plain", enabled=False)
    instrumented_config = _write_config(tmp_path / "instrumented")
    plain, plain_rng = _run_synthetic(plain_config)
    instrumented, instrumented_rng = _run_synthetic(instrumented_config)
    plain_results = plain_config.parent / "results"
    instrumented_results = instrumented_config.parent / "results"

    assert len(plain["history"]) == 3
    assert plain["history"] == instrumented["history"]
    assert plain["best_epoch"] == instrumented["best_epoch"]
    assert plain["macro_f1"] == instrumented["macro_f1"]
    assert torch.equal(plain_rng, instrumented_rng)
    pd.testing.assert_frame_equal(
        pd.read_csv(plain_results / "validation_predictions.csv"),
        pd.read_csv(instrumented_results / "validation_predictions.csv"),
        check_exact=True,
    )
    plain_checkpoint = torch.load(plain_results / "model_checkpoint.pt", weights_only=True)
    selected = torch.load(instrumented_results / "model_checkpoint.pt", weights_only=True)
    for name, weights in plain_checkpoint["state_dict"].items():
        assert torch.equal(weights, selected["state_dict"][name])
    assert not (plain_results / "diagnostics").exists()

    diagnostics = instrumented_results / "diagnostics"
    for epoch, history in enumerate(instrumented["history"], start=1):
        evidence = json.loads((diagnostics / "epochs" / f"epoch-{epoch:03d}.json").read_text())
        assert evidence["optimizer"]["learning_rates"] == [0.001]
        assert evidence["optimizer"]["gradient_steps"] == 4
        assert evidence["optimizer"]["gradient_norm_pre_clip_mean"] > 0
        assert evidence["optimizer"]["gradient_norm_pre_clip_max"] > 0
        assert 0 <= evidence["optimizer"]["clipped_step_fraction"] <= 1
        assert evidence["training"]["stochastic_nll"] == history["train_loss"]
        assert evidence["training"]["evaluation_nll"] > 0
        assert evidence["validation_criterion_nll"] == history["validation_loss"]
        assert evidence["validation"]["nll"] == pytest.approx(history["validation_loss"], abs=1e-6)
        assert evidence["validation"]["count"] == 8
    intermediate = torch.load(
        diagnostics / "checkpoints" / f"epoch-{instrumented['best_epoch']:03d}.pt",
        weights_only=True,
    )
    for name, weights in selected["state_dict"].items():
        assert torch.equal(weights, intermediate["state_dict"][name])
    for key in ("scaler_mean", "scaler_scale", "features", "context_epochs", "model_config"):
        assert intermediate[key] == selected[key]
    training = pd.read_csv(instrumented_config.parent / "prepared/train/train1.csv")
    np.testing.assert_allclose(intermediate["scaler_mean"], training.loc[:, FEATURE_COLUMNS].mean())
    assert intermediate["features"] == list(FEATURE_COLUMNS)
    assert intermediate["training_config"]["transformer"]["dropout"] == 0.2


def test_diagnostic_preflight_creates_no_evidence_or_results(tmp_path: Path) -> None:
    """Keep instrumented shape checks free from optimization and evidence artifacts.

    Args:
        tmp_path (Path): Isolated synthetic preflight experiment directory.
    """
    config = _write_config(tmp_path / "check")
    experiment = GRUExperiment(config)
    result = experiment.run_check()

    assert result["input_shape"] == [2, 2, 6]
    assert result["logit_shape"] == [2, 4]
    assert experiment.diagnostics is None
    assert not (config.parent / "results").exists()


class _FailingSecondEpochRecorder(EpochDiagnostics):
    """Inject one synthetic recording failure after a complete epoch has been persisted."""

    def record_epoch(self, epoch: int, **values: Any) -> Path:
        """Fail before saving epoch two while retaining the first epoch's evidence.

        Args:
            epoch (int): Completed optimization epoch supplied by the trainer.
            values (dict[str, Any]): Remaining named evidence fields from the trainer.
        """
        if epoch == 2:
            raise RuntimeError("Synthetic epoch-two evidence failure")
        return super().record_epoch(epoch=epoch, **values)


def test_failed_training_retains_completed_epoch_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the first completed epoch when a later diagnostic write fails.

    Args:
        tmp_path (Path): Isolated synthetic failure experiment directory.
        monkeypatch (pytest.MonkeyPatch): Replace only the evidence recorder with a failure probe.
    """
    config = _write_config(tmp_path / "failure")
    monkeypatch.setattr("sleep_classifier.gru_training.EpochDiagnostics",
                        _FailingSecondEpochRecorder)
    with pytest.raises(RuntimeError, match="Synthetic epoch-two"):
        GRUExperiment(config).run()

    results = config.parent / "results"
    diagnostics = results / "diagnostics"
    first = json.loads((diagnostics / "epochs/epoch-001.json").read_text())
    assert first["epoch"] == 1
    assert (diagnostics / first["files"]["validation_predictions"]["path"]).is_file()
    assert (diagnostics / first["files"]["checkpoint"]["path"]).is_file()
    assert not (diagnostics / "epochs/epoch-002.json").exists()
    assert not (results / "metrics.json").exists()
