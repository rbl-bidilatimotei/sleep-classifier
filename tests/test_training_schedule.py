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

from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.gru_training import GRUExperiment, TrainingConfig  # noqa: E402


def _write_config(
    root: Path, controls: dict[str, Any] | None = None, epochs: int = 6, patience: int = 12,
) -> Path:
    """Write deterministic synthetic features and a tiny active-dropout Transformer recipe.

    Args:
        root (Path): Isolated synthetic experiment directory.
        controls (dict[str, Any] | None): Optional scheduler settings.
        epochs (int): Maximum synthetic training passes.
        patience (int): Independent checkpoint-selection stopping patience.
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
        pd.DataFrame(values).assign(time=indexes * 30, target=[0, 1, 2, 3] * 2,
                                    eligible=True, segment_id=0).to_csv(
            directory / f"{identifier}.csv", index=False,
        )
        participants.append({"participant_id": identifier, "partition": partition,
                             "eligible_epochs": 8})
    (prepared / "provenance.json").write_text(json.dumps({
        "split": {"train": ["train1"], "validation": ["valid1"], "test": ["reserved"]},
        "feature_columns": list(FEATURE_COLUMNS), "participants": participants,
        "scaling_fitted": False, "test_observations_read": False,
    }))
    (root / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "prepared"}))
    settings: dict[str, Any] = {
        "prepare_config": "prepare.yaml", "output_dir": "results", "model_type": "transformer",
        "seed": 42, "epochs": epochs, "batch_size": 2, "context_epochs": 3,
        "learning_rate": 0.001, "patience": patience, "gradient_clip": 1.0, "device": "cpu",
        "transformer": {"d_model": 8, "nhead": 2, "num_layers": 1, "feedforward_dim": 16,
                        "dropout": 0.2, "max_context": 3},
        "diagnostics": {"enabled": True, "evaluate_training": True,
                        "save_predictions": True, "checkpoint_epochs": list(range(1, epochs + 1))},
    }
    if controls is not None:
        settings["lr_scheduler"] = {"kind": "reduce_on_plateau"} | controls
    path = root / "model.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


class _ForcedPlateauExperiment(GRUExperiment):
    """Hold only the observed selection score constant to exercise scheduler timing."""

    def _epoch_summary(
        self, epoch: int, train_loss: float, validation_loss: float,
        targets: np.ndarray, predictions: np.ndarray, frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Retain genuine synthetic losses while supplying a controlled plateau score.

        Args:
            epoch (int): Completed synthetic epoch.
            train_loss (float): Actual stochastic training loss.
            validation_loss (float): Actual validation criterion loss.
            targets (np.ndarray): Synthetic held-out targets.
            predictions (np.ndarray): Actual model predictions.
            frame (pd.DataFrame): Ordered synthetic participant rows.
        """
        result = super()._epoch_summary(epoch, train_loss, validation_loss,
                                        targets, predictions, frame)
        result["validation_macro_f1"] = 0.5
        return result


def _run(config: Path, forced: bool = False) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    """Fit only the tiny synthetic model and capture its selected state and random stream.

    Args:
        config (Path): Synthetic CPU experiment configuration.
        forced (bool): Supply a controlled plateau observation after validation.
    """
    experiment = _ForcedPlateauExperiment(config) if forced else GRUExperiment(config)
    path = experiment.run()
    checkpoint = torch.load(path.parent / "model_checkpoint.pt", weights_only=True)
    return json.loads(path.read_text()), checkpoint, torch.get_rng_state().clone()


def test_nonfiring_schedule_preserves_dropout_trajectory_and_random_state(tmp_path: Path) -> None:
    """Keep optimizer trajectories identical until the first actual rate reduction.

    Args:
        tmp_path (Path): Isolated reference and scheduled synthetic outputs.
    """
    plain_config = _write_config(tmp_path / "plain", epochs=3)
    scheduled_config = _write_config(tmp_path / "scheduled", {"patience": 99}, epochs=3)
    plain, plain_state, plain_rng = _run(plain_config)
    candidate, candidate_state, candidate_rng = _run(scheduled_config)

    for left, right in zip(plain["history"], candidate["history"], strict=True):
        assert left == {key: right[key] for key in left}
        assert "learning_rates_used" not in left
        assert right["learning_rates_used"] == [0.001]
        assert not right["scheduler_reduced"]
    for key, weights in plain_state["state_dict"].items():
        assert torch.equal(weights, candidate_state["state_dict"][key])
    assert plain["best_epoch"] == candidate["best_epoch"]
    assert torch.equal(plain_rng, candidate_rng)
    pd.testing.assert_frame_equal(
        pd.read_csv(plain_config.parent / "results/validation_predictions.csv"),
        pd.read_csv(scheduled_config.parent / "results/validation_predictions.csv"),
        check_exact=True,
    )


def test_schedule_is_reproducible_and_diagnostics_record_rates_used(tmp_path: Path) -> None:
    """Apply reductions to the next epoch and preserve correct current-epoch provenance.

    Args:
        tmp_path (Path): Isolated repeated scheduled experiments.
    """
    first_config = _write_config(tmp_path / "first", {"patience": 0})
    first, first_state, first_rng = _run(first_config, forced=True)
    second, second_state, second_rng = _run(
        _write_config(tmp_path / "second", {"patience": 0}), forced=True,
    )
    assert first["history"] == second["history"]
    assert torch.equal(first_rng, second_rng)
    for name, weights in first_state["state_dict"].items():
        assert torch.equal(weights, second_state["state_dict"][name])
    plain_config = _write_config(tmp_path / "plain")
    plain, _, _ = _run(plain_config, forced=True)
    for scheduled_row, plain_row in zip(first["history"][:2], plain["history"][:2], strict=True):
        assert plain_row == {key: scheduled_row[key] for key in plain_row}
    for epoch in (1, 2):
        filename = f"results/diagnostics/checkpoints/epoch-{epoch:03d}.pt"
        before_change = torch.load(first_config.parent / filename, weights_only=True)
        baseline = torch.load(plain_config.parent / filename, weights_only=True)
        for name, weights in baseline["state_dict"].items():
            assert torch.equal(weights, before_change["state_dict"][name])
    expected = [0.001, 0.001, 0.0003, 0.00009, 0.00003, 0.00003]
    for epoch, row in enumerate(first["history"], 1):
        record = json.loads((first_config.parent / "results/diagnostics/epochs"
                             / f"epoch-{epoch:03d}.json").read_text())
        assert row["learning_rates_used"] == pytest.approx([expected[epoch - 1]])
        assert record["optimizer"]["learning_rates"] == row["learning_rates_used"]
        if epoch < len(expected):
            assert row["learning_rates_next_epoch"] == pytest.approx([expected[epoch]])
        else:
            assert row["learning_rates_next_epoch"] is None
            assert not row["scheduler_reduced"]


def test_schedule_does_not_reset_early_stopping_or_advance_after_stop(tmp_path: Path) -> None:
    """Stop on unchanged selection patience despite a preceding rate reduction.

    Args:
        tmp_path (Path): Isolated controlled-plateau experiment.
    """
    config = _write_config(tmp_path / "stop", {"patience": 0}, epochs=6, patience=2)
    experiment = _ForcedPlateauExperiment(config)
    metrics = json.loads(experiment.run().read_text())

    assert len(metrics["history"]) == 3 and metrics["best_epoch"] == 1
    assert metrics["history"][1]["scheduler_reduced"]
    assert metrics["history"][-1]["learning_rates_next_epoch"] is None
    assert experiment.schedule.scheduler.last_epoch == 2


def test_scheduled_preflight_constructs_no_schedule_or_evidence(tmp_path: Path) -> None:
    """Validate schedule settings without optimizer construction or output artifacts.

    Args:
        tmp_path (Path): Isolated synthetic preflight directory.
    """
    config = _write_config(tmp_path / "preflight", {"patience": 3})
    experiment = GRUExperiment(config)
    result = experiment.run_check()

    assert result["logit_shape"] == [2, 4]
    assert experiment.schedule is None
    assert not (config.parent / "results").exists()
    settings = yaml.safe_load(config.read_text())
    settings["optimizer_agent"] = {"backend": "keep", "goal": "test", "decisions": 1,
                                   "every": 1}
    with pytest.raises(ValueError, match="plain Adam"):
        TrainingConfig(settings)


def test_scheduler_floor_cannot_exceed_initial_rate_during_preflight(tmp_path: Path) -> None:
    """Reject an impossible schedule before creating models or result files.

    Args:
        tmp_path (Path): Isolated invalid synthetic configuration directory.
    """
    config = _write_config(tmp_path / "invalid", {"min_lr": 0.01})
    with pytest.raises(ValueError, match="min_lr"):
        GRUExperiment(config).run_check()
    assert not (config.parent / "results").exists()
