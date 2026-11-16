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
from torch import nn  # noqa: E402

from sleep_classifier.experiment_evidence import EpochEvidenceReport  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.gru_training import GRUExperiment, TrainingConfig  # noqa: E402
from sleep_classifier.loss_settings import LossSettings  # noqa: E402


def _write_synthetic_config(root: Path, loss: dict[str, Any] | None = None) -> Path:
    """Create disjoint synthetic features and an instrumented two-epoch CPU recipe.

    Args:
        root (Path): Temporary experiment root.
        loss (dict[str, Any] | None): Optional training loss override.
    """
    root.mkdir(parents=True, exist_ok=True)
    features = root / "prepared"
    features.mkdir()
    people = {"train": "train1", "validation": "valid1"}
    provenance = {
        "split": {"train": ["train1"], "validation": ["valid1"], "test": ["untouched"]},
        "feature_columns": list(FEATURE_COLUMNS),
        "participants": [
            {"participant_id": person, "partition": partition, "eligible_epochs": 4}
            for partition, person in people.items()
        ],
    }
    (features / "provenance.json").write_text(json.dumps(provenance))
    for partition, person in people.items():
        directory = features / partition
        directory.mkdir()
        frame = pd.DataFrame({name: [0.0, 1.0, 2.0, 3.0] for name in FEATURE_COLUMNS})
        frame = frame.assign(time=[0, 30, 60, 90], target=[0, 1, 2, 3],
                             eligible=True, segment_id=0)
        frame.to_csv(directory / f"{person}.csv", index=False)
    (root / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "prepared"}))
    settings = {
        "prepare_config": "prepare.yaml", "output_dir": "fold-01/results", "seed": 42,
        "epochs": 2, "batch_size": 2, "context_epochs": 2, "hidden_size": 4,
        "learning_rate": 0.001, "patience": 2, "gradient_clip": 1.0, "device": "cpu",
        "diagnostics": {"enabled": True, "evaluate_training": True,
                        "save_predictions": True, "checkpoint_epochs": [1]},
    }
    if loss is not None:
        settings["loss"] = loss
    config = root / "gru.yaml"
    config.write_text(yaml.safe_dump(settings))
    return config


@pytest.mark.parametrize("value", [
    [], "cross_entropy", {"name": "focal"}, {"weight": [1, 2, 3, 4]},
    {"label_smoothing": -0.01}, {"label_smoothing": 1.01},
    {"label_smoothing": True}, {"label_smoothing": "0.05"},
    {"label_smoothing": float("nan")}, {"label_smoothing": float("inf")},
])
def test_loss_settings_reject_unsupported_objectives(value: Any) -> None:
    """Reject invalid amounts and unused loss options before model preparation.

    Args:
        value (Any): Invalid loss recipe.
    """
    with pytest.raises(ValueError):
        LossSettings(value)


def test_label_smoothing_matches_uniform_target_mixture_and_zero_gradient() -> None:
    """Match the smoothed formula and preserve the zero-smoothing gradient."""
    logits = torch.tensor([[1.0, -0.4, 0.2, 2.0], [-1.0, 0.3, 0.7, 0.0]], requires_grad=True)
    targets = torch.tensor([3, 1])
    settings = LossSettings({"label_smoothing": 0.05})
    amount = settings.values["label_smoothing"]
    actual = nn.CrossEntropyLoss(label_smoothing=amount)(logits, targets)
    distribution = torch.nn.functional.one_hot(targets, 4) * (1 - amount) + amount / 4
    expected = -(distribution * logits.log_softmax(1)).sum(1).mean()
    torch.testing.assert_close(actual, expected)

    plain = nn.CrossEntropyLoss()(logits, targets)
    zero = nn.CrossEntropyLoss(label_smoothing=0.0)(logits, targets)
    assert torch.equal(plain, zero)
    assert torch.equal(torch.autograd.grad(plain, logits)[0], torch.autograd.grad(zero, logits)[0])
    assert LossSettings(None).provenance()["evaluation"]["label_smoothing"] == 0.0


def test_zero_smoothing_preserves_synthetic_history_and_selected_weights(tmp_path: Path) -> None:
    """Preserve the existing seeded optimization path when smoothing remains zero.

    Args:
        tmp_path (Path): Temporary synthetic recipe root.
    """
    plain = GRUExperiment(_write_synthetic_config(tmp_path / "plain"))
    explicit = GRUExperiment(_write_synthetic_config(tmp_path / "zero", {"label_smoothing": 0.0}))
    first = json.loads(plain.run().read_text())
    second = json.loads(explicit.run().read_text())
    assert first["history"] == second["history"]
    assert first["best_epoch"] == second["best_epoch"]
    states = [
        torch.load(experiment.output_dir / "gru_checkpoint.pt", weights_only=True)["state_dict"]
        for experiment in (plain, explicit)
    ]
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
    records = list((explicit.output_dir / "diagnostics/epochs").glob("*.json"))
    assert all(json.loads(path.read_text())["schema_version"] == 1 for path in records)


def test_smoothed_training_keeps_evaluation_nll_and_evidence_unsmoothed(tmp_path: Path) -> None:
    """Export true-class probability loss while labeling the smoothed objective separately.

    Args:
        tmp_path (Path): Temporary synthetic experiment root.
    """
    experiment = GRUExperiment(_write_synthetic_config(tmp_path, {"label_smoothing": 0.05}))
    check = experiment.run_check()
    assert check["loss"] == LossSettings({"label_smoothing": 0.05}).provenance()
    metrics = json.loads(experiment.run().read_text())
    assert metrics["loss"] == check["loss"]
    predictions = pd.read_csv(experiment.output_dir / "validation_predictions.csv")
    probabilities = predictions[[f"probability_{label}" for label in range(4)]].to_numpy()
    target_probability = probabilities[np.arange(len(predictions)), predictions.target.to_numpy()]
    selected = metrics["history"][metrics["best_epoch"] - 1]
    assert selected["validation_loss"] == pytest.approx(
        -np.log(target_probability).mean(), abs=1e-6,
    )

    report = EpochEvidenceReport(tmp_path).build()
    assert len(report["folds"][0]["epochs"]) == 2
    for record in report["folds"][0]["epochs"]:
        assert record["training_stochastic_nll"] is None
        assert record["training_stochastic_objective"] > 0
        assert record["training_evaluation_nll"] > 0
        assert record["validation_criterion_nll"] == pytest.approx(
            record["validation_nll"], abs=1e-6,
        )
    saved = json.loads((experiment.output_dir / "diagnostics/epochs/epoch-001.json").read_text())
    assert saved["schema_version"] == 2
    assert "stochastic_nll" not in saved["training"]
    assert saved["loss"]["evaluation"]["label_smoothing"] == 0.0

    figure = EpochEvidenceReport(tmp_path).plot_curves()
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert "Train objective, stochastic" in labels
    assert "Validation hard-label NLL" in labels
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_smoothed_training_with_opal_is_rejected(tmp_path: Path) -> None:
    """Reject an unverified live-agent and smoothing combination before training.

    Args:
        tmp_path (Path): Temporary synthetic configuration root.
    """
    config = _write_synthetic_config(tmp_path, {"label_smoothing": 0.05})
    settings = yaml.safe_load(config.read_text())
    settings["optimizer_agent"] = {"backend": "keep", "goal": "Keep", "decisions": 1, "every": 20}
    with pytest.raises(ValueError, match="OPAL"):
        TrainingConfig(settings)
