from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sleep_classifier.checkpoint_audit import CheckpointReplayAudit  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.sequences import SequenceBatchCollator  # noqa: E402


def _write_checkpoint_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a selected synthetic model and matching validation probabilities.

    Args:
        root (Path): Temporary fold directory.
        monkeypatch (pytest.MonkeyPatch): Replace only the prepared-file reader.
    """
    frame = pd.DataFrame({column: np.arange(3, dtype=float) for column in FEATURE_COLUMNS})
    frame = frame.assign(participant_id="p1", segment_id=0, time=[0, 30, 60],
                         target=[0, 1, 2], eligible=True)
    dataset = Mock()
    dataset.load.return_value = {"partitions": {"validation": frame}}
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.PreparedFeatureDataset",
                        Mock(return_value=dataset))
    factory = ModelFactory({"model_type": "transformer", "context_epochs": 3,
                            "transformer": {"max_context": 3, "dropout": 0.2}})
    model = factory.build(6, 4).eval()
    checkpoint = {"state_dict": model.state_dict(), "model_type": "transformer",
                  "model_config": factory.checkpoint_config(6, 4),
                  "features": list(FEATURE_COLUMNS), "scaler_mean": [1.0] * 6,
                  "scaler_scale": [2.0] * 6, "context_epochs": 3}
    results = root / "results"
    results.mkdir()
    torch.save(checkpoint, results / "model_checkpoint.pt")
    normalized = (frame.loc[:, FEATURE_COLUMNS].to_numpy() - 1.0) / 2.0
    samples = [{"features": torch.tensor(normalized[:index + 1], dtype=torch.float32),
                "target": torch.tensor(index), "participant_id": "p1", "time": index * 30}
               for index in range(3)]
    batch = SequenceBatchCollator()(samples)
    with torch.no_grad():
        probabilities = model(batch["features"], batch["lengths"]).softmax(dim=1).numpy()
    predictions = frame[["participant_id", "time", "target"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    for stage in range(4):
        predictions[f"probability_{stage}"] = probabilities[:, stage]
    predictions.to_csv(results / "validation_predictions.csv", index=False)

    return results / "model_checkpoint.pt"


def test_checkpoint_replay_uses_stored_scaling_and_never_updates_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce saved predictions twice without gradients or parameter changes.

    Args:
        tmp_path (Path): Temporary checkpoint directory.
        monkeypatch (pytest.MonkeyPatch): Isolate synthetic validation input loading.
    """
    checkpoint = _write_checkpoint_fixture(tmp_path, monkeypatch)
    before = checkpoint.read_bytes()
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.StandardScaler.fit",
                        Mock(side_effect=AssertionError("Replay must not fit preprocessing")))
    result = CheckpointReplayAudit(tmp_path).build()

    assert result["epochs"] == 3
    assert result["evaluation_mode"] and result["weights_unchanged"]
    assert not result["gradients_created"]
    assert result["repeat_max_abs_probability_difference"] == 0
    assert result["matches_saved_atol_2e_6"]
    assert result["validation_nll"] > 0
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("mutation", ["features", "scale", "targets"])
def test_checkpoint_replay_rejects_mismatched_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    """Reject incompatible features, invalid scaling, and changed target populations.

    Args:
        tmp_path (Path): Temporary checkpoint directory.
        monkeypatch (pytest.MonkeyPatch): Isolate synthetic validation input loading.
        mutation (str): Stored artifact inconsistency to introduce.
    """
    checkpoint_path = _write_checkpoint_fixture(tmp_path, monkeypatch)
    if mutation == "targets":
        path = checkpoint_path.parent / "validation_predictions.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "target"] = 3
        frame.to_csv(path, index=False)
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if mutation == "features":
            checkpoint["features"] = ["other"] * 6
        else:
            checkpoint["scaler_scale"][0] = 0
        torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError):
        CheckpointReplayAudit(tmp_path).build()
