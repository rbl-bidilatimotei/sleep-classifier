from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sleep_classifier.opal_smoke import OpalSmokeExperiment  # noqa: E402


def test_offline_keep_preserves_adam_updates(tmp_path: Path) -> None:
    pytest.importorskip("opal")
    result = OpalSmokeExperiment(tmp_path / "smoke", steps=4).run()
    assert result["agent_calls"] == 0
    assert result["max_loss_difference"] == 0
    assert result["plain_parameter_sha256"] == result["wrapped_parameter_sha256"]
    assert (tmp_path / "smoke/summary.json").exists()


def test_comparison_rejects_changed_parameters(tmp_path: Path) -> None:
    experiment = OpalSmokeExperiment(tmp_path)
    plain = {"losses": [1.0], "state": {"weight": torch.tensor([0.0])}}
    changed = {"losses": [1.0], "state": {"weight": torch.tensor([1.0])}}
    with pytest.raises(AssertionError):
        experiment._compare(plain, changed)


def test_existing_smoke_output_is_preserved(tmp_path: Path) -> None:
    pytest.importorskip("opal")
    with pytest.raises(FileExistsError):
        OpalSmokeExperiment(tmp_path, steps=4).run()
