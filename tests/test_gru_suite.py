import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sleep_classifier.gru_suite import GRUExperimentSuite  # noqa: E402
from sleep_classifier.gru_training import TrainingConfig  # noqa: E402


def _suite_config(tmp_path):
    base = {
        "prepare_config": "missing-prepare.yaml",
        "output_dir": "original",
        "seed": 42,
        "epochs": 2,
        "batch_size": 2,
        "context_epochs": 10,
        "hidden_size": 32,
        "learning_rate": 0.001,
        "patience": 5,
        "gradient_clip": 1.0,
        "device": "cpu",
    }
    (tmp_path / "gru.yaml").write_text(yaml.safe_dump(base))
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_config": "gru.yaml",
                "output_dir": "results",
                "trials": [
                    {"name": "reference", "overrides": {}},
                    {"name": "context1", "overrides": {"context_epochs": 1}},
                ],
            }
        )
    )
    return path


class SyntheticExperiment:
    def __init__(self, config_path):
        self.settings = yaml.safe_load(config_path.read_text())

    def run(self):
        assert Path(self.settings["prepare_config"]).is_absolute()
        assert self.settings["device"] == "cpu"
        output = Path(self.settings["output_dir"])
        output.mkdir()
        metrics_path = output / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "classes": {},
                    "best_epoch": 1,
                    "history": [{}],
                    "device": "cpu",
                }
            )
        )
        return metrics_path


def test_plan_neither_loads_data_resolves_device_nor_creates_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingConfig, "device", Mock(side_effect=AssertionError("device")))
    suite = GRUExperimentSuite(_suite_config(tmp_path))

    plan = suite.plan()

    assert plan["maximum_epochs"] == 4
    assert plan["seed"] == 42
    assert not suite.output_dir.exists()


def test_suite_freezes_device_resolves_paths_and_exports_each_result(tmp_path, monkeypatch):
    monkeypatch.setattr("sleep_classifier.gru_suite.GRUExperiment", SyntheticExperiment)
    suite = GRUExperimentSuite(_suite_config(tmp_path))

    summary_path = suite.run()

    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "complete"
    assert [result["name"] for result in summary["results"]] == ["reference", "context1"]
    configs = suite.output_dir / "configs"
    reference = yaml.safe_load((configs / "reference.yaml").read_text())
    changed = yaml.safe_load((configs / "context1.yaml").read_text())
    assert {key for key in reference if reference[key] != changed[key]} == {
        "context_epochs",
        "output_dir",
    }
    assert reference["seed"] == changed["seed"] == 42
    assert reference["prepare_config"] == str(tmp_path / "missing-prepare.yaml")
    assert all(len(result["metrics_sha256"]) == 64 for result in summary["results"])


def test_suite_refuses_existing_outputs_before_running(tmp_path, monkeypatch):
    experiment = Mock(side_effect=AssertionError("training"))
    monkeypatch.setattr("sleep_classifier.gru_suite.GRUExperiment", experiment)
    suite = GRUExperimentSuite(_suite_config(tmp_path))
    suite.output_dir.mkdir()
    marker = suite.output_dir / "keep.txt"
    marker.write_text("existing results")

    with pytest.raises(FileExistsError):
        suite.run()

    assert marker.read_text() == "existing results"
    experiment.assert_not_called()


def test_suite_preserves_failure_status(tmp_path, monkeypatch):
    experiment = Mock()
    experiment.return_value.run.side_effect = RuntimeError("synthetic failure")
    monkeypatch.setattr("sleep_classifier.gru_suite.GRUExperiment", experiment)
    suite = GRUExperimentSuite(_suite_config(tmp_path))

    with pytest.raises(RuntimeError, match="synthetic failure"):
        suite.run()

    summary = json.loads((suite.output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_trial"] == "reference"
    assert summary["results"] == []


@pytest.mark.parametrize(
    "trial",
    [
        {"name": "../escape", "overrides": {"context_epochs": 1}},
        {"name": "reference", "overrides": {"context_epochs": 1}},
        {"name": "test", "overrides": {"seed": 99}},
        {"name": "test", "overrides": {"context_epochs": 1, "hidden_size": 64}},
        {"name": "test", "overrides": {"hidden_size": 32}},
        {"name": "test", "overrides": {}},
    ],
)
def test_suite_rejects_invalid_comparisons(tmp_path, trial):
    path = _suite_config(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["trials"][1] = trial
    path.write_text(yaml.safe_dump(settings))

    with pytest.raises(ValueError):
        GRUExperimentSuite(path)


@pytest.mark.parametrize("seeds", [[42, 42], [True, 43], [-1, 43], [42], "42,43"])
def test_suite_rejects_invalid_shared_seeds(tmp_path, seeds):
    path = _suite_config(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["seeds"] = seeds
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        GRUExperimentSuite(path)


def test_shared_seeds_pair_configs_and_aggregate_results(tmp_path, monkeypatch):
    monkeypatch.setattr("sleep_classifier.gru_suite.GRUExperiment", SyntheticExperiment)
    path = _suite_config(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["seeds"] = [42, 43, 44]
    path.write_text(yaml.safe_dump(settings))
    suite = GRUExperimentSuite(path)
    assert suite.plan()["maximum_epochs"] == 12
    summary = json.loads(suite.run().read_text())
    assert len(summary["results"]) == 6
    for seed in settings["seeds"]:
        configs = [
            yaml.safe_load(p.read_text())
            for p in (suite.output_dir / "configs").glob(f"*-seed-{seed}.yaml")
        ]
        assert len(configs) == 2
        assert all(config["seed"] == seed for config in configs)
    assert summary["seed_comparison"]["paired"]["context1"]["mean_difference"] == 0


def test_paired_summary_uses_matching_seeds(tmp_path):
    suite = GRUExperimentSuite(_suite_config(tmp_path))
    suite.seeds = [42, 43, 44]
    results = [
        {"recipe": recipe, "seed": seed, "macro_f1": value + improvement}
        for seed, value in [(42, 0.2), (43, 0.4), (44, 0.6)]
        for recipe, improvement in [("reference", 0), ("patience12", 0.1)]
    ]
    summary = suite._summarize_seeds(results)
    assert summary["recipes"]["reference"]["mean_macro_f1"] == pytest.approx(0.4)
    assert summary["recipes"]["reference"]["sample_std_macro_f1"] == pytest.approx(0.2)
    paired = summary["paired"]["patience12"]
    assert paired["mean_difference"] == pytest.approx(0.1)
    assert paired["sample_std_difference"] == pytest.approx(0)
