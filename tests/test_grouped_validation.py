import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sleep_classifier.baseline import PreparedFeatureDataset  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.grouped_validation import GroupedValidationExperiment  # noqa: E402


def _configuration(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    participants = [str(index) for index in range(1, 9)]
    split = {"train": participants[:6], "validation": participants[6:], "test": ["reserved"]}
    summaries = []
    for partition in ("train", "validation"):
        directory = source / partition
        directory.mkdir(parents=True)
        for participant in split[partition]:
            values = {column: [1.0, 2.0, 0.0] for column in FEATURE_COLUMNS}
            values.update(
                time=[0, 30, 60],
                target=[0, 1, -1],
                eligible=[True, True, False],
                segment_id=[0, 0, -1],
            )
            pd.DataFrame(values).to_csv(directory / f"{participant}.csv", index=False)
            summaries.append(
                {
                    "partition": partition,
                    "participant_id": participant,
                    "epochs": 3,
                    "eligible_epochs": 2,
                }
            )
    (source / "test").mkdir()
    (source / "test/reserved.csv").write_text("DO NOT READ FINAL TEST OBSERVATIONS")
    (source / "provenance.json").write_text(
        json.dumps(
            {
                "split": split,
                "participants": summaries,
                "feature_columns": list(FEATURE_COLUMNS),
                "scaling_fitted": False,
                "test_observations_read": False,
            }
        )
    )
    (tmp_path / "prepare.yaml").write_text(yaml.safe_dump({"output_dir": "source"}))
    model = {
        "prepare_config": "prepare.yaml",
        "output_dir": "original",
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
    (tmp_path / "gru.yaml").write_text(yaml.safe_dump(model))
    path = tmp_path / "grouped.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source_prepare_config": "prepare.yaml",
                "base_model_config": "gru.yaml",
                "output_dir": "grouped",
                "folds": 3,
                "split_seed": 42,
            }
        )
    )
    return path


class SyntheticFoldExperiment:
    def __init__(self, config_path):
        self.config_path = config_path

    def run(self):
        settings = yaml.safe_load(self.config_path.read_text())
        assert settings["device"] == "cpu"
        prepare = self.config_path.parent / settings["prepare_config"]
        features = prepare.parent / yaml.safe_load(prepare.read_text())["output_dir"]
        data = PreparedFeatureDataset(features).load()
        validation = data["partitions"]["validation"]
        predictions = validation[["participant_id", "time", "target"]].copy()
        predictions["gru_prediction"] = predictions.target
        participants = {}
        for participant, frame in validation.groupby("participant_id"):
            participants[participant] = {"eligible_epochs": len(frame), "macro_f1": 0.5}
        metrics = {
            "accuracy": 1.0,
            "macro_f1": 0.5,
            "best_epoch": 1,
            "classes": {},
            "coverage": {},
            "participants": participants,
        }
        output = self.config_path.parent / settings["output_dir"]
        output.mkdir()
        predictions.to_csv(output / "validation_predictions.csv", index=False)
        path = output / "metrics.json"
        path.write_text(json.dumps(metrics))
        return path


def test_plan_reads_metadata_only_and_holds_out_every_development_person_once(
    tmp_path, monkeypatch
):
    experiment = GroupedValidationExperiment(_configuration(tmp_path))
    monkeypatch.setattr(
        PreparedFeatureDataset, "load", Mock(side_effect=AssertionError("data read"))
    )
    plan = experiment.plan()
    held_out = [person for fold in plan["folds"] for person in fold["validation"]]
    assert sorted(held_out) == plan["development"]
    assert len(held_out) == len(set(held_out)) == 8
    assert all(not set(fold["train"]) & set(fold["validation"]) for fold in plan["folds"])
    assert "reserved" not in held_out
    assert not experiment.output_dir.exists()
    assert experiment.plan() == plan


def test_prepare_preserves_full_tables_and_never_copies_test(tmp_path):
    experiment = GroupedValidationExperiment(_configuration(tmp_path))
    manifest_path = experiment.prepare()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["device"] == "cpu"
    for fold in manifest["plan"]["folds"]:
        root = experiment.output_dir / fold["name"]
        data = PreparedFeatureDataset(root / "features").load()
        assert set(data["partitions"]["train"].participant_id) == set(fold["train"])
        assert set(data["partitions"]["validation"].participant_id) == set(fold["validation"])
        for path in (root / "features").rglob("*.csv"):
            assert len(pd.read_csv(path)) == 3
            assert path.stem != "reserved"
        assert not (root / "results").exists()
    assert not any("reserved.csv" in name for name in manifest["source_hashes"])
    with pytest.raises(FileExistsError):
        experiment.prepare()


def test_mocked_fold_runs_pool_every_eligible_epoch_exactly_once(tmp_path, monkeypatch):
    experiment = GroupedValidationExperiment(_configuration(tmp_path))
    experiment.prepare()
    monkeypatch.setattr(
        "sleep_classifier.grouped_validation.GRUExperiment", SyntheticFoldExperiment
    )
    summary = json.loads(experiment.run().read_text())
    assert summary["status"] == "complete"
    assert len(summary["results"]) == 3
    aggregate = summary["aggregate"]
    assert aggregate["pooled_accuracy"] == 1.0
    assert aggregate["pooled_macro_f1"] == 0.5
    assert aggregate["sample_std_fold_macro_f1"] == 0
    assert aggregate["eligible_epochs"] == 16
    assert aggregate["validation_participants"] == 8
    assert aggregate["independent_outer_test"] is False
    with pytest.raises(FileExistsError):
        experiment.run()


@pytest.mark.parametrize("mutation", ["source", "recipe", "derived"])
def test_changed_inputs_are_rejected_before_any_training(tmp_path, monkeypatch, mutation):
    path = _configuration(tmp_path)
    experiment = GroupedValidationExperiment(path)
    experiment.prepare()
    if mutation == "source":
        changed = tmp_path / "source/train/1.csv"
    elif mutation == "recipe":
        changed = tmp_path / "gru.yaml"
    else:
        changed = experiment.output_dir / "fold-01/gru.yaml"
    changed.write_text(changed.read_text() + "\n")
    model = Mock(side_effect=AssertionError("training must not start"))
    monkeypatch.setattr("sleep_classifier.grouped_validation.GRUExperiment", model)
    with pytest.raises(ValueError, match="changed"):
        experiment.run()
    model.assert_not_called()


def test_fold_failure_preserves_summary_and_completed_results(tmp_path, monkeypatch):
    experiment = GroupedValidationExperiment(_configuration(tmp_path))
    experiment.prepare()
    model = Mock()
    model.return_value.run.side_effect = RuntimeError("synthetic failure")
    monkeypatch.setattr("sleep_classifier.grouped_validation.GRUExperiment", model)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        experiment.run()
    summary = json.loads((experiment.output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_fold"] == "fold-01"
    assert summary["results"] == []


@pytest.mark.parametrize("folds", [True, 1, 9])
def test_invalid_fold_counts_are_rejected(tmp_path, folds):
    path = _configuration(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["folds"] = folds
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        GroupedValidationExperiment(path).plan()


class FailingSecondFoldExperiment(SyntheticFoldExperiment):
    def run(self):
        if self.config_path.parent.name == "fold-02":
            raise RuntimeError("second fold failed")
        return super().run()


def test_later_fold_failure_retains_completed_fold_artifacts(tmp_path, monkeypatch):
    experiment = GroupedValidationExperiment(_configuration(tmp_path))
    experiment.prepare()
    monkeypatch.setattr(
        "sleep_classifier.grouped_validation.GRUExperiment", FailingSecondFoldExperiment
    )
    with pytest.raises(RuntimeError, match="second fold failed"):
        experiment.run()
    summary = json.loads((experiment.output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_fold"] == "fold-02"
    assert [result["fold"] for result in summary["results"]] == ["fold-01"]
    assert (experiment.output_dir / "fold-01/results/metrics.json").exists()
    with pytest.raises(FileExistsError):
        experiment.run()


def test_plan_rejects_development_overlap_with_reserved_participants(tmp_path):
    path = _configuration(tmp_path)
    provenance_path = tmp_path / "source/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["split"]["validation"].append("reserved")
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="disjoint"):
        GroupedValidationExperiment(path).plan()


def test_synthetic_real_fold_training_fits_each_scaler_on_its_training_people(tmp_path):
    import numpy as np
    import torch

    path = _configuration(tmp_path)
    for source in (tmp_path / "source").glob("*/*.csv"):
        if source.parent.name == "test":
            continue
        frame = pd.read_csv(source)
        for column in FEATURE_COLUMNS:
            frame.loc[:1, column] = [float(source.stem), float(source.stem) + 1.0]
        frame.to_csv(source, index=False)
    experiment = GroupedValidationExperiment(path)
    experiment.prepare()
    summary = json.loads(experiment.run().read_text())
    assert summary["status"] == "complete"
    for fold in summary["plan"]["folds"]:
        checkpoint = torch.load(
            experiment.output_dir / fold["name"] / "results/gru_checkpoint.pt", weights_only=True
        )
        expected_mean = np.mean([float(person) + 0.5 for person in fold["train"]])
        np.testing.assert_allclose(checkpoint["scaler_mean"], expected_mean)
    assert summary["aggregate"]["eligible_epochs"] == 16


def test_synthetic_transformer_folds_use_generic_predictions(tmp_path: Path) -> None:
    """Run tiny Transformer folds without changing the grouped evaluation interface.

    Args:
        tmp_path (Path): Temporary root for synthetic development participants.
    """
    path = _configuration(tmp_path)
    model_path = tmp_path / "gru.yaml"
    settings = yaml.safe_load(model_path.read_text())
    settings.pop("hidden_size")
    settings["model_type"] = "transformer"
    settings["transformer"] = {
        "d_model": 8, "nhead": 2, "num_layers": 1, "feedforward_dim": 16,
        "dropout": 0.0, "max_context": 2,
    }
    model_path.write_text(yaml.safe_dump(settings))

    experiment = GroupedValidationExperiment(path)
    experiment.prepare()
    summary = json.loads(experiment.run().read_text())
    pooled = pd.read_csv(experiment.output_dir / "development_predictions.csv")

    assert summary["status"] == "complete"
    assert summary["aggregate"]["eligible_epochs"] == 16
    assert "prediction" in pooled
    assert "gru_prediction" not in pooled
    assert len(pooled) == 16
    for fold in summary["plan"]["folds"]:
        assert (experiment.output_dir / fold["name"] / "results/model_checkpoint.pt").is_file()
