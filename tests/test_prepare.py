import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleep_classifier import cohort, data, prepare


def test_prepare_writes_development_features_without_loading_test_data(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cohort_path = config_dir / "cohort.yaml"
    cohort_path.write_text(
        "data_dir: ../raw\n"
        "output_dir: ../cohort\n"
        "split_path: ../cohort/split.json\n"
        "seed: 7\n"
        "test_count: 1\n"
        "validation_count: 1\n"
        "forced_development: []\n"
    )
    coverage_path = config_dir / "coverage.yaml"
    coverage_path.write_text(
        "cohort_config: cohort.yaml\n"
        "selected_rule: moderate\n"
        "candidates:\n"
        "  - name: moderate\n"
        "    max_heart_gap_seconds: 15\n"
        "    min_heart_readings: 3\n"
        "    min_motion_bin_fraction: 0.8\n"
        "    max_motion_gap_seconds: 5\n"
    )
    prepare_config = config_dir / "prepare.yaml"
    prepare_config.write_text("coverage_config: coverage.yaml\noutput_dir: ../features\n")

    config = cohort.load_config(cohort_path)
    subjects = ["100", "200", "300", "400"]
    inventory = {
        "version": 1,
        "participants": [
            {
                "participant_id": subject,
                "eligible": True,
                "reasons": [],
                "sha256": {modality: "verified" for modality in data.SENSORS},
            }
            for subject in subjects
        ],
    }
    config["output_dir"].mkdir(parents=True)
    (config["output_dir"] / "inventory.json").write_text(json.dumps(inventory))
    split = cohort.create_split(inventory, config)
    config["split_path"].parent.mkdir(parents=True, exist_ok=True)
    config["split_path"].write_text(json.dumps(split))

    loaded_paths = []
    hash_paths = []
    sensor_frames = {
        "heart_rate": pd.DataFrame(
            {"time": [0, 10, 20, 30, 40, 50], "heart_rate": [60, 61, 62, 63, 64, 65]}
        ),
        "motion": pd.DataFrame(
            {
                "time": np.arange(60, dtype=float),
                "x": np.zeros(60),
                "y": np.zeros(60),
                "z": np.ones(60),
            }
        ),
        "labels": pd.DataFrame({"time": [0, 30], "stage": [0, -1]}),
    }

    def load_mock(path: Path, sensor: str) -> pd.DataFrame:
        loaded_paths.append(path)
        return sensor_frames[sensor].copy()

    monkeypatch.setattr(prepare, "load_sensor", load_mock)
    monkeypatch.setattr(prepare, "sha256", lambda path: hash_paths.append(path) or "verified")

    result_dir = prepare.prepare(prepare_config)

    expected_development = split["train"] + split["validation"]
    test_ids = set(split["test"])
    loaded_ids = {path.name.split("_", 1)[0] for path in loaded_paths}
    hashed_ids = {path.name.split("_", 1)[0] for path in hash_paths}
    assert loaded_ids == set(expected_development)
    assert loaded_ids.isdisjoint(test_ids)
    assert hashed_ids.isdisjoint(test_ids)
    assert {path.parent.name for path in loaded_paths} == set(data.SENSORS) - {"steps"}
    assert {path.name for path in (result_dir / "train").glob("*.csv")} == {
        f"{subject}.csv" for subject in split["train"]
    }
    assert {path.name for path in (result_dir / "validation").glob("*.csv")} == {
        f"{subject}.csv" for subject in split["validation"]
    }
    assert not (result_dir / "test").exists()

    provenance = json.loads((result_dir / "provenance.json").read_text())
    assert provenance["split"] == split
    assert provenance["steps_used"] is False
    assert provenance["scaling_fitted"] is False
    assert provenance["test_observations_read"] is False
    assert {item["eligible_epochs"] for item in provenance["participants"]} == {1}
    for partition in ("train", "validation"):
        for subject in split[partition]:
            table = pd.read_csv(result_dir / partition / f"{subject}.csv")
            assert table.target.tolist() == [0, -1]
            assert table.eligible.tolist() == [True, False]
            assert table.heart_sample_count.tolist() == [3, 3]
            assert table.motion_sample_count.tolist() == [30, 30]
