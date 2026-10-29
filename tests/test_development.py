"""Ensure development reporting does not load test-participant observations."""

import json
from pathlib import Path

import pandas as pd
import yaml

from sleep_classifier import cohort, development


def test_development_report_loads_only_development_subjects(tmp_path, monkeypatch):
    inventory = {
        "participants": [
            {
                "participant_id": str(i),
                "eligible": True,
                "sha256": {
                    sensor: "hash" for sensor in ("heart_rate", "motion", "steps", "labels")
                },
            }
            for i in range(5)
        ]
    }
    config = {
        "data_dir": tmp_path / "raw",
        "output_dir": tmp_path,
        "split_path": tmp_path / "split.json",
        "seed": 42,
        "test_count": 1,
        "validation_count": 1,
        "forced_development": ["0"],
    }
    split = cohort.create_split(inventory, config)
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    config["split_path"].write_text(json.dumps(split))
    settings = tmp_path / "coverage.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "cohort_config": "cohort.yaml",
                "selected_rule": "observed",
                "candidates": [
                    {
                        "name": "observed",
                        "min_heart_readings": 1,
                        "max_heart_gap_seconds": 30,
                        "min_motion_bin_fraction": 0,
                        "max_motion_gap_seconds": 30,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(development, "load_config", lambda _: config)
    loaded = set()

    def read(path: Path, sensor: str):
        subject = path.name.split("_")[0]
        assert subject not in split["test"]
        loaded.add(subject)
        if sensor == "labels":
            return pd.DataFrame({"time": [0, 30], "stage": [0, -1]})
        if sensor == "steps":
            return pd.DataFrame({"time": [-600, 0], "steps": [10, 0]})
        return pd.DataFrame({"time": [0, 10], "value": [1, 2]})

    monkeypatch.setattr(development, "sha256", lambda _: "hash")
    monkeypatch.setattr(development, "load_sensor", read)
    output = development.analyze(settings)
    assert loaded == set(split["train"] + split["validation"])
    table = pd.read_csv(output / "participants.csv")
    assert table.scored_epochs.tolist() == [1] * 4
