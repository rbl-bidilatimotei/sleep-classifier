"""Prepare causal development features without opening test observations."""

import argparse
import json
from pathlib import Path

import yaml

from sleep_classifier.cohort import create_split, load_config
from sleep_classifier.data import load_sensor, sensor_path, sha256
from sleep_classifier.features import FEATURE_COLUMNS, build_epoch_features


def prepare(config_path: Path) -> Path:
    """Write development feature tables with raw-file and split provenance.

    Args:
        config_path (Path): Preparation YAML path.

    Returns:
        Path: Generated feature directory.
    """
    settings = yaml.safe_load(config_path.read_text())
    coverage_path = config_path.parent / settings["coverage_config"]
    coverage = yaml.safe_load(coverage_path.read_text())
    config = load_config(coverage_path.parent / coverage["cohort_config"])
    inventory = json.loads((config["output_dir"] / "inventory.json").read_text())
    split = json.loads(config["split_path"].read_text())
    if split != create_split(inventory, config):
        raise ValueError("Split differs from the saved inventory and configuration")
    rules = [r for r in coverage["candidates"] if r["name"] == coverage["selected_rule"]]
    if len(rules) != 1:
        raise ValueError("Exactly one selected coverage rule is required")
    records = {r["participant_id"]: r for r in inventory["participants"]}
    output = config_path.parent / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for partition in ("train", "validation"):
        for subject in split[partition]:
            frames = {}
            for sensor in ("heart_rate", "motion", "labels"):
                path = sensor_path(config["data_dir"], subject, sensor)
                if sha256(path) != records[subject]["sha256"][sensor]:
                    raise ValueError(f"Raw file changed since inventory: {path}")
                frames[sensor] = load_sensor(path, sensor)
            table = build_epoch_features(
                frames["heart_rate"], frames["motion"], frames["labels"], rules[0]
            )
            directory = output / partition
            directory.mkdir(parents=True, exist_ok=True)
            table.to_csv(directory / f"{subject}.csv", index=False)
            summaries.append(
                {
                    "participant_id": subject,
                    "partition": partition,
                    "epochs": len(table),
                    "eligible_epochs": int(table.eligible.sum()),
                }
            )
            print(f"Prepared {partition}: {subject}", flush=True)
    provenance = {
        "split": split,
        "rule": rules[0],
        "source_inventory": inventory,
        "epoch_start_assumed": True,
        "steps_used": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "scaling_fitted": False,
        "test_observations_read": False,
        "participants": summaries,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.config))


if __name__ == "__main__":
    main()
