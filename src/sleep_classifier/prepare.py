import argparse
import json
from pathlib import Path

import yaml

from sleep_classifier.cohort import create_split, load_config
from sleep_classifier.data import load_sensor, sensor_path, sha256
from sleep_classifier.features import FEATURE_COLUMNS, EpochFeatureBuilder


class FeaturePreparationJob:
    """Prepare checksum-verified development feature tables and provenance."""

    def __init__(self, config_path: Path) -> None:
        """Store the preparation settings path."""
        self.config_path = config_path

    def run(self) -> Path:
        """Write train and validation features without reading test observations."""
        context = self._load_context()
        output = context["output"]
        output.mkdir(parents=True, exist_ok=True)
        summaries = []
        builder = EpochFeatureBuilder(context["rule"])
        for partition in ("train", "validation"):
            for participant_id in context["split"][partition]:
                frames = self._load_participant_frames(
                    participant_id, context["records"], context["config"]
                )
                table = builder.build(frames["heart_rate"], frames["motion"], frames["labels"])
                summaries.append(self._write_participant(output, partition, participant_id, table))
                print(f"Prepared {partition}: {participant_id}", flush=True)
        provenance = self._provenance(context, summaries)
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return output

    def _load_context(self) -> dict:
        """Load settings, inventory, split, selected rule, and output path."""
        settings = yaml.safe_load(self.config_path.read_text())
        coverage_path = self.config_path.parent / settings["coverage_config"]
        coverage = yaml.safe_load(coverage_path.read_text())
        config = load_config(coverage_path.parent / coverage["cohort_config"])
        inventory = json.loads((config["output_dir"] / "inventory.json").read_text())
        split = json.loads(config["split_path"].read_text())
        if split != create_split(inventory, config):
            raise ValueError("Split differs from the saved inventory and configuration")
        rule = self._selected_rule(coverage)
        return {
            "config": config,
            "inventory": inventory,
            "split": split,
            "rule": rule,
            "records": {record["participant_id"]: record for record in inventory["participants"]},
            "output": self.config_path.parent / settings["output_dir"],
        }

    def _selected_rule(self, coverage: dict) -> dict:
        """Return the single rule selected by the coverage configuration."""
        rules = [
            candidate
            for candidate in coverage["candidates"]
            if candidate["name"] == coverage["selected_rule"]
        ]
        if len(rules) != 1:
            raise ValueError("Exactly one selected coverage rule is required")
        return rules[0]

    def _load_participant_frames(
        self, participant_id: str, records: dict, config: dict
    ) -> dict[str, object]:
        """Verify and load only the participant's required raw sensor files."""
        frames = {}
        for sensor in ("heart_rate", "motion", "labels"):
            path = sensor_path(config["data_dir"], participant_id, sensor)
            if sha256(path) != records[participant_id]["sha256"][sensor]:
                raise ValueError(f"Raw file changed since inventory: {path}")
            frames[sensor] = load_sensor(path, sensor)
        return frames

    def _write_participant(self, output: Path, partition: str, participant_id: str, table) -> dict:
        """Write one feature table and return its provenance summary."""
        directory = output / partition
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / f"{participant_id}.csv", index=False)
        return {
            "participant_id": participant_id,
            "partition": partition,
            "epochs": len(table),
            "eligible_epochs": int(table.eligible.sum()),
        }

    def _provenance(self, context: dict, summaries: list[dict]) -> dict:
        """Build the preparation provenance document."""
        return {
            "split": context["split"],
            "rule": context["rule"],
            "source_inventory": context["inventory"],
            "epoch_start_assumed": True,
            "steps_used": False,
            "feature_columns": list(FEATURE_COLUMNS),
            "scaling_fitted": False,
            "test_observations_read": False,
            "participants": summaries,
        }


def prepare(config_path: Path) -> Path:
    return FeaturePreparationJob(config_path).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare development feature tables")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.config))


if __name__ == "__main__":
    main()
