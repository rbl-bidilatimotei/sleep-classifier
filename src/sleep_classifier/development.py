import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sleep_classifier.cohort import create_split, load_config
from sleep_classifier.coverage import epoch_coverage, step_bins
from sleep_classifier.data import load_sensor, sensor_path, sha256


def analyze(config_path: Path) -> Path:
    """Save participant coverage and candidate retention summaries.

    Args:
        config_path (Path): Coverage YAML with cohort settings and candidate rules.

    Returns:
        Path: Generated development report directory.
    """
    settings = yaml.safe_load(config_path.read_text())
    if settings.get("selected_rule") not in {r["name"] for r in settings["candidates"]}:
        raise ValueError("selected_rule must identify a configured candidate")
    config = load_config(config_path.parent / settings["cohort_config"])
    inventory = json.loads((config["output_dir"] / "inventory.json").read_text())
    split = json.loads(config["split_path"].read_text())
    if split != create_split(inventory, config):
        raise ValueError("Split does not match the current inventory and settings")
    records = {r["participant_id"]: r for r in inventory["participants"]}
    participants = sorted(split["train"] + split["validation"])
    if set(participants) & set(split["test"]):
        raise ValueError("Development participants overlap the test set")
    rows, retained = [], []
    output = config["output_dir"] / "development"
    output.mkdir(parents=True, exist_ok=True)
    for subject in participants:
        print(f"Development coverage: {subject}", flush=True)

        def read(sensor):
            path = sensor_path(config["data_dir"], subject, sensor)
            if sha256(path) != records[subject]["sha256"][sensor]:
                raise ValueError(f"Raw file changed since inventory: {path}")
            return load_sensor(path, sensor)

        labels = read("labels")
        labels = labels.loc[labels.stage != -1]
        heart = epoch_coverage(read("heart_rate"), labels)
        motion = epoch_coverage(read("motion"), labels)
        steps = step_bins(read("steps"))
        # Endpoints identify recorded bins; overlap does not allocate partial step counts
        end = steps.end.clip(upper=0)
        start = steps.start.clip(lower=-86400)
        step_hours = (end - start).clip(lower=0).sum() / 3600
        row = {
            "participant_id": subject,
            "scored_epochs": len(labels),
            "heart_empty_fraction": float((heart.sample_count == 0).mean()),
            "motion_empty_fraction": float((motion.sample_count == 0).mean()),
            "previous_day_step_bin_hours": float(step_hours),
        }
        rows.append(row)
        for rule in settings["candidates"]:
            valid = (
                (heart.sample_count >= rule["min_heart_readings"])
                & (heart.largest_observation_gap <= rule["max_heart_gap_seconds"])
                & (motion.sample_count > 0)
                & (motion.occupied_bin_fraction >= rule["min_motion_bin_fraction"])
                & (motion.largest_observation_gap <= rule["max_motion_gap_seconds"])
            )
            retained.append(
                {
                    "participant_id": subject,
                    "rule": rule["name"],
                    "scored_epochs": len(labels),
                    "retained_epochs": int(valid.sum()),
                    "retained_fraction": float(valid.mean()),
                }
            )
    table = pd.DataFrame(rows)
    retention = pd.DataFrame(retained)
    table.to_csv(output / "participants.csv", index=False)
    retention.to_csv(output / "candidate-retention.csv", index=False)
    aggregate = retention.groupby("rule")[["scored_epochs", "retained_epochs"]].sum()
    aggregate["retained_fraction"] = aggregate.retained_epochs / aggregate.scored_epochs
    aggregate.to_csv(output / "aggregate-retention.csv")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    table.set_index("participant_id")[["heart_empty_fraction", "motion_empty_fraction"]].plot.bar(
        ax=axes[0]
    )
    retention.pivot(index="participant_id", columns="rule", values="retained_fraction").plot.bar(
        ax=axes[1]
    )
    table.set_index("participant_id").previous_day_step_bin_hours.plot.bar(ax=axes[2])
    axes[0].set_ylabel("Empty epoch fraction")
    axes[1].set_ylabel("Retained scored fraction")
    axes[2].set_ylabel("Prior-day step-bin hours")
    axes[2].set_xlabel("Development participant")
    fig.tight_layout()
    fig.savefig(output / "development-coverage.png", dpi=120)
    plt.close(fig)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "split": split,
                "candidates": settings["candidates"],
                "selected_rule": settings["selected_rule"],
                "epoch_start_assumed": True,
                "unscored_epochs_excluded": True,
                "test_files_read": False,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.config))


if __name__ == "__main__":
    main()
