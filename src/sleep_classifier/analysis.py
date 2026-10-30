import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sleep_classifier.coverage import deduplicate_sensor, epoch_coverage, step_bins
from sleep_classifier.data import audit_subject, load_config, load_sensor, sensor_path


def analyze(config_path: Path) -> Path:
    """Save coverage tables and figures for the configured development participant.

    Args:
        config_path (Path): Existing exploration configuration.

    Returns:
        Path: Directory containing the generated reports.
    """
    config = load_config(config_path)
    audit = audit_subject(config["data_dir"], config["subject_id"])
    output = config["report_path"].parent / f"{config['subject_id']}-coverage"
    output.mkdir(parents=True, exist_ok=True)
    
    frames = {
        name: load_sensor(sensor_path(config["data_dir"], config["subject_id"], name), name)
        for name in audit["sensors"]
    }
    labels = frames["labels"]
    summary = {"subject_id": config["subject_id"], "epoch_start_assumed": True, "sensors": {}}
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    for sensor in ("heart_rate", "motion"):
        clean = deduplicate_sensor(frames[sensor])
        table = epoch_coverage(clean, labels)
        table.to_csv(output / f"{sensor}-epochs.csv", index=False)
        summary["sensors"][sensor] = {
            "removed_exact_rows": len(frames[sensor]) - len(clean),
            "empty_epochs": int((table.sample_count == 0).sum()),
            "epochs": len(table),
            "largest_epoch_gap_seconds": float(table.largest_observation_gap.max()),
        }
        hours = table.time / 3600
        axes[0].plot(hours, table.sample_count, label=sensor)
        axes[1].plot(hours, table.occupied_bin_fraction, label=sensor)
        axes[2].plot(hours, table.largest_observation_gap, label=sensor)
    
    axes[0].set_ylabel("Unique readings / epoch")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].set_ylabel("Occupied 1-second bins")
    axes[2].set_ylabel("Largest gap (seconds)")
    axes[2].set_xlabel("Hours from PSG start")
    
    for axis in axes:
        axis.legend()
    
    fig.suptitle("Development participant coverage — assumed epoch starts")
    fig.tight_layout()
    fig.savefig(output / "coverage.png", dpi=120)
    plt.close(fig)

    bins = step_bins(frames["steps"])
    bins.to_csv(output / "step-bins.csv", index=False)
    fig, axis = plt.subplots(figsize=(12, 3))
    axis.scatter(bins.start / 86400, bins.steps, s=8)
    axis.axvspan(-1, 0, color="orange", alpha=0.2, label="Previous 24 hours")
    axis.set(xlabel="Days from PSG start", ylabel="Steps per recorded bin")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "steps.png", dpi=120)
    plt.close(fig)
    summary["step_bins_with_unknown_end"] = int(bins.end.isna().sum())
    summary["source_audit"] = audit
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(analyze(args.config))


if __name__ == "__main__":
    main()
