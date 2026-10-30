import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sleep_classifier import data

MODALITIES = tuple(data.SENSORS)
VALID_STAGES = {-1, 0, 1, 2, 3, 4, 5}


def load_config(path: Path) -> dict[str, Any]:
    """Load cohort settings and resolve paths relative to the YAML file.

    Args:
        path (Path): Cohort configuration path.

    Returns:
        dict[str, Any]: Validated settings with resolved paths.
    """
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    expected = {
        "data_dir",
        "output_dir",
        "split_path",
        "seed",
        "test_count",
        "validation_count",
        "forced_development",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Config must contain exactly {sorted(expected)}")
    for key in ("data_dir", "output_dir", "split_path"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty path")
        config[key] = (path.parent / config[key]).resolve()
    for key in ("seed", "test_count", "validation_count"):
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f"{key} must be an integer")
    if config["test_count"] < 1 or config["validation_count"] < 1:
        raise ValueError("test_count and validation_count must be positive")
    forced = config["forced_development"]
    if not isinstance(forced, list) or any(
        not isinstance(subject, str) or not re.fullmatch(r"\d+", subject) for subject in forced
    ):
        raise ValueError("forced_development must be a list of quoted numeric IDs")
    if len(set(forced)) != len(forced):
        raise ValueError("forced_development IDs must be unique")
    return config


def _participant_files(root: Path, checksums: dict[str, str]) -> dict[str, dict[str, str]]:
    """Map manifest-listed participant files to modality and relative filename.
    
    Args:
        root (Path): Root directory containing participant files.
        checksums (dict[str, str]): Mapping of relative filenames to SHA-256 checksums.

    Returns:
        dict[str, dict[str, str]]: Mapping of participant IDs to modality and relative filename.
    """
    result: dict[str, dict[str, str]] = {}
    for filename in checksums:
        path = Path(filename)
        if len(path.parts) != 2 or path.parts[0] not in MODALITIES:
            continue
        suffix = f"_{data.SENSORS[path.parts[0]][0]}.txt"
        if path.name.endswith(suffix):
            subject = path.name.removesuffix(suffix)
            if re.fullmatch(r"\d+", subject):
                result.setdefault(subject, {})[path.parts[0]] = filename
    return result


def _conflicting_duplicates(frame: Any) -> bool:
    """Check if a sensor frame has conflicting duplicate timestamps.

    Args:
        frame (Any): Sensor data frame with a "time" column.

    Returns:
        bool: True if there are conflicting duplicate timestamps, False otherwise.
    """
    duplicates = frame.loc[frame["time"].duplicated(keep=False)]
    if duplicates.empty:
        return False
    return bool(duplicates.groupby("time").nunique().gt(1).any(axis=1).any())


def _eligible_checks(
    subject: str, root: Path, checksums: dict[str, str], files: dict[str, str]
) -> dict[str, Any]:
    """Check the eligibility of a participant's sensor files based on checksums and data validity.

    Args:
        subject (str): Participant ID.
        root (Path): Root directory containing participant files.
        checksums (dict[str, str]): Mapping of relative filenames to SHA-256 checksums.
        files (dict[str, str]): Mapping of modality to relative filename for the participant.

    Returns:
        dict[str, Any]: Dictionary containing reasons for ineligibility, hashes, and loaded frames.
    """
    reasons: list[str] = []
    hashes: dict[str, str] = {}
    frames: dict[str, Any] = {}
    for modality in MODALITIES:
        filename = files.get(modality)
        if filename is None:
            reasons.append(f"missing_modality:{modality}")
            continue
        path = root / filename
        if not path.is_file():
            reasons.append(f"missing_file:{modality}")
            continue
        digest = data.sha256(path)
        hashes[modality] = digest
        if digest != checksums[filename]:
            reasons.append(f"checksum_mismatch:{modality}")
            continue
        try:
            frames[modality] = data.load_sensor(path, modality)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append(f"invalid_sensor:{modality}:{type(exc).__name__}")

    for modality in MODALITIES:
        if modality == "labels":
            continue
        frame = frames.get(modality)
        if frame is not None and _conflicting_duplicates(frame):
            reasons.append(f"conflicting_duplicate_timestamps:{modality}")

    labels = frames.get("labels")
    if labels is not None:
        times = labels["time"].to_numpy(dtype=float)
        stages = labels["stage"].to_numpy(dtype=float)
        if not np.isin(stages, list(VALID_STAGES)).all():
            reasons.append("invalid_label_stage")
        if len(times):
            on_grid = np.isclose(
                (times - times[0]) / 30,
                np.rint((times - times[0]) / 30),
                rtol=0,
                atol=1e-8,
            )
            if np.any(np.diff(times) <= 0) or not on_grid.all():
                reasons.append("labels_not_sorted_unique_30_second_grid")
        else:
            reasons.append("empty_labels")

        scored_times = times[stages != -1]
        if not len(scored_times):
            reasons.append("no_scored_epochs")
        if len(scored_times) and np.all(np.diff(times) > 0):
            for modality in ("heart_rate", "motion"):
                frame = frames.get(modality)
                if frame is None:
                    continue
                observed = np.sort(frame["time"].to_numpy(dtype=float))
                positions = np.searchsorted(observed, scored_times, side="left")
                inside = positions < len(observed)
                inside[inside] &= observed[positions[inside]] < scored_times[inside] + 30
                if not inside.any():
                    reasons.append(f"no_observations_in_labeled_interval:{modality}")

    steps = frames.get("steps")
    if steps is not None:
        values = steps["steps"].to_numpy(dtype=float)
        if np.any(values < 0):
            reasons.append("invalid_steps:negative")
        if np.any(values % 1 != 0):
            reasons.append("invalid_steps:fractional")

    return {
        "participant_id": subject,
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "sha256": dict(sorted(hashes.items())),
    }


def build_inventory(root: Path) -> dict[str, Any]:
    """Audit all participants listed in any publisher modality manifest.

    Args:
        root (Path): Dataset root containing the publisher manifest.

    Returns:
        dict[str, Any]: Eligibility records with raw-file hashes.
    """
    checksums = data.manifest(root)
    participants = _participant_files(root, checksums)
    records = [
        _eligible_checks(subject, root, checksums, files)
        for subject, files in sorted(participants.items())
    ]
    return {"version": 1, "participants": records}


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to a file, refusing to overwrite if the content differs.

    Args:
        path (Path): Path to the JSON file to write.
        payload (dict[str, Any]): JSON-serializable dictionary to write.

    Raises:
        FileExistsError: If the file already exists and its content differs from the payload.
    """
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to overwrite different existing file: {path}")
        return
    path.write_text(content)


def create_split(inventory: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Select deterministic participant splits from an eligibility inventory.

    Args:
        inventory (dict[str, Any]): Output from ``build_inventory``.
        config (dict[str, Any]): Validated cohort settings.

    Returns:
        dict[str, Any]: Train, validation, and test participant IDs.
    """
    eligible = sorted(
        record["participant_id"] for record in inventory["participants"] if record["eligible"]
    )
    forced = sorted(config["forced_development"])
    missing_forced = sorted(set(forced) - set(eligible))
    if missing_forced:
        raise ValueError(f"Forced development IDs are not eligible: {missing_forced}")
    candidates = [subject for subject in eligible if subject not in forced]
    test_count = config["test_count"]
    validation_count = config["validation_count"]
    if test_count + validation_count >= len(eligible):
        raise ValueError("Requested counts leave no training participants")
    if test_count > len(candidates):
        raise ValueError("Not enough non-forced participants for test split")

    rng = random.Random(config["seed"])
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    test = sorted(shuffled[:test_count])
    development = sorted(subject for subject in eligible if subject not in test)
    rng.shuffle(development)
    validation = sorted(development[:validation_count])
    train = sorted(subject for subject in development if subject not in validation)
    if not train or not validation or not test:
        raise ValueError("All splits must be nonempty")
    return {
        "inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest(),
        "version": 1,
        "seed": config["seed"],
        "test_count": test_count,
        "validation_count": validation_count,
        "forced_development": forced,
        "train": train,
        "validation": validation,
        "test": test,
    }


def main() -> None:
    """Run inventory or participant split operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["inventory", "split"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    inventory_path = config["output_dir"] / "inventory.json"
    if args.operation == "inventory":
        inventory = build_inventory(config["data_dir"])
        _write_immutable(inventory_path, inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        inventory = json.loads(inventory_path.read_text())
        split = create_split(inventory, config)
        _write_immutable(config["split_path"], split)
        print(json.dumps(split, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
