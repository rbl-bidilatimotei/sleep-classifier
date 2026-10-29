import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

BASE_URL = "https://physionet.org/files/sleep-accel/1.0.0/"
SENSORS = {
    "heart_rate": ("heartrate", ["time", "heart_rate"]),
    "motion": ("acceleration", ["time", "x", "y", "z"]),
    "steps": ("steps", ["time", "steps"]),
    "labels": ("labeled_sleep", ["time", "stage"]),
}


def load_config(path: Path) -> dict[str, Path | str]:
    """Resolve experiment paths relative to the YAML file.
    
    Args:
        path (Path): Path to the YAML configuration file.
    
    Returns:
        dict[str, Path | str]: Resolved configuration dictionary.
    """
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    expected = {"subject_id", "data_dir", "report_path"}
    
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Config must contain exactly {sorted(expected)}")
    if not isinstance(config["subject_id"], str) or not re.fullmatch(r"\d+", config["subject_id"]):
        raise ValueError("subject_id must be a quoted numeric string")
    
    for key in ("data_dir", "report_path"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty path")
        config[key] = (path.parent / config[key]).resolve()
    
    return config


def sha256(path: Path) -> str:
    """Hash large motion files without loading them into memory.
    
    Args:
        path (Path): Path to the file to hash.
    
    Returns:
        str: SHA-256 hash of the file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url: str, destination: Path, expected_hash: str | None = None) -> None:
    """Download atomically, reusing valid files and rejecting corrupted local files.
    
    Args:
        url (str): URL of the file to download.
        destination (Path): Local path to save the downloaded file.
        expected_hash (str | None): Optional SHA-256 hash to verify the file.
    
    Raises:
        ValueError: If the file's checksum does not match the expected hash.
    """
    if destination.exists():
        if expected_hash and sha256(destination) != expected_hash:
            raise ValueError(f"Checksum mismatch: {destination}; inspect before removing it")
        return
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if expected_hash and sha256(temporary) != expected_hash:
            raise ValueError(f"Downloaded checksum mismatch: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def manifest(root: Path) -> dict[str, str]:
    """Read the publisher's filename-to-checksum mapping.
    
    Args:
        root (Path): Root directory containing the SHA256SUMS.txt file.
    
    Returns:
        dict[str, str]: Mapping from filenames to their SHA-256 checksums.
    """
    return {
        name: checksum
        for checksum, name in (
            line.split() for line in (root / "SHA256SUMS.txt").read_text().splitlines()
        )
    }


def sensor_path(root: Path, subject: str, sensor: str) -> Path:
    """Construct the path to a sensor file for a given subject.

    Args:
        root (Path): Root directory containing sensor files.
        subject (str): Subject identifier.
        sensor (str): Sensor name.

    Returns:
        Path: Full path to the sensor file.
    """
    return root / sensor / f"{subject}_{SENSORS[sensor][0]}.txt"


def download_subject(root: Path, subject: str) -> None:
    """Fetch four sensor files and license, verifying publisher checksums.
    
    Args:
        root (Path): Root directory to store downloaded files.
        subject (str): Subject identifier.
    
    Raises:
        ValueError: If any file is absent from the publisher manifest or has a checksum mismatch.
    """
    fetch_file(BASE_URL + "SHA256SUMS.txt", root / "SHA256SUMS.txt")
    checksums = manifest(root)
    paths = [Path("LICENSE.txt")] + [
        sensor_path(root, subject, sensor).relative_to(root) for sensor in SENSORS
    ]
    for relative in paths:
        name = relative.as_posix()
        if name not in checksums:
            raise ValueError(f"File is absent from publisher manifest: {name}")
        print(f"Verify/download {name}", flush=True)
        fetch_file(BASE_URL + name, root / relative, checksums[name])


def load_sensor(path: Path, sensor: str) -> pd.DataFrame:
    """Read numeric raw observations without sorting, deduplicating, or imputing.

    Args:
        path (Path): Path to the sensor file.
        sensor (str): Sensor name.

    Returns:
        pd.DataFrame: DataFrame containing the sensor data.

    Raises:
        ValueError: If the file has an invalid shape or contains nonfinite values.
    """
    columns = SENSORS[sensor][1]
    
    # The release mixes comma-delimited sensor files and whitespace labels/steps
    with path.open() as stream:
        separator = "," if "," in stream.readline() else r"\s+"
    
    frame = pd.read_csv(path, sep=separator, header=None)
    if frame.shape[1] != len(columns) or frame.empty:
        raise ValueError(f"Invalid shape for {sensor}: {frame.shape}")
    
    frame.columns = columns
    frame = frame.astype(float)
    
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Nonfinite values in {path}")
    
    return frame


def summarize(frame: pd.DataFrame, sensor: str) -> dict[str, Any]:
    """Describe ordering and coverage without silently repairing data.
    
    Args:
        frame (pd.DataFrame): Sensor data.
        sensor (str): Sensor name.

    Returns:
        dict[str, Any]: Summary statistics for the sensor data.
    """
    times = frame["time"]
    deltas = times.diff().dropna()
    unique_times = np.sort(times.unique())
    result = {
        "rows": len(frame),
        "min_time": float(times.min()),
        "max_time": float(times.max()),
        "backward_jumps": int((deltas < 0).sum()),
        "duplicate_timestamps": int(times.duplicated().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "conflicting_duplicate_timestamps": int(
            frame.loc[times.duplicated(keep=False)]
            .groupby("time")
            .nunique()
            .gt(1)
            .any(axis=1)
            .sum()
        ),
        "median_sorted_unique_interval": (
            float(np.median(np.diff(unique_times))) if len(unique_times) > 1 else None
        ),
        "largest_sorted_gap": (
            float(np.diff(unique_times).max()) if len(unique_times) > 1 else None
        ),
    }
    if sensor == "labels":
        result["label_counts"] = {
            str(stage): int(count) for stage, count in frame.stage.value_counts().items()
        }
        result["unexpected_labels"] = sorted(
            float(stage) for stage in set(frame.stage) - {0, 1, 2, 3, 5}
        )
        result["non_30_second_intervals"] = int((deltas != 30).sum())
    
    if sensor == "steps":
        result["negative_counts"] = int((frame.steps < 0).sum())
        result["fractional_counts"] = int((frame.steps % 1 != 0).sum())
        # Only bins whose next timestamp is observed have a known endpoint.
        result["complete_bins_in_previous_24h"] = int(
            ((times >= -86400) & (times.shift(-1) <= 0) & (times.shift(-1) > times)).sum()
        )
    
    return result


def audit_subject(root: Path, subject: str) -> dict[str, Any]:
    """Verify and summarize all modalities for one participant.
    
    Args:
        root (Path): Root directory of the dataset.
        subject (str): Subject identifier.

    Returns:
        dict[str, Any]: Summary and verification results for the subject.
    """
    checksums = manifest(root)
    result = {"subject_id": subject, "dataset": BASE_URL, "sensors": {}}
    for sensor in SENSORS:
        path = sensor_path(root, subject, sensor)
        digest = sha256(path)
        if digest != checksums[path.relative_to(root).as_posix()]:
            raise ValueError(f"Checksum mismatch: {path}")
        result["sensors"][sensor] = {
            "sha256": digest,
            **summarize(load_sensor(path, sensor), sensor),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["download", "audit"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.operation == "download":
        download_subject(config["data_dir"], config["subject_id"])
    else:
        report = audit_subject(config["data_dir"], config["subject_id"])
        config["report_path"].parent.mkdir(parents=True, exist_ok=True)
        config["report_path"].write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
