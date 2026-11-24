import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, call

import pandas as pd
import pytest

from sleep_classifier import sensor_audit_source
from sleep_classifier.data import sensor_path
from sleep_classifier.sensor_audit_source import DevelopmentSensorSource

RAW_CONTENTS = {
    "heart_rate": b"0,60\n30,61\n",
    "motion": b"0,1,0,0\n30,2,0,0\n",
    "labels": b"0 0\n30 5\n",
    "steps": b"0 1\n30 0\n",
}


@pytest.fixture
def source_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create frozen split metadata, prepared tables, and raw observations.

    Args:
        tmp_path (Path): Temporary test directory.

    Returns:
        tuple[Path, Path, Path]: Prepared directory, raw directory, and split path.
    """
    prepared = tmp_path / "prepared"
    raw = tmp_path / "raw"
    split_path = tmp_path / "split.json"
    split = {"train": ["101"], "validation": ["102"], "test": ["103"]}
    split_path.write_text(json.dumps(split))

    for partition in ("train", "validation"):
        directory = prepared / partition
        directory.mkdir(parents=True)
        (directory / f"{split[partition][0]}.csv").write_text("time,target\n0,1\n")

    records = []
    for participant in ("101", "102", "103"):
        checksums = {}
        for sensor, data in RAW_CONTENTS.items():
            path = sensor_path(raw, participant, sensor)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            checksums[sensor] = hashlib.sha256(data).hexdigest()
        records.append({"participant_id": participant, "sha256": checksums})
    (prepared / "provenance.json").write_text(
        json.dumps({"split": split, "rule": {}, "source_inventory": {"participants": records}})
    )
    return prepared, raw, split_path


def test_source_rejects_split_overlap(source_paths: tuple[Path, Path, Path]) -> None:
    """Reject a reserved participant listed in the training partition.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
    """
    prepared, _, split_path = source_paths
    split = json.loads(split_path.read_text())
    split["train"].append("103")
    split_path.write_text(json.dumps(split))
    provenance_path = prepared / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["split"] = split
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="disjoint"):
        DevelopmentSensorSource(*source_paths)


@pytest.mark.parametrize("unexpected", ["finaltest.csv", "train/103.csv"])
def test_source_rejects_unexpected_prepared_file(
    source_paths: tuple[Path, Path, Path], unexpected: str
) -> None:
    """Reject prepared entries outside the allowed development inventory.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
        unexpected (str): Extra test-result path relative to the prepared directory.
    """
    prepared, _, _ = source_paths
    (prepared / unexpected).write_text("time,target\n0,3\n")
    with pytest.raises(ValueError, match="unexpected entries|development allowlist"):
        DevelopmentSensorSource(*source_paths)


def test_source_only_loads_requested_development_sensors(
    source_paths: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Block test-participant files and raw labels before opening either.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
        monkeypatch (pytest.MonkeyPatch): Fixture for spying on file operations.
    """
    prepared, raw, _ = source_paths
    finaltest = prepared.parent / "finaltest"
    finaltest.mkdir()
    (finaltest / "103.csv").write_text("time,target\n0,3\n")
    loader = Mock(wraps=sensor_audit_source.load_sensor)
    hash_file = Mock(wraps=sensor_audit_source.sha256)
    read_csv = Mock(wraps=pd.read_csv)
    monkeypatch.setattr(sensor_audit_source, "load_sensor", loader)
    monkeypatch.setattr(sensor_audit_source, "sha256", hash_file)
    monkeypatch.setattr(sensor_audit_source.pd, "read_csv", read_csv)

    source = DevelopmentSensorSource(*source_paths)
    for participant, sensor in (("101", "labels"), ("102", "steps"), ("103", "heart_rate")):
        assert sensor_path(raw, participant, sensor).is_file()
        with pytest.raises(ValueError, match="development"):
            source.load_raw(participant, sensor)
    with pytest.raises(ValueError, match="development"):
        source.load_reference("103")
    loader.assert_not_called()
    read_csv.assert_not_called()
    assert hash_file.call_args_list == [
        call(prepared / "provenance.json"), call(source_paths[2])
    ]

    source.load_raw("101", "heart_rate")
    source.load_raw("102", "motion")
    assert loader.call_args_list == [
        call(sensor_path(raw, "101", "heart_rate"), "heart_rate"),
        call(sensor_path(raw, "102", "motion"), "motion"),
    ]


def test_source_records_verified_hashes(source_paths: tuple[Path, Path, Path]) -> None:
    """Read expected values and retain the digest of each input used.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
    """
    prepared, raw, split_path = source_paths
    source = DevelopmentSensorSource(*source_paths)
    heart = source.load_raw("101", "heart_rate")
    motion = source.load_raw("102", "motion")
    reference = source.load_reference("102")

    assert heart.to_dict(orient="list") == {"time": [0.0, 30.0], "heart_rate": [60.0, 61.0]}
    assert motion.to_dict(orient="list") == {
        "time": [0.0, 30.0], "x": [1.0, 2.0], "y": [0.0, 0.0], "z": [0.0, 0.0]
    }
    assert reference.to_dict(orient="list") == {"time": [0], "target": [1]}
    inputs = (
        prepared / "provenance.json", split_path, prepared / "validation" / "102.csv",
        sensor_path(raw, "101", "heart_rate"), sensor_path(raw, "102", "motion"),
    )
    assert source.input_hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }


def test_source_checks_raw_bytes_before_loading(
    source_paths: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject changed bytes even when the file remains valid CSV.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
        monkeypatch (pytest.MonkeyPatch): Fixture for spying on the raw loader.
    """
    _, raw, _ = source_paths
    source = DevelopmentSensorSource(*source_paths)
    path = sensor_path(raw, "101", "heart_rate")
    path.write_bytes(b"0,60\n30,62\n")
    loader = Mock(wraps=sensor_audit_source.load_sensor)
    monkeypatch.setattr(sensor_audit_source, "load_sensor", loader)

    with pytest.raises(ValueError, match="checksum"):
        source.load_raw("101", "heart_rate")
    loader.assert_not_called()
    assert str(path) not in source.input_hashes


def test_source_rejects_symlink_to_valid_raw_bytes(source_paths: tuple[Path, Path, Path]) -> None:
    """Reject a raw symlink even if its destination matches the checksum.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
    """
    _, raw, _ = source_paths
    source = DevelopmentSensorSource(*source_paths)
    link = sensor_path(raw, "101", "heart_rate")
    destination = sensor_path(raw, "102", "heart_rate")
    link.unlink()
    link.symlink_to(destination)

    with pytest.raises(ValueError, match="declared directory"):
        source.load_raw("101", "heart_rate")
    assert str(link) not in source.input_hashes


def test_source_rejects_prepared_symlink(source_paths: tuple[Path, Path, Path]) -> None:
    """Reject a prepared entry linked to another development table.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
    """
    prepared, _, _ = source_paths
    link = prepared / "train" / "101.csv"
    link.unlink()
    link.symlink_to(prepared / "validation" / "102.csv")

    with pytest.raises(ValueError, match="declared directory"):
        DevelopmentSensorSource(*source_paths)


def test_source_rejects_unknown_participant(source_paths: tuple[Path, Path, Path]) -> None:
    """Reject a participant absent from the frozen split.

    Args:
        source_paths (tuple[Path, Path, Path]): Synthetic source paths.
    """
    source = DevelopmentSensorSource(*source_paths)

    with pytest.raises(ValueError, match="development"):
        source.load_raw("999", "motion")
    with pytest.raises(ValueError, match="development"):
        source.load_reference("999")
