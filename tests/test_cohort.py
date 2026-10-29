import json
from pathlib import Path

import pandas as pd
import pytest

from sleep_classifier import cohort, data


def _frames(conflict: bool = False) -> dict[str, pd.DataFrame]:
    heart_times = [0, 30, 60]
    heart_rates = [60, 61, 62]
    if conflict:
        heart_times = [0, 0, 30]
        heart_rates = [60, 61, 62]
    return {
        "heart_rate": pd.DataFrame({"time": heart_times, "heart_rate": heart_rates}),
        "motion": pd.DataFrame(
            {"time": [0, 30], "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 1.0]}
        ),
        "steps": pd.DataFrame({"time": [0, 30], "steps": [0.0, 1.0]}),
        "labels": pd.DataFrame({"time": [0.0, 30.0], "stage": [0.0, 5.0]}),
    }


def _dataset(tmp_path: Path, subjects: dict[str, set[str]] | None = None) -> Path:
    root = tmp_path / "raw"
    if subjects is None:
        subjects = {"100": set(data.SENSORS)}
    manifest_lines = []
    for subject, modalities in subjects.items():
        for modality in modalities:
            path = data.sensor_path(root, subject, modality)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{subject}-{modality}\n")
            manifest_lines.append(f"{data.sha256(path)}  {path.relative_to(root).as_posix()}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n")
    return root


def _mock_load(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
    monkeypatch.setattr(data, "load_sensor", lambda _path, modality: frames[modality].copy())


def test_inventory_is_eligibility_only_and_flags_missing_modality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset(tmp_path, {"100": set(data.SENSORS), "200": {"motion"}})
    _mock_load(monkeypatch, _frames())

    inventory = cohort.build_inventory(root)

    assert [record["participant_id"] for record in inventory["participants"]] == ["100", "200"]
    assert inventory["participants"][0]["eligible"] is True
    assert inventory["participants"][1]["eligible"] is False
    assert "missing_modality:labels" in inventory["participants"][1]["reasons"]
    assert set(inventory["participants"][0]) == {"participant_id", "eligible", "reasons", "sha256"}
    assert "label_counts" not in json.dumps(inventory)


@pytest.mark.parametrize(
    ("frames", "reason"),
    [
        (_frames(conflict=True), "conflicting_duplicate_timestamps:heart_rate"),
        (
            {**_frames(), "labels": pd.DataFrame({"time": [0.0, 31.0], "stage": [0.0, 5.0]})},
            "labels_not_sorted_unique_30_second_grid",
        ),
        (
            {**_frames(), "labels": pd.DataFrame({"time": [0.0, 30.0], "stage": [0.0, 99.0]})},
            "invalid_label_stage",
        ),
        (
            {**_frames(), "steps": pd.DataFrame({"time": [0, 30], "steps": [-1.0, 0.5]})},
            "invalid_steps:fractional",
        ),
    ],
)
def test_inventory_marks_invalid_records_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: dict[str, pd.DataFrame],
    reason: str,
) -> None:
    root = _dataset(tmp_path)
    _mock_load(monkeypatch, frames)

    record = cohort.build_inventory(root)["participants"][0]

    assert record["eligible"] is False
    assert reason in record["reasons"]


def test_inventory_records_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset(tmp_path)
    path = data.sensor_path(root, "100", "motion")
    path.write_text("modified\n")
    _mock_load(monkeypatch, _frames())

    record = cohort.build_inventory(root)["participants"][0]

    assert "checksum_mismatch:motion" in record["reasons"]
    assert record["eligible"] is False


def test_split_is_reproducible_disjoint_and_preserves_forced_development() -> None:
    inventory = {
        "participants": [
            {"participant_id": f"{number:03d}", "eligible": True} for number in range(10)
        ]
    }
    config = {
        "seed": 42,
        "test_count": 2,
        "validation_count": 2,
        "forced_development": ["000"],
    }

    first = cohort.create_split(inventory, config)
    second = cohort.create_split(inventory, config)

    assert first == second
    assert "000" not in first["test"]
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set(first["train"] + first["validation"] + first["test"]) == {
        f"{number:03d}" for number in range(10)
    }


def test_split_requires_eligible_forced_ids() -> None:
    inventory = {"participants": [{"participant_id": "001", "eligible": False}]}
    config = {"seed": 1, "test_count": 1, "validation_count": 1, "forced_development": ["001"]}

    with pytest.raises(ValueError, match="not eligible"):
        cohort.create_split(inventory, config)


def test_immutable_write_reuses_identical_content_and_rejects_changes(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    payload = {"test": ["1"]}
    cohort._write_immutable(path, payload)
    original = path.read_text()

    cohort._write_immutable(path, payload)
    assert path.read_text() == original
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cohort._write_immutable(path, {"test": ["2"]})


def test_legacy_n4_and_unscored_labels_do_not_exclude_subject(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    frames = _frames()
    frames["labels"] = pd.DataFrame({"time": [0, 30, 60], "stage": [-1, 4, 5]})
    _mock_load(monkeypatch, frames)
    assert cohort.build_inventory(root)["participants"][0]["eligible"]
