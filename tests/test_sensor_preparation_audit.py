import json
from functools import partial
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
import yaml

from sleep_classifier import sensor_audit_source
from sleep_classifier.data import load_sensor, sensor_path, sha256
from sleep_classifier.features import EpochFeatureBuilder
from sleep_classifier.motion_grid import FixedLagMotionFeatures
from sleep_classifier.sensor_preparation_audit import LabelFreeSensorAudit

RULE = {
    "max_heart_gap_seconds": 15,
    "min_heart_readings": 3,
    "min_motion_bin_fraction": 0.8,
    "max_motion_gap_seconds": 5,
}


def _write_motion_snapshot(root: Path, rows: pd.DataFrame) -> None:
    """Write the relevant data and checksums of a historical motion snapshot.

    Args:
        root (Path): Existing snapshot directory.
        rows (pd.DataFrame): Fixed-lag motion summaries and participant IDs.
    """
    rows.to_csv(root / "rows.csv", index=False)
    (root / "summary.json").write_text(json.dumps({
        "development_participants": ["101", "102"],
        "reserved_test_participants": ["103"],
        "motion_grid": {
            "step_seconds": 1.0,
            "max_sample_age_seconds": 0.2,
            "points_per_epoch": 29,
            "pairs_per_epoch": 28,
        },
    }))
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "outputs": {name: sha256(root / name) for name in ("summary.json", "rows.csv")},
    }))


def _read_motion_and_change_config(
    audit: LabelFreeSensorAudit, source: sensor_audit_source.DevelopmentSensorSource,
) -> pd.DataFrame:
    """Change the clock configuration after the audit starts reading observations.

    Args:
        audit (LabelFreeSensorAudit): Audit with an in-progress run.
        source (DevelopmentSensorSource): Validated development source.

    Returns:
        pd.DataFrame: Verified historical motion rows.
    """
    motion = LabelFreeSensorAudit._motion_reference(audit, source)
    config = yaml.safe_load(audit.config_path.read_text())
    config["session_start_seconds"] = 30.0
    audit.config_path.write_text(yaml.safe_dump(config))
    return motion


@pytest.fixture
def audit(tmp_path: Path) -> LabelFreeSensorAudit:
    """Set up an audit with a label exclusion, sensor gap, and distinct endpoints.

    Args:
        tmp_path (Path): Temporary directory for all synthetic sources and results.

    Returns:
        LabelFreeSensorAudit: Configured audit with unpublished results.
    """
    times = np.arange(-30, 1201, dtype=float)
    times = times[(times < 120) | (times >= 150)]
    heart = pd.DataFrame({"time": times[::10], "heart_rate": 60 + times[::10] / 10})
    motion = pd.DataFrame({"time": times, "x": 1 + times % 2, "y": 0.0, "z": 0.0})
    legacy_epochs = pd.DataFrame({"time": [*range(0, 1170, 30), 1200], "stage": 5})
    legacy_epochs.loc[legacy_epochs.time.eq(30), "stage"] = -1
    reference = EpochFeatureBuilder(RULE).build(heart, motion, legacy_epochs)
    motion_reference = FixedLagMotionFeatures().motion_summaries(motion, legacy_epochs)

    prepared, raw = tmp_path / "prepared", tmp_path / "raw"
    split = {"train": ["101"], "validation": ["102"], "test": ["103"]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    for partition in ("train", "validation"):
        directory = prepared / partition
        directory.mkdir(parents=True)
        reference.to_csv(directory / f"{split[partition][0]}.csv", index=False)
    finaltest = tmp_path / "finaltest"
    finaltest.mkdir()
    reference.to_csv(finaltest / "103.csv", index=False)

    inventory = []
    raw_frames = {"heart_rate": heart, "motion": motion, "labels": legacy_epochs}
    for participant_id in ("101", "102", "103"):
        hashes = {}
        for sensor, frame in raw_frames.items():
            path = sensor_path(raw, participant_id, sensor)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, header=False, index=False)
            hashes[sensor] = sha256(path)
        inventory.append({"participant_id": participant_id, "sha256": hashes})
    (prepared / "provenance.json").write_text(json.dumps({
        "split": split, "rule": RULE, "source_inventory": {"participants": inventory},
    }))

    snapshot = tmp_path / "motion"
    snapshot.mkdir()
    _write_motion_snapshot(snapshot, pd.concat([
        motion_reference.assign(participant_id=participant_id)
        for participant_id in ("101", "102")
    ], ignore_index=True))
    config = tmp_path / "audit.yaml"
    config.write_text(yaml.safe_dump({
        "prepared_dir": "prepared",
        "raw_data_dir": "raw",
        "split_path": "split.json",
        "motion_snapshot_dir": "motion",
        "output_dir": "output",
        "session_start_seconds": 0.0,
    }))
    return LabelFreeSensorAudit(config)


def test_audit_reports_eligibility_and_context(audit: LabelFreeSensorAudit) -> None:
    """Distinguish eligibility effects from a shared gap and unequal endpoints.

    Args:
        audit (LabelFreeSensorAudit): Audit with known comparison cases.
    """
    assert audit.run() == audit.output_dir
    rows, summary = audit.load_report()
    participant = rows.loc[rows.participant_id.eq("101")].set_index("time")
    history = ["history_epochs", "history_epochs_reference"]

    assert participant.loc[30, "label_only_exclusion"]
    assert participant.loc[30, "eligible"]
    assert not participant.loc[30, "eligible_reference"]
    assert participant.loc[60, history].tolist() == [3, 1]
    assert participant.loc[60, "context_changed"]

    assert not participant.loc[120, ["eligible", "eligible_reference"]].any()
    assert participant.loc[120, history].tolist() == [0, 0]
    assert participant.loc[150, history].tolist() == [1, 1]
    assert not participant.loc[150, "context_changed"]
    assert participant.loc[990, history].tolist() == [29, 29]
    assert participant.loc[1020, history].tolist() == [30, 30]
    assert participant.loc[1140, history].tolist() == [30, 30]

    assert participant.loc[0, "epoch_scope"] == "both"
    assert participant.loc[1170, "epoch_scope"] == "left_only"
    assert participant.loc[1200, "epoch_scope"] == "right_only"
    assert summary["status"] == "passed"
    assert summary["status_scope"] == "numerical_features_and_sensor_coverage"
    assert not summary["shared_eligible_context_parity"]
    for record in summary["participants"]:
        assert record["clock_only_epochs"] == record["reference_only_epochs"] == 1
        assert record["clock_only_eligible_epochs"] == 1
        assert record["label_only_exclusions"] == 1
        assert record["context_changed_epochs"] == 2
        assert record["eligibility_rule_mismatches"] == 0
        assert not any(record["feature_mismatches"].values())
    assert summary["motion_parity"] == {
        "compared_epochs": 78,
        "saved_epochs_outside_clock": 2,
        "mismatches": 0,
        "maximum_difference": 0.0,
    }
    causes = audit.context_causes(rows).set_index("participant_id")
    assert causes.loc["101", "changed_epochs"] == 2
    assert causes.loc["101", "unscored_history_epochs"] == 2
    assert causes.loc["101", "unscheduled_history_epochs"] == 0
    assert causes.loc["101", "boundary_history_epochs"] == 0


def test_audit_reports_overlapping_context_causes(audit: LabelFreeSensorAudit) -> None:
    """Count an earlier reference start and a history with two contributing causes.

    Args:
        audit (LabelFreeSensorAudit): Audit with raw observations before its clock origin.
    """
    raw = audit.raw_data_dir
    heart = load_sensor(sensor_path(raw, "101", "heart_rate"), "heart_rate")
    motion = load_sensor(sensor_path(raw, "101", "motion"), "motion")
    epochs = pd.DataFrame({"time": [*range(-30, 1170, 30), 1200], "stage": 5})
    epochs = epochs.loc[epochs.time.ne(60)]
    epochs.loc[epochs.time.eq(30), "stage"] = -1
    reference = EpochFeatureBuilder(RULE).build(heart, motion, epochs)
    reference.to_csv(audit.prepared_dir / "train" / "101.csv", index=False)
    snapshot = audit.motion_snapshot_dir
    saved = pd.read_csv(snapshot / "rows.csv", dtype={"participant_id": str})
    earlier_epoch = epochs.loc[epochs.time.eq(-30), ["time"]]
    earlier_motion = FixedLagMotionFeatures().motion_summaries(motion, earlier_epoch)
    _write_motion_snapshot(snapshot, pd.concat([
        saved, earlier_motion.assign(participant_id="101"),
    ], ignore_index=True))

    audit.run()
    rows, summary = audit.load_report()
    causes = audit.context_causes(rows).set_index("participant_id")
    participant = rows.loc[rows.participant_id.eq("101")].set_index("time")
    assert participant.index[participant.context_changed].tolist() == [0, 90]
    assert summary["participants"][0]["reference_only_eligible_epochs"] == 1
    assert causes.loc["101", "changed_epochs"] == 2
    assert causes.loc["101", "unscored_history_epochs"] == 1
    assert causes.loc["101", "unscheduled_history_epochs"] == 1
    assert causes.loc["101", "boundary_history_epochs"] == 1


def test_audit_uses_chronological_reference_context(audit: LabelFreeSensorAudit) -> None:
    """Preserve the context comparison when saved reference rows are reordered.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit with ordered references.
    """
    audit.run()
    ordered_rows, ordered_summary = audit.load_report()

    reference_path = audit.prepared_dir / "train" / "101.csv"
    reference = pd.read_csv(reference_path)
    reference.iloc[::-1].to_csv(reference_path, index=False)
    config = yaml.safe_load(audit.config_path.read_text())
    config["output_dir"] = "shuffled-output"
    shuffled_config = audit.config_path.with_name("shuffled.yaml")
    shuffled_config.write_text(yaml.safe_dump(config))

    shuffled_audit = LabelFreeSensorAudit(shuffled_config)
    shuffled_audit.run()
    shuffled_rows, shuffled_summary = shuffled_audit.load_report()
    context = ["participant_id", "time", "history_epochs_reference", "context_changed"]
    pd.testing.assert_frame_equal(ordered_rows[context], shuffled_rows[context])
    pd.testing.assert_frame_equal(audit.context_causes(ordered_rows),
                                  shuffled_audit.context_causes(shuffled_rows))
    assert shuffled_summary["participants"] == ordered_summary["participants"]


def test_audit_reads_only_development_inputs(
    audit: LabelFreeSensorAudit, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check metadata-only planning and the observation allowlist during the run.

    Args:
        audit (LabelFreeSensorAudit): Audit with present raw labels and test data.
        monkeypatch (pytest.MonkeyPatch): Fixture for recording input reads.
    """
    load_sensor = Mock(wraps=sensor_audit_source.load_sensor)
    read_csv = Mock(wraps=pd.read_csv)
    monkeypatch.setattr(sensor_audit_source, "load_sensor", load_sensor)
    monkeypatch.setattr(pd, "read_csv", read_csv)

    plan = audit.plan()
    assert plan["development_participants"] == ["101", "102"]
    assert plan["reserved_test_participants"] == ["103"]
    assert not plan["model_loaded"]
    assert not plan["training_performed"]
    load_sensor.assert_not_called()
    read_csv.assert_not_called()

    audit.run()
    raw_paths = [
        (sensor_path(audit.raw_data_dir, participant, sensor), sensor)
        for participant in ("101", "102")
        for sensor in ("heart_rate", "motion")
    ]
    assert load_sensor.call_args_list == [call(path, sensor) for path, sensor in raw_paths]
    assert [entry.args[0] for entry in read_csv.call_args_list] == [
        raw_paths[0][0], raw_paths[1][0], audit.prepared_dir / "train" / "101.csv",
        raw_paths[2][0], raw_paths[3][0], audit.prepared_dir / "validation" / "102.csv",
        audit.motion_snapshot_dir / "rows.csv",
    ]
    _, summary = audit.load_report()
    assert str(audit.config_path.parent / "finaltest" / "103.csv") not in summary["input_hashes"]
    for participant in ("101", "102", "103"):
        path = sensor_path(audit.raw_data_dir, participant, "labels")
        assert str(path) not in summary["input_hashes"]


def test_audit_reports_numeric_and_missing_motion_differences(audit: LabelFreeSensorAudit) -> None:
    """Count numeric and finite-to-missing differences while matching paired NaNs.

    Args:
        audit (LabelFreeSensorAudit): Audit with initially matching references.
    """
    reference_path = audit.prepared_dir / "train" / "101.csv"
    reference = pd.read_csv(reference_path)
    reference.loc[reference.time.eq(0), "heart_rate_mean"] += 0.25
    reference.loc[reference.time.eq(30), "heart_rate_mean"] += 1e-9
    reference.to_csv(reference_path, index=False)

    motion_path = audit.motion_snapshot_dir / "rows.csv"
    motion = pd.read_csv(motion_path, dtype={"participant_id": str})
    feature = "motion_grid_change_mean"
    assert motion.loc[motion.time.eq(120), feature].isna().all()
    train = motion.participant_id.eq("101")
    motion.loc[train & motion.time.eq(0), feature] = np.nan
    motion.loc[train & motion.time.eq(30), feature] += 0.5
    _write_motion_snapshot(audit.motion_snapshot_dir, motion)

    audit.run()
    _, summary = audit.load_report()
    record = summary["participants"][0]
    assert summary["status"] == "differences_detected"
    assert record["feature_mismatches"]["heart_rate_mean"] == 1
    assert record["maximum_feature_differences"]["heart_rate_mean"] == pytest.approx(0.25)
    assert summary["motion_parity"]["mismatches"] == 2
    assert summary["motion_parity"]["maximum_difference"] == pytest.approx(0.5)


def test_cached_audit_loads_without_sources_and_refuses_overwrite(
    audit: LabelFreeSensorAudit, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read an unchanged snapshot without opening its source and prevent reruns.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
        monkeypatch (pytest.MonkeyPatch): Fixture for blocking source construction.
    """
    audit.run()
    original = {path.name: path.read_bytes() for path in audit.output_dir.iterdir()}
    source = Mock(side_effect=AssertionError("unexpected source access"))
    monkeypatch.setattr(audit, "_source", source)

    rows, summary = audit.load_report()
    assert set(rows.participant_id) == {"101", "102"}
    assert summary["status"] == "passed"
    with pytest.raises(FileExistsError, match="Audit exists"):
        audit.run()
    source.assert_not_called()
    assert {path.name: path.read_bytes() for path in audit.output_dir.iterdir()} == original


@pytest.mark.parametrize("filename", ["rows.csv", "summary.json"])
def test_cached_audit_rejects_modified_output(audit: LabelFreeSensorAudit, filename: str) -> None:
    """Reject cached rows or summary bytes that differ from the manifest.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
        filename (str): Snapshot file to modify after successful publication.
    """
    audit.run()
    output = audit.output_dir / filename
    output.write_bytes(output.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        audit.load_report()


def test_cached_audit_rejects_changed_configuration(audit: LabelFreeSensorAudit) -> None:
    """Reject a cached result when the declared session origin changes.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
    """
    audit.run()
    config = yaml.safe_load(audit.config_path.read_text())
    config["session_start_seconds"] = 30.0
    audit.config_path.write_text(yaml.safe_dump(config))
    for configured_audit in (audit, LabelFreeSensorAudit(audit.config_path)):
        with pytest.raises(ValueError, match="configuration differs"):
            configured_audit.load_report()


def test_audit_rejects_config_changes_after_construction(audit: LabelFreeSensorAudit) -> None:
    """Require a new audit instance to run with a new declared session origin.

    Args:
        audit (LabelFreeSensorAudit): Audit initialized before changing the configuration.
    """
    config = yaml.safe_load(audit.config_path.read_text())
    config["session_start_seconds"] = 30.0
    audit.config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration"):
        audit.run()
    assert not audit.output_dir.exists()

    current_audit = LabelFreeSensorAudit(audit.config_path)
    current_audit.run()
    rows, summary = current_audit.load_report()
    assert summary["plan"]["session_start_seconds"] == 30.0
    assert rows.loc[rows.epoch_scope.ne("right_only"), "time"].min() == 30.0


def test_audit_rejects_config_changes_during_run(
    audit: LabelFreeSensorAudit, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discard the audit if its configuration changes during observation reads.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
        monkeypatch (pytest.MonkeyPatch): Fixture for injecting the configuration edit.
    """
    monkeypatch.setattr(audit, "_motion_reference", partial(_read_motion_and_change_config, audit))
    with pytest.raises(ValueError, match="configuration changed during the run"):
        audit.run()
    assert not audit.output_dir.exists()


def test_audit_rejects_tampered_motion_snapshot(audit: LabelFreeSensorAudit) -> None:
    """Reject changed historical motion rows before publishing a new report.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
    """
    motion_path = audit.motion_snapshot_dir / "rows.csv"
    motion_path.write_bytes(motion_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Motion snapshot checksum differs"):
        audit.run()
    assert not audit.output_dir.exists()


@pytest.mark.parametrize("remove_participant", [True, False], ids=["participant", "one-epoch"])
def test_audit_requires_complete_eligible_motion_coverage(
    audit: LabelFreeSensorAudit, remove_participant: bool,
) -> None:
    """Reject missing saved motion identities even when the snapshot checksums match.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
        remove_participant (bool): Remove all validation rows or one eligible epoch.
    """
    snapshot = audit.motion_snapshot_dir
    motion = pd.read_csv(snapshot / "rows.csv", dtype={"participant_id": str})
    validation = motion.participant_id.eq("102")
    missing = validation if remove_participant else validation & motion.time.eq(0)
    _write_motion_snapshot(snapshot, motion.loc[~missing])

    expected = "unique development identities" if remove_participant else "every prepared eligible"
    with pytest.raises(ValueError, match=expected):
        audit.run()
    assert not audit.output_dir.exists()


def test_audit_requires_motion_overlap_for_each_participant(audit: LabelFreeSensorAudit) -> None:
    """Reject a complete saved motion snapshot whose validation rows precede the clock.

    Args:
        audit (LabelFreeSensorAudit): Audit with observations before the configured origin.
    """
    raw = audit.raw_data_dir
    heart = load_sensor(sensor_path(raw, "102", "heart_rate"), "heart_rate")
    motion = load_sensor(sensor_path(raw, "102", "motion"), "motion")
    epochs = pd.DataFrame({"time": [-30, 0, 30], "stage": [5, -1, -1]})
    reference = EpochFeatureBuilder(RULE).build(heart, motion, epochs)
    reference.to_csv(audit.prepared_dir / "validation" / "102.csv", index=False)
    eligible_epochs = reference.loc[reference.eligible, ["time"]]
    earlier_motion = FixedLagMotionFeatures().motion_summaries(motion, eligible_epochs)
    snapshot = audit.motion_snapshot_dir
    saved = pd.read_csv(snapshot / "rows.csv", dtype={"participant_id": str})
    _write_motion_snapshot(snapshot, pd.concat([
        saved.loc[saved.participant_id.eq("101")], earlier_motion.assign(participant_id="102"),
    ], ignore_index=True))

    with pytest.raises(ValueError, match="clock overlap for every participant"):
        audit.run()
    assert not audit.output_dir.exists()


def test_audit_requires_shared_epochs_for_each_participant(audit: LabelFreeSensorAudit) -> None:
    """Stop when one participant has no reference epochs on the sensor clock.

    Args:
        audit (LabelFreeSensorAudit): Configured synthetic audit.
    """
    reference_path = audit.prepared_dir / "validation" / "102.csv"
    reference = pd.read_csv(reference_path)
    reference["time"] += 3000
    reference.to_csv(reference_path, index=False)

    with pytest.raises(ValueError, match="overlapping epoch starts"):
        audit.run()
    assert not audit.output_dir.exists()
