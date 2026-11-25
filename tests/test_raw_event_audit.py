import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sleep_classifier import raw_event_audit  # noqa: E402
from sleep_classifier.raw_event_audit import RawEventAudit  # noqa: E402
from sleep_classifier.recorded_event_replay import RecordedEventReplay  # noqa: E402

RULE = {
    "max_heart_gap_seconds": 15.0,
    "min_heart_readings": 3.0,
    "min_motion_bin_fraction": 0.8,
    "max_motion_gap_seconds": 5.0,
}
PEOPLE = ["001", "002"]
CHUNKS = [128, 1024]
FOLDS = [{"name": "fold-01", "validation": ["001"]},
         {"name": "fold-02", "validation": ["002"]}]
DIGEST = "a" * 64
requires_exclusive_rename = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="Requires an exclusive directory rename",
)


@pytest.fixture
def audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RawEventAudit:
    """Bind a raw audit to a temporary configuration and synthetic development sources.

    Args:
        tmp_path (Path): Root for temporary configuration and report files.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing source and backend dependencies.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "raw.yaml"
    config.write_text(yaml.safe_dump({
        "replay_config": "replay.yaml", "output_dir": "../raw-audit",
        "chunk_sizes": CHUNKS, "max_buffered_events": 8192,
    }))
    replay = Mock(output_dir=tmp_path / "replay", device="mps")
    replay.output_dir.mkdir()
    (replay.output_dir / "manifest.json").write_text("{}\n")
    replay.chronological.config_path = configs / "chronological.yaml"
    replay.chronological.reference.plan.return_value = {"reserved_test": ["003"]}
    replay.audit.output_dir = tmp_path / "sensor-audit"
    replay.audit.prepared_dir = tmp_path / "prepared"
    replay.audit.raw_data_dir = tmp_path / "raw"
    replay.audit.split_path = configs / "split.json"
    replay.audit.clock.start_seconds = 0.5
    replay.read_report.return_value = {
        "status": "passed", "pooled": {"scopes": {"candidate_epochs": 4}},
        "plan": {"folds": FOLDS, "chronological_plan": {"sources": {
            fold["name"]: {"model_checkpoint.pt": DIGEST} for fold in FOLDS
        }}},
    }
    source = Mock(partitions=dict.fromkeys(PEOPLE, "development"), reserved=["003"],
                  rule=RULE, input_hashes={"synthetic-source": DIGEST})
    export = Mock()
    export.protected_roots.return_value = (tmp_path / "frozen",)
    monkeypatch.setattr(raw_event_audit, "SensorOnlyReplayStudy", Mock(return_value=replay))
    monkeypatch.setattr(raw_event_audit, "EvaluationExportSource", Mock(return_value=export))
    monkeypatch.setattr(raw_event_audit, "DevelopmentSensorSource", Mock(
        return_value=source, _inside=raw_event_audit.DevelopmentSensorSource._inside,
    ))
    monkeypatch.setattr(raw_event_audit.StreamingEpochPredictor, "_resolve_device", Mock())
    return RawEventAudit(config)


@pytest.fixture
def report(audit: RawEventAudit) -> dict[str, Any]:
    """Create passing synthetic records with every declared participant, chunk, and fold.

    Args:
        audit (RawEventAudit): Audit with a verified synthetic replay plan.
    """
    return {
        "status": "passed", "training_performed": False, "test_observations_read": False,
        "plan": audit.plan(),
        "participants": [{
            "participant_id": person, "chunk_size": chunk, "epochs": 2, "emitted_epochs": 2,
            "feature_maximum_difference": 0.0, "feature_mismatches": 0,
            "historical_feature_maximum_difference": 0.0, "historical_feature_mismatches": 0,
            "eligibility_matches": True, "segments_match": True, "closed": True,
            "buffered_events": 0, "peak_buffered_events": 100, "event_conservation_matches": True,
        } for person in PEOPLE for chunk in CHUNKS],
        "prediction_parity": [{
            "fold": fold["name"], "checkpoint_sha256": DIGEST, "epochs": 2,
            "maximum_probability_difference": 0.0, "class_disagreements": 0,
            "history_lengths_match": True, "reset_reasons_match": True,
            "weights_unchanged": True, "saved_transforms_unchanged": True,
            "evaluation_mode": True, "gradients_created": False, "training_performed": False,
        } for fold in FOLDS],
    }


@pytest.fixture
def feature_rows() -> pd.DataFrame:
    """Create one eligible row and one ineligible row with missing motion."""
    return pd.DataFrame({
        "time": [0.0, 30.0], "heart_rate_mean": [65.0, 70.0],
        "motion_grid_change_mean": [1.5, np.nan], "eligible": [True, False],
        "segment_id": [0, -1], "available_at": [30.0, 60.0],
        "watermark_at_emission": [32.0, 60.0],
    })


def test_feature_parity_accepts_matching_nan_and_small_numeric_error(
    feature_rows: pd.DataFrame,
) -> None:
    """Compare eligible and ineligible rows with an absolute numerical tolerance.

    Args:
        feature_rows (pd.DataFrame): Synthetic expected feature rows.
    """
    actual = feature_rows.assign(heart_rate_mean=feature_rows.heart_rate_mean + 5e-9)
    expected = feature_rows.drop(columns=["available_at", "watermark_at_emission"])
    assert RawEventAudit._feature_parity(actual, expected) == {
        "feature_maximum_difference": pytest.approx(5e-9, abs=1e-13),
        "feature_mismatches": 0, "eligibility_matches": True, "segments_match": True,
    }


@pytest.mark.parametrize(("column", "value", "reason"), [
    ("time", 31.0, "epoch schedule"), ("heart_rate_mean", 71.0, "summaries"),
    ("motion_grid_change_mean", 2.0, "summaries"), ("eligible", True, "eligibility"),
    ("segment_id", 0, "resets"),
])
def test_feature_parity_rejects_mismatched_rows(
    feature_rows: pd.DataFrame, column: str, value: Any, reason: str,
) -> None:
    """Reject a changed schedule, summary, eligibility decision, or reset.

    Args:
        feature_rows (pd.DataFrame): Synthetic expected feature rows.
        column (str): Field to change in the ineligible row.
        value (Any): Mismatching field value.
        reason (str): Expected text in the error message.
    """
    actual = feature_rows.copy()
    actual.loc[1, column] = value
    expected = feature_rows.drop(columns=["available_at", "watermark_at_emission"])
    with pytest.raises(ValueError, match=reason):
        RawEventAudit._feature_parity(actual, expected)


def test_feature_parity_rejects_missing_epoch(feature_rows: pd.DataFrame) -> None:
    """Reject rows that omit a scheduled epoch.

    Args:
        feature_rows (pd.DataFrame): Synthetic expected feature rows.
    """
    with pytest.raises(ValueError, match="epoch schedule"):
        RawEventAudit._feature_parity(feature_rows.iloc[:1], feature_rows)


@pytest.mark.parametrize("column", ["available_at", "watermark_at_emission"])
def test_feature_parity_requires_timestamp_metadata(
    feature_rows: pd.DataFrame, column: str,
) -> None:
    """Require an availability timestamp and emission watermark on every output.

    Args:
        feature_rows (pd.DataFrame): Synthetic expected feature rows.
        column (str): Required timestamp field to remove.
    """
    expected = feature_rows.drop(columns=["available_at", "watermark_at_emission"])
    with pytest.raises(ValueError):
        RawEventAudit._feature_parity(feature_rows.drop(columns=column), expected)


@pytest.mark.parametrize(("column", "value"), [
    ("available_at", 59.0), ("watermark_at_emission", 59.0), ("watermark_at_emission", np.inf),
])
def test_feature_parity_checks_availability_before_emission(
    feature_rows: pd.DataFrame, column: str, value: float,
) -> None:
    """Validate that emission occurs at a finite watermark at or beyond the end.

    Args:
        feature_rows (pd.DataFrame): Synthetic expected feature rows.
        column (str): Required timestamp field to change.
        value (float): Invalid timestamp or watermark.
    """
    actual = feature_rows.copy()
    actual.loc[1, column] = value
    expected = feature_rows.drop(columns=["available_at", "watermark_at_emission"])
    with pytest.raises(ValueError):
        RawEventAudit._feature_parity(actual, expected)


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Create two normalized predictions and their consecutive context lengths."""
    return pd.DataFrame({
        "participant_id": ["001", "001"], "time": [0.5, 30.5],
        "prediction": [0, 1], "history_epochs": [1, 2],
        "reset_reason": ["first_epoch", ""], "probability_0": [0.7, 0.1],
        "probability_1": [0.1, 0.7], "probability_2": [0.1, 0.1], "probability_3": [0.1, 0.1],
    })


def test_prediction_parity_compares_probability_and_state(predictions: pd.DataFrame) -> None:
    """Allow floating error within tolerance and equivalent empty reset encodings.

    Args:
        predictions (pd.DataFrame): Synthetic label-free predictions.
    """
    actual = predictions.assign(
        reset_reason=["first_epoch", None], probability_0=predictions.probability_0 + 1e-7,
        probability_1=predictions.probability_1 - 1e-7,
    )
    assert RawEventAudit._prediction_parity(actual, predictions) == {
        "epochs": 2, "maximum_probability_difference": pytest.approx(1e-7),
        "class_disagreements": 0, "history_lengths_match": True, "reset_reasons_match": True,
    }


@pytest.mark.parametrize(("column", "value", "reason"), [
    ("participant_id", "002", "identities"), ("time", 31.5, "identities"),
    ("probability_1", np.inf, "finite"), ("probability_1", 0.6, "sum to 1"),
    ("prediction", 2, "argmax"), ("history_epochs", 1, "predictions or resets"),
    ("reset_reason", "sensor_gap", "predictions or resets"),
])
def test_prediction_parity_rejects_mismatches(
    predictions: pd.DataFrame, column: str, value: Any, reason: str,
) -> None:
    """Reject invalid probabilities and changed identity or context boundaries.

    Args:
        predictions (pd.DataFrame): Expected synthetic predictions.
        column (str): Field to change in the second prediction.
        value (Any): Mismatching field value.
        reason (str): Expected text in the error message.
    """
    actual = predictions.copy()
    actual.loc[1, column] = value
    with pytest.raises(ValueError, match=reason):
        RawEventAudit._prediction_parity(actual, predictions)


def test_prediction_parity_rejects_normalized_probability_difference(
    predictions: pd.DataFrame,
) -> None:
    """Reject normalized outputs whose probabilities differ beyond the tolerance.

    Args:
        predictions (pd.DataFrame): Expected synthetic predictions.
    """
    actual = predictions.assign(probability_0=predictions.probability_0 + 3e-6,
                                probability_1=predictions.probability_1 - 3e-6)
    with pytest.raises(ValueError, match="predictions or resets"):
        RawEventAudit._prediction_parity(actual, predictions)


@pytest.mark.parametrize("input_index", [0, 1])
def test_prediction_parity_rejects_targets_in_either_input(
    predictions: pd.DataFrame, input_index: int,
) -> None:
    """Require both freshly produced and cached predictions to be label-free.

    Args:
        predictions (pd.DataFrame): Valid label-free predictions.
        input_index (int): Input to which the test adds a target column.
    """
    inputs = [predictions, predictions]
    inputs[input_index] = predictions.assign(target=0)
    with pytest.raises(ValueError, match="target column"):
        RawEventAudit._prediction_parity(*inputs)


@pytest.mark.parametrize(("section", "column", "value", "reason"), [
    ("participants", "chunk_size", 1024, "participant and chunk"),
    ("participants", "participant_id", "003", "participant and chunk"),
    ("participants", "peak_buffered_events", 8193, "buffer conditions"),
    ("participants", "buffered_events", 1, "buffer conditions"),
    ("participants", "closed", False, "closure"),
    ("participants", "segments_match", False, "reset"),
    ("participants", "historical_feature_mismatches", 1, "conditions"),
    ("participants", "event_conservation_matches", False, "conditions"),
    ("prediction_parity", "fold", "fold-02", "every frozen fold"),
    ("prediction_parity", "epochs", 1, "predicted population"),
    ("prediction_parity", "checkpoint_sha256", "b" * 64, "identity or parity"),
    ("prediction_parity", "gradients_created", True, "identity or parity"),
    ("prediction_parity", "evaluation_mode", False, "identity or parity"),
    ("prediction_parity", "saved_transforms_unchanged", False, "identity or parity"),
    ("prediction_parity", "training_performed", True, "identity or parity"),
    ("prediction_parity", "weights_unchanged", False, "identity or parity"),
])
def test_report_validation_rejects_missing_coverage_or_failed_conditions(
    report: dict[str, Any], section: str, column: str, value: Any, reason: str,
) -> None:
    """Validate measured coverage, state, and buffer integrity before publication.

    Args:
        report (dict[str, Any]): Passing synthetic report.
        section (str): Record list to mutate.
        column (str): Record field to change.
        value (Any): Value that breaks an acceptance condition.
        reason (str): Expected text in the error message.
    """
    RawEventAudit._validate_report(report)
    report[section][0][column] = value
    with pytest.raises(ValueError, match=reason):
        RawEventAudit._validate_report(report)


@pytest.mark.parametrize("flag", ["training_performed", "test_observations_read"])
def test_report_validation_rejects_out_of_scope_activity(report: dict[str, Any], flag: str) -> None:
    """Reject reports that indicate training or reserved test-data access.

    Args:
        report (dict[str, Any]): Passing synthetic report.
        flag (str): Scope flag to set.
    """
    report[flag] = True
    with pytest.raises(ValueError, match="exclude training and final-test observations"):
        RawEventAudit._validate_report(report)


def test_participant_checks_raw_event_conservation(
    audit: RawEventAudit, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a replay result when a raw event is missing from the accounting.

    Args:
        audit (RawEventAudit): Audit configured to read synthetic sources.
        monkeypatch (pytest.MonkeyPatch): Fixture for changing the replay status.
    """
    heart = pd.DataFrame({"time": [0.5, 30.5], "heart_rate": [60.0, 70.0]})
    motion = pd.DataFrame({"time": [0.5, 30.5], "x": 1.0, "y": 0.0, "z": 0.0})
    historical = audit._batch(heart, motion, RULE).assign(participant_id="001")
    result, status = RecordedEventReplay(RULE, 0.5, 8192).run(
        "001", heart, motion, CHUNKS[0], 30.5,
    )
    source = audit._source()
    source.load_raw.side_effect = [heart, motion]
    monkeypatch.setattr(raw_event_audit.RecordedEventReplay, "run", Mock(return_value=(
        result, status | {"input_events": status["input_events"] + 1},
    )))

    with pytest.raises(ValueError, match="event|conservation"):
        audit._participant("001", source, historical)


@pytest.mark.parametrize(("partitions", "reserved"), [
    (["001", "003"], ["002"]), (PEOPLE, ["004"]),
])
def test_run_rejects_membership_drift_before_reading_raw_observations(
    audit: RawEventAudit, partitions: list[str], reserved: list[str],
) -> None:
    """Compare current development and reserved membership with the frozen cohort.

    Args:
        audit (RawEventAudit): Audit configured with a frozen synthetic cohort.
        partitions (list[str]): Current source's development participants.
        reserved (list[str]): Current source's reserved test participants.
    """
    source = audit._source()
    source.partitions = dict.fromkeys(partitions, "development")
    source.reserved = reserved

    with pytest.raises(ValueError, match="membership differs from the frozen replay cohort"):
        audit.run()
    source.load_raw.assert_not_called()
    assert not audit.output_dir.exists()


@requires_exclusive_rename
def test_run_replays_only_development_sessions_and_publishes_verified_report(
    audit: RawEventAudit, report: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run both chunk sizes against the real synthetic preparation and a mocked model replay.

    Args:
        audit (RawEventAudit): Configured raw audit with synthetic source access.
        report (dict[str, Any]): Passing fold records to return from the frozen replay mock.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing model prediction.
    """
    heart_times = 0.5 + np.arange(0, 61, 10)
    motion_times = 0.5 + np.arange(0, 61, 0.05)
    heart = pd.DataFrame({"time": heart_times, "heart_rate": 60 + heart_times / 10})
    motion = pd.DataFrame({"time": motion_times, "x": motion_times / 10, "y": 0.0, "z": 1.0})
    source = audit._source()
    source.load_raw.side_effect = [heart, motion] * len(PEOPLE)
    batch = audit._batch(heart, motion, RULE)
    historical = pd.concat([batch.assign(participant_id=person, target=0) for person in PEOPLE])
    audit.replay.audit.output_dir.mkdir()
    historical.to_csv(audit.replay.audit.output_dir / "rows.csv", index=False)
    predict = Mock(return_value=report["prediction_parity"])
    monkeypatch.setattr(audit, "_predictions", predict)

    assert audit.run() == audit.output_dir
    result = audit.read_report()
    assert result["status"] == "passed"
    assert result["raw_input_hashes"] == source.input_hashes
    assert [(row["participant_id"], row["chunk_size"]) for row in result["participants"]] == [
        (person, chunk) for person in PEOPLE for chunk in CHUNKS
    ]
    assert all(row["epochs"] == row["eligible_epochs"] == 2 for row in result["participants"])
    assert all(row["event_conservation_matches"] for row in result["participants"])
    assert all(row["historical_feature_mismatches"] == 0 for row in result["participants"])
    assert source.load_raw.call_args_list == [
        call(person, sensor) for person in PEOPLE for sensor in ("heart_rate", "motion")
    ]
    predict.assert_called_once()
    candidate = predict.call_args.args[0]
    assert "target" not in candidate
    assert candidate.participant_id.tolist() == ["001", "001", "002", "002"]


@requires_exclusive_rename
def test_publication_is_exclusive_and_report_checksum_is_verified(
    audit: RawEventAudit, report: dict[str, Any],
) -> None:
    """Reject overwrites and changed report bytes after publication.

    Args:
        audit (RawEventAudit): Configured audit with a fresh output path.
        report (dict[str, Any]): Passing synthetic report to publish.
    """
    audit._publish(report)
    assert audit.read_report() == report
    with pytest.raises(FileExistsError):
        audit._publish(report)
    with pytest.raises(FileExistsError):
        audit.validate_run_start()
    path = audit.output_dir / "report.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        audit.read_report()


@requires_exclusive_rename
def test_read_report_rejects_drifted_configuration(
    audit: RawEventAudit, report: dict[str, Any],
) -> None:
    """Check current configuration identity when reading cached audit evidence.

    Args:
        audit (RawEventAudit): Configured raw audit with a writable temporary config.
        report (dict[str, Any]): Passing synthetic report to publish.
    """
    audit._publish(report)
    audit.config_path.write_text(audit.config_path.read_text() + "\n")
    with pytest.raises(ValueError, match="configuration changed"):
        audit.read_report()
