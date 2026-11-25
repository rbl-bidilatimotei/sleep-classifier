import json
import sys
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sleep_classifier import sensor_only_replay  # noqa: E402
from sleep_classifier.data import sha256  # noqa: E402
from sleep_classifier.sensor_only_replay import SensorOnlyReplayStudy  # noqa: E402
from sleep_classifier.sensor_replay_report import SensorReplayReport  # noqa: E402

FOLDS = [
    {"name": "fold-01", "train": ["002"], "validation": ["001"]},
    {"name": "fold-02", "train": ["001"], "validation": ["002"]},
]
DEVELOPMENT = ["001", "002"]
RESERVED = ["003"]
CHECKPOINT_SHA256 = "a" * 64
TOLERANCE = 2e-6
requires_exclusive_rename = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="Requires an exclusive directory rename",
)


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SensorOnlyReplayStudy:
    """Bind a real study to temporary source artifacts and a mocked frozen backend.

    Args:
        tmp_path (Path): Temporary root for audit metadata and result snapshots.
        monkeypatch (pytest.MonkeyPatch): Fixture for binding synthetic sources and the backend.
    """
    root = tmp_path.resolve()
    configs = root / "configs"
    configs.mkdir()
    config_path = configs / "replay.yaml"
    config_path.write_text(yaml.safe_dump({
        "sensor_audit_config": "sensor.yaml",
        "chronological_config": "chronological.yaml",
        "output_dir": "../results",
        "device": "mps",
    }))
    audit_config = configs / "sensor.yaml"
    audit_config.write_text("{}\n")
    audit_dir = root / "sensor"
    audit_dir.mkdir()
    audit_rows = pd.DataFrame({"participant_id": DEVELOPMENT})
    audit_rows.to_csv(audit_dir / "rows.csv", index=False)
    audit_summary = {"status": "passed", "plan": {
        "development_participants": DEVELOPMENT,
        "reserved_test_participants": RESERVED,
        "training_performed": False,
        "test_observations_read": False,
        "raw_labels_read": False,
        "model_loaded": False,
    }}
    (audit_dir / "summary.json").write_text(json.dumps(audit_summary))
    (audit_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "config_sha256": sha256(audit_config),
        "outputs": {name: sha256(audit_dir / name) for name in ("rows.csv", "summary.json")},
    }))
    audit = Mock(
        config_path=audit_config, output_dir=audit_dir, raw_data_dir=root / "raw",
        prepared_dir=root / "prepared",
    )
    audit.load_report.return_value = (audit_rows, audit_summary)

    chronological_dir = root / "chronological"
    chronological_dir.mkdir()
    chronological_report = {
        "status": "passed",
        "training_performed": False,
        "test_observations_read": False,
        "plan": {
            "folds": [fold["name"] for fold in FOLDS],
            "sources": {
                fold["name"]: {"model_checkpoint.pt": CHECKPOINT_SHA256} for fold in FOLDS
            },
            "device": "mps",
        },
    }
    (chronological_dir / "report.json").write_text(json.dumps(chronological_report))
    chronological = Mock(
        config_path=configs / "chronological.yaml", output_dir=chronological_dir,
        settings={"device": "mps", "absolute_tolerance": TOLERANCE},
    )
    chronological.read_report.return_value = chronological_report
    chronological.reference.plan.return_value = {
        "folds": FOLDS, "development": DEVELOPMENT, "reserved_test": RESERVED,
    }
    export = Mock()
    export.protected_roots.return_value = (chronological_dir, root / "frozen")
    monkeypatch.setattr(sensor_only_replay, "LabelFreeSensorAudit", Mock(return_value=audit))
    monkeypatch.setattr(
        sensor_only_replay, "ChronologicalReplayStudy", Mock(return_value=chronological),
    )
    monkeypatch.setattr(sensor_only_replay, "EvaluationExportSource", Mock(return_value=export))
    monkeypatch.setattr(sensor_only_replay.StreamingEpochPredictor, "_resolve_device", Mock())
    return SensorOnlyReplayStudy(config_path)


def _fold_result(
    fold: dict[str, Any], audit_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create one scored prediction per validation participant without loading a model.

    Args:
        fold (dict[str, Any]): Frozen fold name and membership.
        audit_rows (pd.DataFrame): All cached development rows supplied to the fold.
    """
    held_out = audit_rows.participant_id.isin(fold["validation"])
    participants = audit_rows.loc[held_out, "participant_id"]
    candidate = participants.to_frame().assign(
        time=0.0, prediction=0, probability_0=1.0, probability_1=0.0,
        probability_2=0.0, probability_3=0.0, history_epochs=1,
    )
    reference = candidate.assign(target=0)
    return candidate, reference, {
        "fold": fold["name"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "control": {
            "status": "passed", "epochs": len(reference), "maximum_probability_difference": 0.0,
            "class_disagreements": 0, "history_lengths_match": True, "reset_reasons_match": True,
        },
        "integrity": {
            "weights_unchanged": True, "evaluation_mode": True, "gradients_created": False,
            "training_performed": False, "saved_transforms_unchanged": True,
            "state_sha256": "b" * 64, "preprocessing_sha256": "c" * 64,
        },
        "comparison": SensorReplayReport.build(candidate, reference),
    }


def test_pending_and_partial_evidence_do_not_open_sources(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose missing and incomplete snapshots without verifying inputs or loading models.

    Args:
        study (SensorOnlyReplayStudy): Configured study without a published snapshot.
        monkeypatch (pytest.MonkeyPatch): Fixture for blocking source verification.
    """
    plan = Mock(side_effect=AssertionError("Unexpected plan access"))
    monkeypatch.setattr(study, "plan", plan)

    assert study.read_report() == {"status": "pending", "folds": []}
    study.output_dir.mkdir()
    (study.output_dir / "report.json").write_text("{}\n")
    assert study.read_report() == {"status": "partial", "folds": []}
    plan.assert_not_called()


@requires_exclusive_rename
def test_run_uses_every_frozen_fold_and_cached_read_checks_the_plan_without_inference(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish both frozen folds with disjoint held-out identities and separate targets.

    Args:
        study (SensorOnlyReplayStudy): Study with an unpublished two-fold reference.
        monkeypatch (pytest.MonkeyPatch): Fixture for tracing fold runs and blocking reinference.
    """
    run_fold = Mock(side_effect=_fold_result)
    monkeypatch.setattr(study, "_run_fold", run_fold)

    assert study.run() == study.output_dir
    assert [call.args[0] for call in run_fold.call_args_list] == FOLDS
    candidate = pd.read_csv(study.output_dir / "candidate-predictions.csv",
                            dtype={"participant_id": str})
    reference = pd.read_csv(study.output_dir / "reference-predictions.csv",
                            dtype={"participant_id": str})
    assert candidate.participant_id.tolist() == DEVELOPMENT
    assert candidate.fold.tolist() == [fold["name"] for fold in FOLDS]
    assert "target" not in candidate
    assert reference.target.tolist() == [0, 0]
    manifest = json.loads((study.output_dir / "manifest.json").read_text())
    module_root = Path(sensor_only_replay.__file__).resolve().parent
    implementation = manifest["implementation_hashes"]
    assert {
        "sensor_only_replay.py", "sensor_replay_inputs.py", "sensor_replay_inference.py",
        "sensor_replay_report.py", "streaming_inference.py", "model_factory.py",
    } <= implementation.keys()
    assert implementation == {name: sha256(module_root / name) for name in implementation}

    forbidden = Mock(side_effect=AssertionError("Cached read started inference"))
    monkeypatch.setattr(study, "_run_fold", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    plan = Mock(wraps=study.plan)
    monkeypatch.setattr(study, "plan", plan)
    report = study.read_report()

    plan.assert_called_once_with()
    forbidden.assert_not_called()
    assert [fold["fold"] for fold in report["folds"]] == [fold["name"] for fold in FOLDS]
    assert report["status"] == "passed"
    assert report["training_performed"] is False
    assert report["test_observations_read"] is False
    assert report["pooled"]["scopes"]["shared_scored_epochs"] == len(DEVELOPMENT)

    chronological_path = study.chronological.output_dir / "report.json"
    chronological_path.write_bytes(chronological_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source identities changed"):
        study.read_report()


@requires_exclusive_rename
@pytest.mark.parametrize("filename", ["candidate-predictions.csv", "reference-predictions.csv",
                                     "report.json"])
def test_cached_read_rejects_modified_snapshot_bytes(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch, filename: str,
) -> None:
    """Detect an edited publication before any source or model work.

    Args:
        study (SensorOnlyReplayStudy): Study with an unpublished two-fold reference.
        monkeypatch (pytest.MonkeyPatch): Fixture for substituting fold inference.
        filename (str): Published file to edit after successful publication.
    """
    monkeypatch.setattr(study, "_run_fold", _fold_result)
    study.run()
    output = study.output_dir / filename
    output.write_bytes(output.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="checksum"):
        study.read_report()


@requires_exclusive_rename
@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "failed"),
        (("training_performed",), True),
        (("test_observations_read",), True),
        (("folds",), []),
        (("folds", 1, "fold"), "fold-01"),
        (("folds", 0, "checkpoint_sha256"), "f" * 64),
        (("folds", 0, "control", "status"), "failed"),
        (("folds", 0, "control", "maximum_probability_difference"), float("nan")),
        (("folds", 0, "control", "maximum_probability_difference"), 3e-6),
        (("folds", 0, "control", "class_disagreements"), 1),
        (("folds", 0, "control", "history_lengths_match"), False),
        (("folds", 0, "control", "reset_reasons_match"), False),
        (("folds", 0, "integrity", "weights_unchanged"), False),
        (("folds", 0, "integrity", "evaluation_mode"), False),
        (("folds", 0, "integrity", "saved_transforms_unchanged"), False),
        (("folds", 0, "integrity", "gradients_created"), True),
        (("folds", 0, "integrity", "training_performed"), True),
        (("folds", 0, "comparison", "scopes", "reference_epochs"), 2),
        (("pooled", "scopes", "candidate_epochs"), 100),
    ],
)
def test_cached_read_rejects_inconsistent_report_with_matching_checksum(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch,
    path: tuple[int | str, ...], replacement: object,
) -> None:
    """Verify fold outcomes and report scope even when file checksums match.

    Args:
        study (SensorOnlyReplayStudy): Study configured with a valid frozen plan.
        monkeypatch (pytest.MonkeyPatch): Fixture for substituting fold inference.
        path (tuple[int | str, ...]): Report keys or indices that select the field to modify.
        replacement (object): Replacement evidence that violates the plan or report contract.
    """
    monkeypatch.setattr(study, "_run_fold", _fold_result)
    study.run()
    report_path = study.output_dir / "report.json"
    manifest_path = study.output_dir / "manifest.json"
    report = json.loads(report_path.read_text())
    node = report
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement
    report_path.write_text(json.dumps(report))
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["report.json"] = sha256(report_path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        study.read_report()


def _failed_control_result(
    fold: dict[str, Any], audit_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return paired predictions with evidence of legacy history drift.

    Args:
        fold (dict[str, Any]): Frozen fold name and membership.
        audit_rows (pd.DataFrame): All cached development rows supplied to the fold.
    """
    candidate, reference, result = _fold_result(fold, audit_rows)
    result["control"]["history_lengths_match"] = False
    return candidate, reference, result


def test_run_rejects_failed_control_before_publication(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Withhold the snapshot if a legacy control violates parity.

    Args:
        study (SensorOnlyReplayStudy): Study with valid sources and an unpublished result.
        monkeypatch (pytest.MonkeyPatch): Fixture for injecting a failed legacy control.
    """
    monkeypatch.setattr(study, "_run_fold", _failed_control_result)

    with pytest.raises(ValueError, match="control.*parity"):
        study.run()
    assert not study.output_dir.exists()


@requires_exclusive_rename
def test_publication_refuses_an_existing_destination(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an existing snapshot or reservation intact when publication loses a race.

    Args:
        study (SensorOnlyReplayStudy): Configured study with an unpublished destination.
        monkeypatch (pytest.MonkeyPatch): Fixture for substituting fold inference.
    """
    monkeypatch.setattr(study, "_run_fold", _fold_result)
    study.run()
    original = {file.name: file.read_bytes() for file in study.output_dir.iterdir()}
    with pytest.raises(FileExistsError, match="exists"):
        study.run()
    with pytest.raises(FileExistsError):
        study._publish(pd.DataFrame(), pd.DataFrame(), {"status": "replacement"})

    assert {file.name: file.read_bytes() for file in study.output_dir.iterdir()} == original
    assert list(study.output_dir.parent.glob(".sensor-replay-*")) == []


def _replay_fold_and_change_source(
    path: Path, fold: dict[str, Any], audit_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Complete a synthetic fold and edit one source before the publication check.

    Args:
        path (Path): Config or control report to change.
        fold (dict[str, Any]): Frozen fold name and membership.
        audit_rows (pd.DataFrame): All cached development rows supplied to the fold.
    """
    result = _fold_result(fold, audit_rows)
    path.write_bytes(path.read_bytes() + b"\n")
    return result


@pytest.mark.parametrize("source", ["config", "control"])
def test_replay_rejects_drift_before_publication(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    """Prevent publication when the plan changes after fold inference starts.

    Args:
        study (SensorOnlyReplayStudy): Study with an unpublished two-fold reference.
        monkeypatch (pytest.MonkeyPatch): Fixture for injecting drift during fold inference.
        source (str): Whether to edit the replay configuration or chronological result.
    """
    path = study.config_path if source == "config" else (
        study.chronological.output_dir / "report.json"
    )
    monkeypatch.setattr(study, "_run_fold", partial(_replay_fold_and_change_source, path))

    with pytest.raises(ValueError, match="configuration changed|sources or settings changed"):
        study.run()
    assert not study.output_dir.exists()


@pytest.mark.parametrize("participants", [DEVELOPMENT + RESERVED, DEVELOPMENT[:1]])
def test_run_rejects_unknown_or_missing_development_rows(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch, participants: list[str],
) -> None:
    """Reject reserved-test rows and incomplete development evidence before inference.

    Args:
        study (SensorOnlyReplayStudy): Study with frozen development membership.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing cached row loading.
        participants (list[str]): Incomplete or contaminated cached row membership.
    """
    loader = Mock(return_value=(pd.DataFrame({"participant_id": participants}), {}))
    run_fold = Mock(side_effect=AssertionError("Unexpected inference"))
    monkeypatch.setattr(study.audit, "load_report", loader)
    monkeypatch.setattr(study, "_run_fold", run_fold)

    with pytest.raises(ValueError, match="exactly the frozen development cohort"):
        study.run()
    run_fold.assert_not_called()
    assert not study.output_dir.exists()


def test_plan_rejects_tampered_sensor_snapshot(study: SensorOnlyReplayStudy) -> None:
    """Verify cached sensor summaries before evidence can enter a replay plan.

    Args:
        study (SensorOnlyReplayStudy): Study with a previously valid sensor snapshot.
    """
    rows = study.audit.output_dir / "rows.csv"
    rows.write_bytes(rows.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Sensor preparation snapshot checksum"):
        study.plan()


def test_plan_checks_frozen_sources_without_loading_observations(
    study: SensorOnlyReplayStudy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cached control and sensor evidence before any inference work.

    Args:
        study (SensorOnlyReplayStudy): Study with a complete two-fold control.
        monkeypatch (pytest.MonkeyPatch): Fixture for blocking prediction and observation reads.
    """
    forbidden = Mock(side_effect=AssertionError("Planning read observations or resolved a device"))
    monkeypatch.setattr(study.audit, "load_report", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(sensor_only_replay.StreamingEpochPredictor, "_resolve_device", forbidden)

    plan = study.plan()

    forbidden.assert_not_called()
    study.chronological.read_report.assert_called_once_with()
    study.chronological.reference.plan.assert_called_once_with()
    assert plan["folds"] == FOLDS
    assert plan["device"] == "mps"
    assert plan["sensor_snapshot"]["rows.csv"] == sha256(study.audit.output_dir / "rows.csv")


def test_constructor_rejects_backend_switch(study: SensorOnlyReplayStudy) -> None:
    """Require the replay to retain the chronological control backend.

    Args:
        study (SensorOnlyReplayStudy): Study constructed with the frozen control backend.
    """
    config = yaml.safe_load(study.config_path.read_text())
    config["device"] = "cpu"
    study.config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="frozen chronological backend"):
        SensorOnlyReplayStudy(study.config_path)
