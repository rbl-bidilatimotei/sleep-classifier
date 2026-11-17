import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")

from sleep_classifier.causal_feature_audit import CausalFeatureAudit  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402


def _configuration(root: Path) -> Path:
    """Write metadata-only audit settings with a reserved participant and no raw observations.

    Args:
        root (Path): Temporary audit project directory.
    """
    reference = root / "reference/summary.json"
    reference.parent.mkdir()
    reference.write_text(json.dumps({"plan": {
        "development": ["1", "3"], "reserved_test": ["2"],
        "folds": [{"name": "fold-01", "train": ["1"], "validation": ["3"]},
                  {"name": "fold-02", "train": ["3"], "validation": ["1"]}],
    }}))
    values = {
        "reference_summary": "reference/summary.json",
        "comparisons": {"alternative": {"summary": "alternative/summary.json",
                                         "comparison": "architecture"}},
        "raw_data_dir": "raw", "output_dir": "audit",
        "heart_history_epochs": 10, "min_heart_history_epochs": 4,
        "max_motion_pair_gap_seconds": 0.2,
    }
    path = root / "audit.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def _rows() -> pd.DataFrame:
    """Create small aligned diagnostic rows for snapshot and statistics contracts."""
    rows = pd.DataFrame({
        "participant_id": ["1", "1", "3", "3"], "time": [0.0, 30.0, 0.0, 30.0],
        "target": [2, 3, 2, 3], "segment_id": [0, 0, 0, 0],
        "prediction_reference": [1, 3, 1, 1], "persistent_to_light": [True, False, True, True],
        "reference_correct": [False, True, False, False], "fold": ["fold-02"] * 2 + ["fold-01"] * 2,
    })
    for feature in FEATURE_COLUMNS:
        rows[feature] = [1.0, 2.0, 100.0, 200.0]
    rows["heart_rate_delta_past"] = [float("nan"), 0.1, float("nan"), 2.0]
    rows["motion_vector_change_mean"] = [0.0, 0.01, 1.0, 2.0]
    return rows


def test_plan_reads_metadata_without_raw_observations(tmp_path: Path) -> None:
    """Plan scope without opening any participant sensor or prediction files.

    Args:
        tmp_path (Path): Temporary metadata-only project.
    """
    audit = CausalFeatureAudit(_configuration(tmp_path))
    result = audit.plan()
    assert result["development_count"] == 2
    assert result["reserved_test_participants"] == ["2"]
    assert not result["test_observations_read"] and not result["model_training_performed"]
    assert not (tmp_path / "raw").exists()
    assert not audit.output_dir.exists()


def test_train_stage_statistics_exclude_heldout_participants(tmp_path: Path) -> None:
    """Fit descriptive stage summaries on the declared training participants only.

    Args:
        tmp_path (Path): Temporary audit project.
    """
    audit = CausalFeatureAudit(_configuration(tmp_path))
    plan = json.loads(audit.settings.reference_summary.read_text())["plan"]
    records = audit._training_stage_summary(_rows(), plan)
    first = next(row for row in records if row["fold"] == "fold-01"
                 and row["feature"] == "heart_rate_mean" and row["target"] == 2)
    assert first["median"] == 1.0 and first["participants"] == 1
    second = next(row for row in records if row["fold"] == "fold-02"
                  and row["feature"] == "heart_rate_mean" and row["target"] == 2)
    assert second["median"] == 100.0


def test_snapshot_roundtrip_and_tampering_detection(tmp_path: Path) -> None:
    """Verify saved outputs without reopening raw observations or fitting a model.

    Args:
        tmp_path (Path): Temporary snapshot project.
    """
    audit = CausalFeatureAudit(_configuration(tmp_path))
    rows = _rows()
    summary = {**audit.plan(), "targets": len(rows), "participants": 2}
    audit._write_snapshot(rows, summary)
    actual, saved = audit.load_report()
    assert len(actual) == 4 and saved["model_training_performed"] is False
    assert set(actual.participant_id) == {"1", "3"}
    assert not (tmp_path / "raw").exists()
    (audit.output_dir / "rows.csv").write_text("altered")
    with pytest.raises(ValueError, match="checksum"):
        audit.load_report()
    with pytest.raises(FileExistsError):
        audit.run()


def test_snapshot_rejects_reserved_rows_even_with_updated_hashes(tmp_path: Path) -> None:
    """Reject reserved observations independently of output checksum verification.

    Args:
        tmp_path (Path): Temporary project with an invalid cached scope.
    """
    audit = CausalFeatureAudit(_configuration(tmp_path))
    rows = _rows()
    rows.loc[0, "participant_id"] = "2"
    audit._write_snapshot(rows, {**audit.plan(), "targets": len(rows), "participants": 3})
    with pytest.raises(ValueError, match="development-only"):
        audit.load_report()


@pytest.mark.parametrize("directory", ["raw/out", "reference/out", "."])
def test_audit_output_cannot_modify_input_directories(tmp_path: Path, directory: str) -> None:
    """Reject output paths that would contain or modify source data.

    Args:
        tmp_path (Path): Temporary audit project.
        directory (str): Unsafe output path relative to its configuration.
    """
    config = _configuration(tmp_path)
    values = yaml.safe_load(config.read_text())
    values["output_dir"] = directory
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="output"):
        CausalFeatureAudit(config)
