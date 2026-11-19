import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn")

from sleep_classifier.audit_evidence import AuditEvidenceLoader  # noqa: E402
from sleep_classifier.audit_motion_source import VerifiedMotionSource  # noqa: E402
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


def _grid_configuration(root: Path) -> Path:
    """Enable fixed-grid motion diagnostics on the metadata-only audit fixture.

    Args:
        root (Path): Temporary project directory.
    """
    config = _configuration(root)
    values = yaml.safe_load(config.read_text())
    values["motion_grid"] = {
        "step_seconds": 1.0, "max_sample_age_seconds": 0.2,
        "support_fractions": [0.5, 0.8, 0.95],
    }
    config.write_text(yaml.safe_dump(values))
    return config


def _grid_motion_source(root: Path) -> VerifiedMotionSource:
    """Write checksum-verified stationary observations for two permitted participants.

    Args:
        root (Path): Temporary raw dataset directory.
    """
    directory = root / "motion"
    directory.mkdir(parents=True)
    records = []
    for participant in ("1", "3"):
        path = directory / f"{participant}_acceleration.txt"
        path.write_text("".join(f"{tick} 0 0 1\n" for tick in range(1, 30)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest} motion/{path.name}\n")
    (root / "SHA256SUMS.txt").write_text("".join(records))
    return VerifiedMotionSource(root, ["1", "3"], ["2"])


@pytest.mark.parametrize("fractions", [
    [], [True], [float("nan")], [float("inf")], [-0.1], [1.1], [0.8, 0.5], [0.5, 0.5], "0.5",
])
def test_grid_support_cutoffs_require_sorted_finite_fractions(
    tmp_path: Path, fractions: Any,
) -> None:
    """Reject invalid support cutoffs before reading observations.

    Args:
        tmp_path (Path): Temporary metadata-only audit directory.
        fractions (Any): Invalid YAML support threshold value.
    """
    config = _grid_configuration(tmp_path)
    values = yaml.safe_load(config.read_text())
    values["motion_grid"]["support_fractions"] = fractions
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="support_fractions"):
        CausalFeatureAudit(config)
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("grid", [None, {}, {"step_seconds": 1.0}, {
    "step_seconds": 1.0, "max_sample_age_seconds": 0.2,
    "support_fractions": [0.5], "unknown": 3,
}])
def test_grid_mapping_requires_exact_documented_keys(tmp_path: Path, grid: Any) -> None:
    """Require the complete optional schema while retaining all legacy required keys.

    Args:
        tmp_path (Path): Temporary audit project directory.
        grid (Any): Invalid optional grid configuration.
    """
    config = _grid_configuration(tmp_path)
    values = yaml.safe_load(config.read_text())
    values["motion_grid"] = grid
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="motion_grid"):
        CausalFeatureAudit(config)


def test_grid_preserves_missing_targets_and_distinguishes_stationary_zero(tmp_path: Path) -> None:
    """Read allowed raw files once and preserve unsupported targets beside genuine zero motion.

    Args:
        tmp_path (Path): Temporary metadata and publisher sensor project.
    """
    audit = CausalFeatureAudit(_grid_configuration(tmp_path))
    source = _grid_motion_source(tmp_path / "raw")
    rows = _rows().drop(columns="motion_vector_change_mean")
    with patch.object(source, "load", wraps=source.load) as loader:
        actual = audit._add_motion(rows, source)
    assert loader.call_count == 2
    assert loader.call_args_list[0].args == ("1",)
    assert loader.call_args_list[1].args == ("3",)
    assert len(actual) == len(rows)
    pd.testing.assert_frame_equal(actual[list(rows.columns)], rows)
    supported = actual.loc[actual.time == 0]
    unsupported = actual.loc[actual.time == 30]
    assert supported.motion_grid_change_mean.eq(0).all()
    assert supported.motion_grid_point_fraction.eq(1).all()
    assert supported.motion_grid_pair_fraction.eq(1).all()
    assert unsupported.motion_grid_change_mean.isna().all()
    assert unsupported.motion_grid_pair_count.eq(0).all()
    assert unsupported.motion_grid_pair_fraction.eq(0).all()
    assert len(source.accessed_files) == 2
    assert len(audit.participant_cadence_support) == 2
    with pytest.raises(ValueError, match="development"):
        source.load("2")
    sensitivity = audit._support_sensitivity(actual)
    assert all(item["total_targets"] == 4 for item in sensitivity)
    assert all(item["available_supported_targets"] == 2 for item in sensitivity)
    relabeled = actual.assign(target=0, prediction_reference=1, persistent_to_light=False)
    assert sensitivity == audit._support_sensitivity(relabeled)


def test_grid_stage_and_reference_error_summaries_keep_missing_denominators(tmp_path: Path) -> None:
    """Include Wake reference misses and retain unsupported epochs in descriptive denominators.

    Args:
        tmp_path (Path): Temporary audit metadata directory.
    """
    audit = CausalFeatureAudit(_grid_configuration(tmp_path))
    rows = _rows()
    rows["target"] = [0, 0, 2, 3]
    rows["prediction_reference"] = [1, 0, 1, 1]
    rows["persistent_to_light"] = [False, False, True, True]
    rows["motion_grid_change_mean"] = [np.nan, 0.0, 1.0, np.nan]
    errors = audit._reference_error_summary(rows)
    wake = next(item for item in errors if item["target"] == 0
                and item["feature"] == "motion_grid_change_mean")
    assert wake["reference_to_light_targets"] == 1
    assert wake["reference_to_light_available"] == 0
    assert wake["reference_correct_median"] == 0
    assert wake["median_difference"] is None
    availability = audit._stage_feature_availability(rows)
    wake_support = next(item for item in availability if item["target"] == 0
                        and item["feature"] == "motion_grid_change_mean")
    assert wake_support["total_targets"] == 2
    assert wake_support["missing_targets"] == 1
    assert wake_support["available_fraction"] == 0.5
    assert not any(item["target"] == 0 for item in audit._matched_error_summary(rows))
    json.dumps(errors + availability, allow_nan=False)


def test_grid_run_constructs_no_model_and_records_snapshot_provenance(tmp_path: Path) -> None:
    """Publish finite optional diagnostics while rejecting model construction and later tampering.

    Args:
        tmp_path (Path): Temporary synthetic audit project.
    """
    torch = pytest.importorskip("torch")
    audit = CausalFeatureAudit(_grid_configuration(tmp_path))
    _grid_motion_source(tmp_path / "raw")
    plan = json.loads(audit.settings.reference_summary.read_text())["plan"]
    evidence = {
        "rows": _rows().drop(columns="motion_vector_change_mean"),
        "participant_ids": ["1", "3"], "reserved_test_ids": ["2"],
        "plan": plan, "input_hashes": {"saved-artifact": "verified-digest"},
    }
    with patch.object(AuditEvidenceLoader, "load", return_value=evidence), patch.object(
        torch.nn.Module, "__init__", side_effect=AssertionError("Audit constructed a model"),
    ):
        audit.run()
    rows, summary = audit.load_report()
    assert len(rows) == 4 and summary["targets"] == 4
    assert summary["persistent_definition"].startswith("True Deep/REM")
    assert summary["motion_grid"]["points_per_epoch"] == 29
    assert summary["motion_grid"]["pairs_per_epoch"] == 28
    assert len(summary["raw_files_read"]) == 2
    assert summary["source_artifact_hashes"] == evidence["input_hashes"]
    assert "motion_grid_change_mean" in summary["diagnostic_columns"]
    assert len(summary["extra_charts"]) == 2
    assert len(summary["participant_cadence_support"]) == 2
    assert summary["reference_error_summary"]
    json.dumps(summary, allow_nan=False)
    manifest = json.loads((audit.output_dir / "manifest.json").read_text())
    assert "motion_grid.py" in manifest["implementation_hashes"]
    with pytest.raises(FileExistsError):
        audit._write_snapshot(rows, summary)
    (audit.output_dir / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        audit.load_report()


def test_legacy_configuration_keeps_inventory_and_cached_snapshot(tmp_path: Path) -> None:
    """Retain the historical feature list and snapshot inventory without optional grid settings.

    Args:
        tmp_path (Path): Temporary metadata-only audit project.
    """
    audit = CausalFeatureAudit(_configuration(tmp_path))
    assert audit.settings.motion_grid is None
    plan = json.loads(audit.settings.reference_summary.read_text())["plan"]
    summary = audit._summarize(_rows(), {"participant_ids": ["1", "3"], "plan": plan})
    assert "motion_grid" not in summary and "diagnostic_columns" not in summary
    assert len(summary["candidate_availability"]) == 8
    audit._write_snapshot(_rows(), summary)
    manifest = json.loads((audit.output_dir / "manifest.json").read_text())
    assert "motion_grid.py" not in manifest["implementation_hashes"]
    assert set(manifest["outputs"]) == {
        "rows.csv", "summary.json", "dashboard.html", "feature-distributions.png",
        "persistent-errors.png",
    }
    rows, cached = audit.load_report()
    assert len(rows) == 4 and "motion_grid" not in cached
