import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sleep_classifier.temporal_slice_report import TemporalSliceReport  # noqa: E402


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a short participant with two contiguous segments."""
    rows = pd.DataFrame({
        "participant_id": ["a"] * 7,
        "time": [0.0, 30.0, 60.0, 90.0, 120.0, 300.0, 330.0],
        "target": [0, 1, 1, 2, 3, 0, 3],
        "segment_id": [0] * 5 + [1] * 2,
    })
    reference = rows.drop(columns="segment_id").assign(prediction=rows.target)
    candidate = reference.assign(prediction=1)
    return reference, candidate, rows


def test_history_bins_reset_and_transitions_overlap() -> None:
    """Reset history at segments and count only within-segment target changes."""
    report = TemporalSliceReport(*_frames(), context_epochs=4, short_history_epochs=1).build()
    slices = {item["name"]: item for item in report["slices"]}
    assert [item["targets"] for item in report["slices"]] == [2, 3, 2, 4]
    assert slices["stage_transition"]["reference"]["classes"]["wake"]["support"] == 0
    assert slices["history_4"]["reference"]["classes"]["REM"]["support"] == 1
    assert slices["stage_transition"]["reference"]["accuracy"] == 1.0
    assert slices["stage_transition"]["candidate"]["accuracy"] == 0.25
    assert report["posthoc"] and not report["independent_outer_test"]


def test_input_order_does_not_change_alignment() -> None:
    """Sort independently before comparing epoch identities and metrics."""
    reference, candidate, rows = _frames()
    expected = TemporalSliceReport(reference, candidate, rows).build()
    actual = TemporalSliceReport(reference.iloc[::-1], candidate, rows.iloc[::-1]).build()
    assert actual == expected
    assert actual["slices"][2]["targets"] == 0
    assert actual["slices"][2]["reference"] is None
    assert actual["slices"][2]["macro_f1_difference"] is None


@pytest.mark.parametrize("mutation", ["duplicate", "target", "missing", "nonfinite", "prediction"])
def test_reject_invalid_prediction_population(mutation: str) -> None:
    """Reject predictions that cannot support a paired comparison.

    Args:
        mutation (str): Invalid prediction condition.
    """
    reference, candidate, rows = _frames()
    if mutation == "duplicate":
        candidate = pd.concat([candidate, candidate.iloc[:1]])
    elif mutation == "target":
        candidate.loc[0, "target"] = 2
    elif mutation == "missing":
        candidate = candidate.iloc[1:]
    elif mutation == "nonfinite":
        candidate.loc[0, "time"] = float("inf")
    else:
        candidate.loc[0, "prediction"] = 9

    with pytest.raises(ValueError):
        TemporalSliceReport(reference, candidate, rows)


@pytest.mark.parametrize("segment", [-1, 0.5, 0])
def test_reject_invalid_segment_or_gap(segment: float) -> None:
    """Reject invalid segment identifiers and gaps inside one segment.

    Args:
        segment (float): Invalid value or a reused segment across a gap.
    """
    reference, candidate, rows = _frames()
    rows["segment_id"] = rows.segment_id.astype(float)
    rows.loc[5:, "segment_id"] = segment
    with pytest.raises(ValueError):
        TemporalSliceReport(reference, candidate, rows)


def test_participants_cannot_create_transitions() -> None:
    """Treat each participant's first epoch as a new history."""
    rows = pd.DataFrame({
        "participant_id": ["a", "b"], "time": [0, 30], "target": [0, 3],
        "segment_id": [0, 0],
    })
    predictions = rows.drop(columns="segment_id").assign(prediction=rows.target)
    slices = TemporalSliceReport(predictions, predictions, rows).build()["slices"]
    assert slices[0]["targets"] == 2
    assert slices[-1]["targets"] == 0


@pytest.mark.parametrize("context,short", [(2, 1), (30, 0), (30, 29), (30.0, 2), (30, True)])
def test_reject_invalid_history_bounds(context: int, short: int) -> None:
    """Require integer bounds with three distinct history bins.

    Args:
        context (int): Proposed maximum context.
        short (int): Proposed upper short-history bound.
    """
    with pytest.raises(ValueError, match="History bounds"):
        TemporalSliceReport(*_frames(), context_epochs=context, short_history_epochs=short)


def test_feature_checksums_reject_changed_and_missing_files(tmp_path: Path) -> None:
    """Reject modified or absent prepared metadata before assigning slices.

    Args:
        tmp_path (Path): Temporary grouped root.
    """
    directory = tmp_path / "fold-01/features"
    directory.mkdir(parents=True)
    provenance = directory / "provenance.json"
    provenance.write_text(json.dumps({"split": {}}))
    hashes = {
        str(provenance.relative_to(tmp_path)): hashlib.sha256(provenance.read_bytes()).hexdigest()
    }
    TemporalSliceReport._verify_features(tmp_path, directory, hashes)

    provenance.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        TemporalSliceReport._verify_features(tmp_path, directory, hashes)
    provenance.unlink()
    with pytest.raises(FileNotFoundError):
        TemporalSliceReport._verify_features(tmp_path, directory, hashes)


def _reject_pair(self: object) -> dict[str, object]:
    """Simulate a failed paired integrity check before metadata loading.

    Args:
        self (object): Paired report instance supplied by the patched method.
    """
    raise ValueError("paired integrity failed")


def test_grouped_adapter_requires_paired_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run paired artifact checks before reading any temporal metadata.

    Args:
        tmp_path (Path): Temporary summary parent.
        monkeypatch (pytest.MonkeyPatch): Scoped paired-validator replacement.
    """
    monkeypatch.setattr(
        "sleep_classifier.temporal_slice_report.PairedArchitectureReport.build", _reject_pair,
    )
    with pytest.raises(ValueError, match="paired integrity failed"):
        TemporalSliceReport.from_grouped(tmp_path / "reference.json", tmp_path / "candidate.json")


def test_metadata_requires_manifest_and_matching_plan(tmp_path: Path) -> None:
    """Reject missing manifests and mismatched plans before loading feature tables.

    Args:
        tmp_path (Path): Temporary grouped root.
    """
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"plan": {"folds": []}}))
    with pytest.raises(FileNotFoundError):
        TemporalSliceReport._load_metadata(summary)

    (tmp_path / "manifest.json").write_text(json.dumps({"plan": {"folds": ["changed"]}}))
    with pytest.raises(ValueError, match="plans differ"):
        TemporalSliceReport._load_metadata(summary)


def test_feature_inventory_rejects_unrecorded_files(tmp_path: Path) -> None:
    """Reject metadata files that were added after preparation.

    Args:
        tmp_path (Path): Temporary grouped root.
    """
    directory = tmp_path / "fold-01/features"
    directory.mkdir(parents=True)
    (directory / "provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        TemporalSliceReport._verify_features(tmp_path, directory, {})


def test_reject_interleaved_segments_with_contiguous_internal_times() -> None:
    """Reject a reused segment even when its own timestamps remain thirty seconds apart."""
    rows = pd.DataFrame({
        "participant_id": ["a"] * 3,
        "time": [0, 15, 30],
        "target": [0, 1, 2],
        "segment_id": [0, 1, 0],
    })
    predictions = rows.drop(columns="segment_id").assign(prediction=rows.target)

    with pytest.raises(ValueError, match="consecutive blocks"):
        TemporalSliceReport(predictions, predictions, rows)
