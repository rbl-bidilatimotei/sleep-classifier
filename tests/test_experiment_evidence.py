import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from sleep_classifier.epoch_diagnostics import (  # noqa: E402
    EpochDiagnostics,
    EpochDiagnosticsSettings,
)
from sleep_classifier.experiment_evidence import EpochEvidenceReport  # noqa: E402


def _write_evidence(root: Path) -> tuple[Path, list[dict[str, object]]]:
    """Record two synthetic epochs with offsetting participant loss changes.

    Args:
        root (Path): Temporary grouped output directory.
    """
    root.mkdir(exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"plan": {"folds": [
        {"name": "fold-01"}, {"name": "fold-04"},
    ]}}))
    frame = pd.DataFrame({"participant_id": ["p1", "p1", "p2", "p2"],
                          "time": [0, 30, 0, 30], "target": [0, 1, 0, 1]})
    model = torch.nn.Linear(1, 4)
    settings = EpochDiagnosticsSettings({"enabled": True, "checkpoint_epochs": [13]})
    recorder = EpochDiagnostics(settings, root / "fold-01/results", {}, 1.0)
    records: list[dict[str, object]] = []
    for epoch, correct in ((13, [0.8, 0.7, 0.6, 0.9]), (14, [0.4, 0.9, 0.75, 0.95])):
        probabilities = np.zeros((4, 4))
        for index, probability in enumerate(correct):
            probabilities[index] = (1 - probability) / 3
            probabilities[index, frame.target.iloc[index]] = probability
        recorder.begin_epoch()
        recorder.record_gradient(0.5)
        recorder.record_gradient(1.5)
        path = recorder.record_epoch(
            epoch, frame, probabilities, 0.7, [0.001], model,
            float(-np.log(correct).mean()), 1.2, 0.5,
        )
        records.append(json.loads(path.read_text()))
    return root, records


def _replace_record(root: Path, record: dict[str, object]) -> None:
    """Replace one synthetic epoch record after an intentional metadata mutation.

    Args:
        root (Path): Temporary grouped output directory.
        record (dict[str, object]): Updated decoded epoch evidence.
    """
    path = root / "fold-01/results/diagnostics/epochs" / f"epoch-{record['epoch']:03d}.json"
    path.write_text(json.dumps(record))


def test_evidence_report_handles_pending_partial_and_failed_runs(tmp_path: Path) -> None:
    """Keep recorded epochs available without fabricating missing fold observations.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    root = tmp_path / "pending"
    report = EpochEvidenceReport(root)
    assert report.build() == {"schema_version": 1, "status": "not_prepared", "folds": []}
    assert not root.exists()
    with pytest.raises(ValueError, match="No completed epoch"):
        report.plot_curves()

    _write_evidence(root)
    result = report.build()
    assert result["status"] == "partial"
    assert [(fold["fold"], fold["status"]) for fold in result["folds"]] == [
        ("fold-01", "partial"), ("fold-04", "prepared"),
    ]
    assert result["folds"][1]["epochs"] == []
    assert result["folds"][1]["best_epoch"] is None
    first = result["folds"][0]["epochs"][0]
    assert first["training_evaluation_nll"] == 0.7
    assert first["training_stochastic_nll"] == 1.2
    assert first["learning_rates"] == [0.001]
    assert first["gradient_norm_pre_clip_mean"] == 1.0
    assert first["clipped_step_fraction"] == 0.5
    assert first["validation_classes"]["0"]["count"] == 2
    assert first["validation_classes"]["2"]["count"] == 0
    assert first["validation_participants"]["p1"]["count"] == 2
    assert first["mean_confidence_wrong"] is None
    assert first["high_confidence_error_fraction"] == 0.0
    assert first["validation_criterion_nll"] == pytest.approx(first["validation_nll"])

    (root / "summary.json").write_text(json.dumps({"status": "failed", "failed_fold": "fold-01"}))
    result = report.build()
    assert result["status"] == result["folds"][0]["status"] == "failed"
    assert len(result["folds"][0]["epochs"]) == 2


def test_evidence_comparison_decomposes_signed_nll_changes(tmp_path: Path) -> None:
    """Match targets and recover the total loss change from either disjoint grouping.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    root, records = _write_evidence(tmp_path)
    result = EpochEvidenceReport(root).compare_epochs("fold-01", 13, 14)
    expected = records[1]["validation"]["nll"] - records[0]["validation"]["nll"]

    assert result["count"] == 4
    assert result["nll_difference"] == pytest.approx(expected)
    for key in ("participants", "classes"):
        assert sum(group["contribution_delta"] for group in result[key]) == pytest.approx(expected)
        assert any(group["contribution_delta"] > 0 for group in result[key])
        assert any(group["contribution_delta"] < 0 for group in result[key])
    assert len(result["largest_targets"]) == 4
    assert result["largest_targets"][0]["participant_id"] == "p1"
    assert result["largest_targets"][0]["time"] == 0
    assert result["largest_targets"][0]["nll_difference"] == pytest.approx(np.log(2))


def test_evidence_plot_uses_separate_units_and_never_loads_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render measured curves while treating intermediate checkpoint bytes as opaque evidence.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
        monkeypatch (pytest.MonkeyPatch): Reject unintended checkpoint deserialization.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root, _ = _write_evidence(tmp_path)
    monkeypatch.setattr(torch, "load", Mock(side_effect=AssertionError("No model loading")))
    report = EpochEvidenceReport(root)
    figure = report.plot_curves()
    try:
        assert len(figure.axes) == 4
        assert len(figure.axes[0].lines) == 3
        assert "gradient" in figure.axes[2].get_title()
        assert "LR" in figure.axes[3].get_title()
        assert not (root / "evidence-summary.json").exists()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("mutation", ["hash", "escape", "absolute", "version", "count", "epoch"])
def test_evidence_report_rejects_corrupt_or_unsafe_artifacts(
    tmp_path: Path, mutation: str,
) -> None:
    """Reject damaged evidence and inconsistent record metadata before reporting scores.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
        mutation (str): Artifact or metadata inconsistency to introduce.
    """
    root, records = _write_evidence(tmp_path)
    record = records[0]
    artifact = record["files"]["validation_predictions"]
    if mutation == "hash":
        path = root / "fold-01/results/diagnostics" / artifact["path"]
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif mutation == "escape":
        artifact["path"] = "../outside.csv.gz"
    elif mutation == "absolute":
        artifact["path"] = str(root / "outside.csv.gz")
    elif mutation == "version":
        record["schema_version"] = 2
    elif mutation == "count":
        record["validation"]["count"] = 0
    else:
        record["epoch"] = 12
        path = root / "fold-01/results/diagnostics/epochs/epoch-013.json"
        path.write_text(json.dumps(record))
    if mutation != "epoch":
        _replace_record(root, record)

    with pytest.raises(ValueError):
        EpochEvidenceReport(root).build()


def test_evidence_comparison_rejects_changed_target_population(tmp_path: Path) -> None:
    """Reject a rehashed artifact whose valid rows no longer identify the same targets.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    root, records = _write_evidence(tmp_path)
    artifact = records[1]["files"]["validation_predictions"]
    path = root / "fold-01/results/diagnostics" / artifact["path"]
    frame = pd.read_csv(path)
    frame.loc[0, "time"] = 90
    frame.to_csv(path, index=False, compression="gzip")
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _replace_record(root, records[1])

    with pytest.raises(ValueError, match="identical target populations"):
        EpochEvidenceReport(root).compare_epochs("fold-01", 13, 14)


@pytest.mark.parametrize("fold,before,after", [
    ("fold-01", 12, 14), ("fold-01", 14, 13), ("fold-99", 13, 14),
])
def test_evidence_comparison_rejects_missing_or_invalid_epochs(
    tmp_path: Path, fold: str, before: int, after: int,
) -> None:
    """Refuse unavailable epochs instead of substituting selected-checkpoint predictions.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
        fold (str): Requested fold identifier.
        before (int): Requested first epoch.
        after (int): Requested second epoch.
    """
    root, _ = _write_evidence(tmp_path)
    with pytest.raises(ValueError):
        EpochEvidenceReport(root).compare_epochs(fold, before, after)


def test_prepared_evidence_preserves_missing_scalars_and_checkpoint_selection(
    tmp_path: Path,
) -> None:
    """Represent preparation and optional measurements without inventing scalar values.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    (tmp_path / "manifest.json").write_text(json.dumps({"plan": {"folds": [{"name": "fold-01"}]}}))
    report = EpochEvidenceReport(tmp_path)
    assert report.build()["status"] == "prepared"
    assert report.build()["folds"][0]["epochs"] == []

    root, records = _write_evidence(tmp_path)
    records[0]["training"]["evaluation_nll"] = None
    records[0]["validation_criterion_nll"] = 3.0
    _replace_record(root, records[0])
    (root / "fold-01/results/metrics.json").write_text(json.dumps({"best_epoch": 13}))
    fold = report.build()["folds"][0]
    assert fold["best_epoch"] == 13
    assert fold["status"] == "complete"
    assert fold["epochs"][0]["training_evaluation_nll"] is None
    assert fold["epochs"][0]["validation_criterion_nll"] == 3.0
    assert fold["epochs"][0]["validation_nll"] != 3.0


@pytest.mark.parametrize("mutation", ["nll", "probabilities"])
def test_epoch_comparison_recomputes_losses_from_validated_probabilities(
    tmp_path: Path, mutation: str,
) -> None:
    """Reject rehashed target evidence with invalid probabilities or inconsistent losses.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
        mutation (str): Invalid numeric evidence to introduce.
    """
    root, records = _write_evidence(tmp_path)
    artifact = records[1]["files"]["validation_predictions"]
    path = root / "fold-01/results/diagnostics" / artifact["path"]
    frame = pd.read_csv(path)
    if mutation == "nll":
        frame.loc[0, "nll"] += 1.0
    else:
        frame.loc[0, "probability_0"] = 1.5
    frame.to_csv(path, index=False, compression="gzip")
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _replace_record(root, records[1])

    with pytest.raises(ValueError, match="NLL|probabilities"):
        EpochEvidenceReport(root).compare_epochs("fold-01", 13, 14)


def test_epoch_comparison_aligns_reordered_rows_by_target_keys(tmp_path: Path) -> None:
    """Compare the same targets correctly when a valid artifact uses a different row order.

    Args:
        tmp_path (Path): Temporary grouped experiment directory.
    """
    root, records = _write_evidence(tmp_path)
    report = EpochEvidenceReport(root)
    expected = report.compare_epochs("fold-01", 13, 14)
    artifact = records[1]["files"]["validation_predictions"]
    path = root / "fold-01/results/diagnostics" / artifact["path"]
    pd.read_csv(path).iloc[::-1].to_csv(path, index=False, compression="gzip")
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _replace_record(root, records[1])

    assert report.compare_epochs("fold-01", 13, 14) == expected


def test_report_cli_does_not_create_unprepared_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Leave an unprepared destination available for later experiment preparation.

    Args:
        tmp_path (Path): Temporary parent for the absent experiment directory.
        monkeypatch (pytest.MonkeyPatch): Supply read-only report command arguments.
        caplog (pytest.LogCaptureFixture): Capture the explicit pending-status message.
    """
    import logging

    from sleep_classifier.experiment_evidence import main

    output = tmp_path / "not-prepared"
    monkeypatch.setattr(
        "sys.argv", ["experiment_evidence", "--output-dir", str(output), "--report"],
    )
    with caplog.at_level(logging.INFO):
        main()

    assert not output.exists()
    assert "status=not_prepared" in caplog.text


def test_diagnostic_notebook_inspects_recorded_epochs_without_optimization(tmp_path: Path) -> None:
    """Execute the notebook's result views against complete synthetic epoch evidence.

    Args:
        tmp_path (Path): Temporary evidence directory with saved synthetic probabilities.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root, _ = _write_evidence(tmp_path)
    reporter = EpochEvidenceReport(root)
    notebook = json.loads((Path(__file__).resolve().parents[1]
                           / "notebooks/20_transformer_diagnostics.ipynb").read_text())
    namespace = {"reporter": reporter, "evidence": reporter.build(), "pd": pd,
                 "logger": Mock(), "display": Mock()}
    try:
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and any(marker in source for marker in (
                "epoch_rows = [", "confidence_rows = []", "available_epochs = ",
            )):
                exec(source, namespace)
        assert len(namespace["epoch_rows"]) == 2
        assert len(namespace["class_rows"]) == 8
        assert len(namespace["confidence_rows"]) == 2
        assert namespace["transition"]["before_epoch"] == 13
        assert namespace["transition"]["after_epoch"] == 14
        assert namespace["display"].call_count >= 6
    finally:
        plt.close("all")
