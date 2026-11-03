import json
from types import SimpleNamespace

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from sleep_classifier.experiment_report import ExperimentComparisonReport


def metric_model(supports=(4, 3, 2, 1)):
    confusion = [[supports[row] if row == col else 0 for col in range(4)] for row in range(4)]
    return {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "classes": {
            str(label): {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": supports[label]}
            for label in range(4)
        },
        "confusion_matrix": confusion,
        "coverage": {
            "scored_epochs": 12,
            "eligible_epochs": 10,
            "label_bearing_epochs": 12,
            "fraction": 10 / 12,
        },
    }


def saved_metrics(path, balanced=False):
    payload = {
        "features": ["hr_mean", "motion_std"],
        "classes": [0, 1, 2, 3],
        "config": {
            "seed": 42,
            "max_iter": 100,
            "class_weight": "balanced" if balanced else None,
            "output_dir": "weighted" if balanced else "reference",
        },
        "provenance": {"split": {"train": ["a"], "validation": ["b"], "test": ["c"]}},
        "models": {"majority": metric_model(), "logistic_regression": metric_model()},
    }
    path.write_text(json.dumps(payload))


def test_report_renders_without_model_artifacts(tmp_path):
    reference = tmp_path / "reference" / "metrics.json"
    comparison = tmp_path / "comparison" / "metrics.json"
    reference.parent.mkdir()
    comparison.parent.mkdir()
    saved_metrics(reference)
    saved_metrics(comparison, balanced=True)
    out = tmp_path / "report"

    result = ExperimentComparisonReport(reference, comparison, out).run()

    summary = json.loads(result.read_text())
    assert summary["normalized_confusion_matrices"]["Majority reference"][0] == [1, 0, 0, 0]
    assert summary["figures"]["coefficients"] is None
    assert summary["model_summary"]["Majority reference"]["balanced_accuracy"] == 1.0
    assert summary["model_summary"]["Majority reference"]["retained_epochs"] == 10
    assert len(summary["input_metrics_sha256"]["reference"]) == 64
    assert summary["configs"]["comparison"]["class_weight"] == "balanced"
    assert summary["optimizer"] == {"reference": None, "comparison": None}
    assert (out / "confusion_matrices.png").is_file()
    assert (out / "per_class_scores.png").is_file()
    readme = (out / "README.md").read_text()
    assert "confusion_matrices.png" in readme
    assert "10 / 12" in readme


def test_incompatible_inputs_fail_before_figures(tmp_path):
    reference = tmp_path / "reference.json"
    comparison = tmp_path / "comparison.json"
    saved_metrics(reference)
    saved_metrics(comparison, balanced=True)
    payload = json.loads(comparison.read_text())
    payload["features"] = ["changed"]
    comparison.write_text(json.dumps(payload))
    out = tmp_path / "report"

    with pytest.raises(ValueError, match="features differ"):
        ExperimentComparisonReport(reference, comparison, out).run()

    assert not out.exists()


def test_coefficients_are_feature_rows_with_shared_scale(tmp_path, monkeypatch):
    reference = tmp_path / "reference" / "metrics.json"
    comparison = tmp_path / "comparison" / "metrics.json"
    reference.parent.mkdir()
    comparison.parent.mkdir()
    saved_metrics(reference)
    saved_metrics(comparison, balanced=True)
    for directory in (reference.parent, comparison.parent):
        models = directory / "models"
        models.mkdir()
        (models / "logistic_regression.joblib").write_bytes(b"fixture")

    pipeline = SimpleNamespace(
        named_steps={
            "logisticregression": SimpleNamespace(coef_=np.arange(8, dtype=float).reshape(4, 2) - 3)
        }
    )
    monkeypatch.setattr("sleep_classifier.experiment_report.joblib.load", lambda _: pipeline)
    rendered = []
    original = matplotlib.axes.Axes.imshow

    def capture(self, values, *args, **kwargs):
        rendered.append((np.asarray(values).shape, kwargs.get("vmin"), kwargs.get("vmax")))
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", capture)
    report = ExperimentComparisonReport(reference, comparison, tmp_path / "report")
    report.output_dir.mkdir()

    assert report._plot_coefficients() == "standardized_coefficients.png"
    heatmaps = [item for item in rendered if item[0] == (2, 4)]
    assert len(heatmaps) == 2
    assert heatmaps[0][1:] == heatmaps[1][1:]
    assert heatmaps[0][1] == -heatmaps[0][2]


def test_provenance_source_inventory_mismatch_fails_before_figures(tmp_path):
    reference = tmp_path / "reference.json"
    comparison = tmp_path / "comparison.json"
    saved_metrics(reference)
    saved_metrics(comparison, balanced=True)
    payload = json.loads(reference.read_text())
    payload["provenance"].update({"rule": {"version": 1}, "source_inventory": {"sha": "a"}})
    reference.write_text(json.dumps(payload))
    payload = json.loads(comparison.read_text())
    payload["provenance"].update({"rule": {"version": 1}, "source_inventory": {"sha": "b"}})
    comparison.write_text(json.dumps(payload))
    out = tmp_path / "report"

    with pytest.raises(ValueError, match="source_inventory differ"):
        ExperimentComparisonReport(reference, comparison, out).run()

    assert not out.exists()


def test_prediction_alignment_rejects_different_rows_or_targets():
    reference = pd.DataFrame({"participant_id": ["p1", "p1"], "time": [0, 30], "target": [0, 1]})
    comparison = reference.iloc[::-1].reset_index(drop=True)
    ExperimentComparisonReport._validate_prediction_alignment(reference, comparison)

    comparison.loc[0, "target"] = 3
    with pytest.raises(ValueError, match="do not align exactly"):
        ExperimentComparisonReport._validate_prediction_alignment(reference, comparison)


def test_timeline_renders_discrete_lanes_with_gaps(tmp_path):
    reference_dir = tmp_path / "reference"
    comparison_dir = tmp_path / "comparison"
    reference_dir.mkdir()
    comparison_dir.mkdir()
    rows = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p1"],
            "time": [0, 30, 90],
            "target": [0, 1, 2],
            "logistic_regression_prediction": [0, 1, 3],
        }
    )
    rows.to_csv(reference_dir / "validation_predictions.csv", index=False)
    rows.to_csv(comparison_dir / "validation_predictions.csv", index=False)
    report = ExperimentComparisonReport(
        reference_dir / "metrics.json", comparison_dir / "metrics.json", tmp_path / "report"
    )
    report.output_dir.mkdir()

    assert report._plot_timelines() == "participant_timelines.png"
    assert (report.output_dir / "participant_timelines.png").is_file()
