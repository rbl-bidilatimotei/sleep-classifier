import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_paired_architecture_report import _write_learning_rate_pair, _write_motion_feature_pair

from sleep_classifier.paired_architecture_report import PairedArchitectureReport


class MotionRateFixture:
    """Create same-input synthetic motion runs with distinct constant learning rates."""

    def __init__(self, root: Path) -> None:
        """Expand both saved roles to the same motion inputs without training.

        Args:
            root (Path): Isolated synthetic grouped-output directory.
        """
        self.paths = _write_motion_feature_pair(root)
        left = json.loads(self.paths[0].read_text())
        right = json.loads(self.paths[1].read_text())
        for original, augmented in zip(left["results"], right["results"], strict=True):
            control = json.loads(Path(original["metrics_path"]).read_text())
            treatment = json.loads(Path(augmented["metrics_path"]).read_text())
            for key in ("features", "model_features", "model_config", "parameter_count",
                        "motion_features"):
                control[key] = copy.deepcopy(treatment[key])
            control["config"]["motion_features"] = copy.deepcopy(
                treatment["config"]["motion_features"]
            )
            treatment["config"]["learning_rate"] = 0.0003
            self.save(original, control)
            self.save(augmented, treatment)
        self.paths[0].write_text(json.dumps(left))
        self.paths[1].write_text(json.dumps(right))


    @staticmethod
    def save(result: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Write synthetic metrics and refresh their registered summary hash.

        Args:
            result (dict[str, Any]): Fold summary entry receiving the checksum.
            metrics (dict[str, Any]): Replacement synthetic model metrics.
        """
        path = Path(result["metrics_path"])
        path.write_text(json.dumps(metrics))
        result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


    def mutate_candidate(self, mutation: str) -> None:
        """Change one unsupported input or training contract in a saved candidate.

        Args:
            mutation (str): Contract selected by the rejection test.
        """
        summary = json.loads(self.paths[1].read_text())
        result = summary["results"][0]
        metrics = json.loads(Path(result["metrics_path"]).read_text())
        if mutation == "device":
            metrics["device"] = "mps"
        elif mutation == "source":
            metrics["motion_features"]["snapshot_manifest_sha256"] = "c" * 64
        elif mutation == "imputation":
            metrics["motion_features"]["imputation_mean"] += 0.1
        elif mutation == "counts":
            metrics["motion_features"]["validation_missing_targets"] = 1
        elif mutation == "hr_state":
            metrics["window_features"]["scaling"]["delta_mean"] += 0.1
        elif mutation == "schema":
            metrics["features"].reverse()
        elif mutation == "effective_channels":
            metrics["model_features"].reverse()
        elif mutation == "constructor":
            metrics["model_config"]["input_size"] = 6
        elif mutation == "capacity":
            metrics["parameter_count"] += 1
        elif mutation == "recipe":
            metrics["config"]["patience"] += 1
        elif mutation == "missing_record":
            metrics.pop("motion_features")
        elif mutation == "removed_input":
            metrics["config"].pop("motion_features")
        else:
            raise ValueError("Unknown synthetic mutation")
        self.save(result, metrics)
        self.paths[1].write_text(json.dumps(summary))



def test_learning_rate_report_accepts_identical_motion_inputs(tmp_path: Path) -> None:
    """Allow constant-rate changes without changing motion or HR preprocessing.

    Args:
        tmp_path (Path): Isolated completed synthetic studies.
    """
    fixture = MotionRateFixture(tmp_path)
    report = PairedArchitectureReport(*fixture.paths, "learning_rate").build()

    assert report["comparison"] == "learning_rate"
    assert report["fixed_motion_inputs"] == {
        "definition": {"name": "fixed_lag_motion", "step_seconds": 1.0,
                       "max_sample_age_seconds": 0.2},
        "audit_config_sha256": "a" * 64, "snapshot_manifest_sha256": "b" * 64,
    }
    assert "capacity_caveat" not in report
    assert report["eligible_epochs"] == 8


@pytest.mark.parametrize("mutation", [
    "device", "source", "imputation", "counts", "hr_state", "schema", "effective_channels",
    "constructor", "capacity", "recipe", "missing_record", "removed_input",
])
def test_learning_rate_report_rejects_motion_or_control_drift(
    tmp_path: Path, mutation: str,
) -> None:
    """Keep every input and training contract fixed except the declared rate.

    Args:
        tmp_path (Path): Isolated completed synthetic studies.
        mutation (str): Undeclared input or training change.
    """
    fixture = MotionRateFixture(tmp_path)
    fixture.mutate_candidate(mutation)

    with pytest.raises(ValueError):
        PairedArchitectureReport(*fixture.paths, "learning_rate").build()



def test_learning_rate_report_rejects_fold_specific_motion_sources(tmp_path: Path) -> None:
    """Require one immutable motion source even when each pair agrees internally.

    Args:
        tmp_path (Path): Isolated synthetic study with two folds.
    """
    fixture = MotionRateFixture(tmp_path)
    for summary_path in fixture.paths:
        summary = json.loads(summary_path.read_text())
        result = summary["results"][0]
        metrics = json.loads(Path(result["metrics_path"]).read_text())
        metrics["motion_features"]["snapshot_manifest_sha256"] = "c" * 64
        fixture.save(result, metrics)
        summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="source differs across folds"):
        PairedArchitectureReport(*fixture.paths, "learning_rate").build()



def test_learning_rate_report_rejects_adding_motion(tmp_path: Path) -> None:
    """Preserve the explicit feature-addition policy instead of hiding it under rates.

    Args:
        tmp_path (Path): Isolated six-input and seven-input synthetic pair.
    """
    paths = _write_motion_feature_pair(tmp_path)
    summary = json.loads(paths[1].read_text())
    for result in summary["results"]:
        metrics = json.loads(Path(result["metrics_path"]).read_text())
        metrics["config"]["learning_rate"] = 0.0003
        MotionRateFixture.save(result, metrics)
    paths[1].write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Learning-rate model differs"):
        PairedArchitectureReport(*paths, "learning_rate").build()



def test_learning_rate_report_preserves_no_motion_behavior(tmp_path: Path) -> None:
    """Keep legacy six-input learning-rate reports free from motion metadata.

    Args:
        tmp_path (Path): Isolated original learning-rate studies.
    """
    paths = _write_learning_rate_pair(tmp_path)
    report = PairedArchitectureReport(*paths, "learning_rate").build()

    assert report["comparison"] == "learning_rate"
    assert "fixed_motion_inputs" not in report
