import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_paired_architecture_report import _write_motion_feature_pair

from sleep_classifier.paired_seed_report import PairedSeedReport


class MotionSeedFixture:
    """Create completed synthetic motion pairs without sensor reads or optimization."""

    def __init__(self, root: Path) -> None:
        """Create one paired fixture for each declared training seed.

        Args:
            root (Path): Isolated study and grouped configuration directory.
        """
        self.root = root
        self.summaries: dict[tuple[int, str], Path] = {}
        self.pairs = [self._write_case(seed) for seed in (42, 43, 44)]
        self.config_path = root / "study.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "comparison": "motion_features", "pairs": self.pairs, "output_dir": "report",
        }))


    def _write_case(self, seed: int) -> dict[str, Any]:
        """Write matched saved artifacts and grouped configuration identities.

        Args:
            seed (int): Training seed recorded by both recipes.
        """
        directory = self.root / f"seed-{seed}"
        directory.mkdir()
        paths = _write_motion_feature_pair(directory)
        pair: dict[str, Any] = {"seed": seed, "reuse_reference": True,
                                "reuse_candidate": seed == 42}
        for role, path in zip(("reference", "candidate"), paths, strict=True):
            self.summaries[seed, role] = path
            saved = json.loads(path.read_text())
            saved["plan"]["training_seed"] = seed
            for result in saved["results"]:
                metrics_path = Path(result["metrics_path"])
                metrics = json.loads(metrics_path.read_text())
                metrics["config"].update(seed=seed, device="mps")
                metrics["device"] = "mps"
                metrics_path.write_text(json.dumps(metrics))
                result["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            path.write_text(json.dumps(saved))
            self._write_configuration(seed, role, path, pair)
        return pair


    def _write_configuration(
        self, seed: int, role: str, summary: Path, pair: dict[str, Any],
    ) -> None:
        """Bind the case to an explicit model seed and grouped output path.

        Args:
            seed (int): Configured training seed.
            role (str): Reference or candidate identifier.
            summary (Path): Completed synthetic grouped summary.
            pair (dict[str, Any]): Study case receiving the grouped configuration path.
        """
        model = self.root / f"{seed}-{role}-model.yaml"
        model.write_text(yaml.safe_dump({"seed": seed, "epochs": 2}))
        grouped = self.root / f"{seed}-{role}-grouped.yaml"
        grouped.write_text(yaml.safe_dump({
            "base_model_config": model.name, "output_dir": str(summary.parent), "folds": 2,
        }))
        pair[f"{role}_config"] = grouped.name


    def update_metric_fields(
        self, seed: int, role: str, scope: str | None, changes: dict[str, Any],
    ) -> None:
        """Modify all metrics in one synthetic run and refresh their registered hashes.

        Args:
            seed (int): Study training seed to modify.
            role (str): Reference or candidate identifier.
            scope (str | None): Nested mapping name, or None for top-level metric fields.
            changes (dict[str, Any]): Replacement fields at the selected level.
        """
        summary_path = self.summaries[seed, role]
        saved = json.loads(summary_path.read_text())
        for result in saved["results"]:
            path = Path(result["metrics_path"])
            metrics = json.loads(path.read_text())
            selected = metrics if scope is None else metrics[scope]
            selected.update(changes)
            path.write_text(json.dumps(metrics))
            result["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(saved))


def test_motion_seed_plan_reuses_controls_and_adds_only_ten_fits(tmp_path: Path) -> None:
    """Count only the two new five-fold candidate recipes.

    Args:
        tmp_path (Path): Synthetic seed study directory.
    """
    fixture = MotionSeedFixture(tmp_path)
    for pair in fixture.pairs:
        for role in ("reference", "candidate"):
            path = tmp_path / pair[f"{role}_config"]
            grouped = yaml.safe_load(path.read_text())
            grouped["folds"] = 5
            path.write_text(yaml.safe_dump(grouped))
            model = tmp_path / grouped["base_model_config"]
            values = yaml.safe_load(model.read_text())
            values["epochs"] = 40
            model.write_text(yaml.safe_dump(values))
    study = PairedSeedReport(fixture.config_path)

    assert study.plan()["new_fits"] == 10
    assert study.plan()["maximum_new_epochs"] == 400
    assert [(run["seed"], run["recipe"]) for run in study.new_runs()] == [
        (43, "candidate"), (44, "candidate"),
    ]


def test_motion_reused_candidate_validates_through_its_reference(tmp_path: Path) -> None:
    """Accept the saved seven-channel candidate through the strict paired policy.

    Args:
        tmp_path (Path): Synthetic completed motion study.
    """
    fixture = MotionSeedFixture(tmp_path)
    study = PairedSeedReport(fixture.config_path)
    study.validate_reused_results()
    fixture.update_metric_fields(42, "candidate", "config", {"learning_rate": 0.002})

    with pytest.raises(ValueError, match="base recipe"):
        study.validate_reused_results()


def test_motion_seed_report_validates_complete_pairs_and_allows_fit_state_variation(
    tmp_path: Path,
) -> None:
    """Summarize complete pairs without comparing fold-specific imputation means as sources.

    Args:
        tmp_path (Path): Synthetic completed motion study.
    """
    fixture = MotionSeedFixture(tmp_path)
    fixture.update_metric_fields(43, "candidate", "motion_features", {"imputation_mean": 0.02})
    report = PairedSeedReport(fixture.config_path).build()

    assert report["comparison"] == "motion_features"
    assert [case["seed"] for case in report["seeds"]] == [42, 43, 44]
    assert report["independent_outer_test"] is False
    assert all(case["report"]["device_comparison"]["status"] == "matched"
               for case in report["seeds"])


@pytest.mark.parametrize("mutation", ["source", "device", "recipe"])
def test_motion_seed_report_rejects_cross_seed_changes(tmp_path: Path, mutation: str) -> None:
    """Reject valid individual pairs that change an undeclared cross-seed control.

    Args:
        tmp_path (Path): Synthetic completed motion study.
        mutation (str): Source, execution device, or base-recipe change.
    """
    fixture = MotionSeedFixture(tmp_path)
    if mutation == "source":
        fixture.update_metric_fields(43, "candidate", "motion_features",
                                     {"snapshot_manifest_sha256": "c" * 64})
    else:
        for role in ("reference", "candidate"):
            if mutation == "device":
                fixture.update_metric_fields(43, role, "config", {"device": "cpu"})
                fixture.update_metric_fields(43, role, None, {"device": "cpu"})
            else:
                fixture.update_metric_fields(43, role, "config", {"learning_rate": 0.002})

    with pytest.raises(ValueError, match="across seeds"):
        PairedSeedReport(fixture.config_path).build()


def test_motion_reuse_rejects_device_mismatch_without_exception(tmp_path: Path) -> None:
    """Keep the reused candidate validation device-strict.

    Args:
        tmp_path (Path): Synthetic completed motion study.
    """
    fixture = MotionSeedFixture(tmp_path)
    fixture.update_metric_fields(42, "candidate", "config", {"device": "cpu"})
    fixture.update_metric_fields(42, "candidate", None, {"device": "cpu"})

    with pytest.raises(ValueError, match="device"):
        PairedSeedReport(fixture.config_path).validate_reused_results()
