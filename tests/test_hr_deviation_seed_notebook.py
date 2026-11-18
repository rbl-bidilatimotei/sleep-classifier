import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from sleep_classifier.paired_seed_report import PairedSeedReport

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/33_hr_deviation_only_seeds.ipynb"


def _notebook() -> dict[str, Any]:
    """Read the output-free source notebook without executing operations."""
    return json.loads(NOTEBOOK.read_text())


def test_deviation_seed_study_reuses_four_runs_and_caps_new_work() -> None:
    """Require all measured controls and only ten new candidate fits."""
    study = PairedSeedReport(ROOT / "configs/hr-deviation-only-seeds.yaml")
    plan = study.plan()
    assert study.comparison == "window_feature_ablation"
    assert plan["new_fits"] == 10 and plan["maximum_new_epochs"] == 400
    assert [(run["seed"], run["recipe"]) for run in study.new_runs()] == [
        (43, "candidate"), (44, "candidate"),
    ]
    assert [case["seed"] for case in study.cases] == [42, 43, 44]
    assert all(case["reuse_reference"] for case in study.cases)
    assert [case["reuse_candidate"] for case in study.cases] == [True, False, False]
    assert [Path(case["reference_config"]).name for case in study.cases] == [
        "transformer-window-local-hr-grouped.yaml",
        "transformer-window-local-hr-seed-43-grouped.yaml",
        "transformer-window-local-hr-seed-44-grouped.yaml",
    ]
    assert [Path(case["candidate_config"]).name for case in study.cases] == [
        "hr-deviation-only-grouped.yaml", "hr-deviation-only-seed-43-grouped.yaml",
        "hr-deviation-only-seed-44-grouped.yaml",
    ]
    assert study.output_dir == ROOT / "artifacts/hr-deviation-only-seed-comparison"


@pytest.mark.parametrize("seed", [43, 44])
def test_deviation_seed_recipes_clone_the_measured_control(seed: int) -> None:
    """Change only seed and output paths while retaining every measured control.

    Args:
        seed (int): Training seed for the new deviation-only candidate.
    """
    original = yaml.safe_load((ROOT / "configs/hr-deviation-only.yaml").read_text())
    candidate = yaml.safe_load(
        (ROOT / f"configs/hr-deviation-only-seed-{seed}.yaml").read_text()
    )
    assert candidate.pop("seed") == seed
    assert original.pop("seed") == 42
    assert candidate.pop("output_dir") == (
        f"../artifacts/hr-deviation-only-seeds/seed-{seed}/single"
    )
    original.pop("output_dir")
    assert candidate == original
    assert candidate["window_features"] == {
        "name": "heart_rate_deviation", "history_epochs": 10,
        "min_history_epochs": 4, "masked_channels": ["heart_history_available"],
    }
    assert candidate["context_epochs"] == 30 and candidate["patch_size"] == 3
    assert candidate["epochs"] == 40 and candidate["patience"] == 12
    assert candidate["device"] == "auto"

    grouped_original = yaml.safe_load(
        (ROOT / "configs/hr-deviation-only-grouped.yaml").read_text()
    )
    grouped = yaml.safe_load(
        (ROOT / f"configs/hr-deviation-only-seed-{seed}-grouped.yaml").read_text()
    )
    assert grouped.pop("base_model_config") == f"hr-deviation-only-seed-{seed}.yaml"
    grouped_original.pop("base_model_config")
    assert grouped.pop("output_dir") == (
        f"../artifacts/hr-deviation-only-seeds/seed-{seed}/candidate"
    )
    grouped_original.pop("output_dir")
    assert grouped == grouped_original
    assert grouped["split_seed"] == 42 and grouped["folds"] == 5
    assert "selected_folds" not in grouped


@pytest.mark.parametrize("mutation", [None, "name", "history_epochs", "min_history_epochs"])
def test_deviation_seed_role_guard_retains_nested_feature_controls(mutation: str | None) -> None:
    """Execute the notebook guard and reject undeclared nested recipe differences.

    Args:
        mutation (str | None): Candidate feature control to change, or no mutation.
    """
    reference = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    reference.pop("seed")
    reference.pop("output_dir")
    candidate = copy.deepcopy(reference)
    candidate["window_features"]["masked_channels"] = ["heart_history_available"]
    if mutation is not None:
        candidate["window_features"][mutation] = (
            "different_transform" if mutation == "name" else 9
        )
    source = "".join(_notebook()["cells"][8]["source"])
    guard = source[source.index("reference_controls ="):source.index("\ndisplay(")]
    namespace: dict[str, Any] = {
        "canonical_recipes": {"reference": reference, "candidate": candidate},
        "study": SimpleNamespace(comparison="window_feature_ablation"),
    }
    if mutation is None:
        exec(compile(guard, str(NOTEBOOK), "exec"), namespace)
        assert candidate["window_features"]["masked_channels"] == ["heart_history_available"]
    else:
        with pytest.raises(ValueError, match="beyond the availability mask"):
            exec(compile(guard, str(NOTEBOOK), "exec"), namespace)


def test_deviation_seed_notebook_explains_and_gates_operations() -> None:
    """Keep report execution inert and require validated targets before optimization."""
    notebook = _notebook()
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        sources.append(source)
        assert not cell["outputs"] and cell["execution_count"] is None
        previous = notebook["cells"][index - 1]
        assert previous["cell_type"] == "markdown"
        explanation = "".join(previous["source"]).lower()
        assert "what" in explanation and "why" in explanation
        if any(marker in source for marker in (
            "if RUN_SETUP:", "if RUN_PREPARATION:", "if RUN_TRAINING:",
            "scripts/execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"]["tags"]
    source = "\n".join(sources)
    assert all(f"RUN_{name} = False" in source for name in ("SETUP", "PREPARATION", "TRAINING"))
    assert "RUN_CHECK = True" in source
    assert 'BRANCH = "experiment/hr-deviation-only-seeds"' in source
    assert 'study.comparison != "window_feature_ablation"' in source
    assert 'candidate_controls["window_features"].pop("masked_channels")' in source
    assert 'candidate_controls.pop("window_features")' not in source
    assert 'resolved_device != "mps"' in source
    assert 'metrics["parameter_count"] != 2852' in source
    assert 'len(metrics["model_features"]) != 8' in source
    assert "short_history_epochs=4" in source
    assert "ConfusionMatrixDisplay" in source and "EpochEvidenceReport" in source
    assert '"checkpoint_nll"' in source
    assert 'if all(row["status"] == "complete" for row in statuses):' in source
    assert source.count("--train") == 1
    for index in (10, 14):
        operation = "".join(notebook["cells"][index]["source"])
        assert 'experiment = GroupedValidationExperiment(Path(run["config"]))' in operation
        assert "grouped_config = experiment.config_path" in operation
        assert "experiment.output_dir" in operation and "masked_channels=%s" in operation
        assert '--config "{grouped_config}"' in operation
        assert 'get_ipython().user_ns["_exit_code"] != 0' in operation
    training = "".join(notebook["cells"][14]["source"])
    assert training.index("experiment.validate_training_start()") < training.index("--train")
    explanation = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "-0.005772" in explanation
    assert "does not establish" in explanation
    assert "+0.01834" not in explanation and "Measured replication findings" not in explanation
