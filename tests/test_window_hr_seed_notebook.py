import json
from pathlib import Path

import pytest
import yaml

from sleep_classifier.paired_seed_report import PairedSeedReport

ROOT = Path(__file__).resolve().parents[1]


def test_hr_seed_study_reuses_all_controls_and_only_trains_two_candidates() -> None:
    """Preserve all measured runs and cap the new work at ten candidate fits."""
    study = PairedSeedReport(ROOT / "configs/transformer-window-local-hr-seeds.yaml")
    assert study.comparison == "window_features"
    assert study.plan()["new_fits"] == 10
    assert study.plan()["maximum_new_epochs"] == 400
    assert [(run["seed"], run["recipe"]) for run in study.new_runs()] == [
        (43, "candidate"), (44, "candidate"),
    ]
    assert all(case["reuse_reference"] for case in study.cases)
    assert [case["reuse_candidate"] for case in study.cases] == [True, False, False]
    assert [Path(case["reference_config"]).name for case in study.cases] == [
        "transformer-patch-grouped.yaml", "transformer-patch-seed-43-grouped.yaml",
        "transformer-patch-seed-44-grouped.yaml",
    ]


@pytest.mark.parametrize("seed", [43, 44])
def test_hr_seed_recipes_preserve_the_measured_candidate(seed: int) -> None:
    """Change only training seed and output paths from the measured HR candidate.

    Args:
        seed (int): Candidate training seed to validate.
    """
    original = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    candidate = yaml.safe_load(
        (ROOT / f"configs/transformer-window-local-hr-seed-{seed}.yaml").read_text()
    )
    assert candidate.pop("seed") == seed
    original.pop("seed")
    assert candidate.pop("output_dir") != original.pop("output_dir")
    assert candidate == original

    original_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-window-local-hr-grouped.yaml").read_text()
    )
    grouped = yaml.safe_load(
        (ROOT / f"configs/transformer-window-local-hr-seed-{seed}-grouped.yaml").read_text()
    )
    original_grouped.pop("base_model_config")
    assert grouped.pop("base_model_config") == f"transformer-window-local-hr-seed-{seed}.yaml"
    assert grouped.pop("output_dir") != original_grouped.pop("output_dir")
    assert grouped == original_grouped
    assert grouped["split_seed"] == 42 and grouped["folds"] == 5
    assert "selected_folds" not in grouped



def test_hr_seed_notebook_explains_and_gates_every_operation() -> None:
    """Keep source reports safe and explain the fixed recipe and diagnostic slices."""
    notebook = json.loads((ROOT / "notebooks/30_window_local_hr_seeds.ipynb").read_text())
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
    assert 'study.comparison != "window_features"' in source
    assert 'candidate_controls.pop("window_features")' in source
    assert "short_history_epochs=4" in source
    assert "ConfusionMatrixDisplay" in source and "EpochEvidenceReport" in source
    assert source.count("--train") == 1
