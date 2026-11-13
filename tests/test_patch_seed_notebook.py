import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("sklearn")

from sleep_classifier.paired_seed_report import PairedSeedReport  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_patch_seed_study_reuses_measured_results_and_limits_new_work() -> None:
    """Prepare only the two new patch candidates while retaining every measured reference."""
    study = PairedSeedReport(ROOT / "configs/transformer-patch-seeds.yaml")
    plan = study.plan()
    assert study.comparison == "architecture"
    assert plan["new_fits"] == 10 and plan["maximum_new_epochs"] == 400
    assert [(run["seed"], run["recipe"]) for run in study.new_runs()] == [
        (43, "candidate"), (44, "candidate"),
    ]
    assert all(case["reuse_reference"] for case in study.cases)
    assert [case["reuse_candidate"] for case in study.cases] == [True, False, False]
    canonical = yaml.safe_load((ROOT / "configs/transformer-patch.yaml").read_text())
    canonical.pop("seed")
    canonical.pop("output_dir")
    for seed in (43, 44):
        model = yaml.safe_load((ROOT / f"configs/transformer-patch-seed-{seed}.yaml").read_text())
        assert model.pop("seed") == seed
        model.pop("output_dir")
        assert model == canonical
        grouped = yaml.safe_load(
            (ROOT / f"configs/transformer-patch-seed-{seed}-grouped.yaml").read_text()
        )
        assert grouped["folds"] == 5 and grouped["split_seed"] == 42
        assert "selected_folds" not in grouped


def test_patch_seed_notebook_explains_cells_and_gates_optimization() -> None:
    """Keep report execution safe and each source code cell self-explanatory."""
    notebook = json.loads((ROOT / "notebooks/24_patch_transformer_seeds.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert cell["outputs"] == [] and cell["execution_count"] is None
        if not source.strip():
            continue
        assert index > 0 and notebook["cells"][index - 1]["cell_type"] == "markdown"
        text = "".join(notebook["cells"][index - 1]["source"]).lower()
        assert "what" in text and "why" in text
        if any(marker in source for marker in ("if RUN_TRAINING:", "if RUN_PREPARATION:",
                                               "scripts/execute_notebook.py")):
            assert "skip-execution" in cell["metadata"]["tags"]
        if "RUN_TRAINING =" in source:
            assert "RUN_TRAINING = False" in source
            assert "RUN_PREPARATION = False" in source
            assert "RUN_SETUP = False" in source
