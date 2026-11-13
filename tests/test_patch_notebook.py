import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_patch_recipe_preserves_full_cohort_and_optimization_controls() -> None:
    """Change representation while preserving the stronger reference's training recipe."""
    reference = yaml.safe_load((ROOT / "configs/transformer-dropout-02.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-patch.yaml").read_text())
    assert candidate.pop("model_type") == "patch_transformer"
    assert reference.pop("model_type") == "transformer"
    assert candidate.pop("patch_size") == 3
    diagnostics = candidate.pop("diagnostics")
    expected = yaml.safe_load((ROOT / "configs/transformer-diagnostics.yaml").read_text())
    assert diagnostics == expected["diagnostics"]
    candidate.pop("output_dir")
    reference.pop("output_dir")
    assert candidate == reference

    grouped = yaml.safe_load((ROOT / "configs/transformer-patch-grouped.yaml").read_text())
    assert grouped["folds"] == 5 and grouped["split_seed"] == 42
    assert "selected_folds" not in grouped


def test_patch_notebook_explains_cells_and_gates_model_optimization() -> None:
    """Require readable procedures, output-free source, and explicit training opt-in."""
    notebook = json.loads((ROOT / "notebooks/23_causal_patch_transformer.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert cell["outputs"] == [] and cell["execution_count"] is None
        if not source.strip():
            continue
        assert index > 0 and notebook["cells"][index - 1]["cell_type"] == "markdown"
        explanation = "".join(notebook["cells"][index - 1]["source"]).lower()
        assert "what" in explanation and "why" in explanation
        if any(marker in source for marker in ("if RUN_TRAINING:", "if RUN_PREPARATION:",
                                               "scripts/execute_notebook.py")):
            assert "skip-execution" in cell["metadata"]["tags"]
        if "RUN_TRAINING =" in source:
            assert "RUN_TRAINING = False" in source
            assert "RUN_PREPARATION = False" in source
            assert "RUN_SETUP = False" in source
