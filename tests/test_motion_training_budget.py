import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_budget_configs_keep_the_recipe_fixed() -> None:
    """Restrict the new recipe to its epoch cap, recording policy, and output paths."""
    reference = yaml.safe_load((ROOT / "configs/motion-learning-rate.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/motion-training-budget.yaml").read_text())
    assert reference.pop("epochs") == 40
    assert candidate.pop("epochs") == 80
    assert reference.pop("output_dir") != candidate.pop("output_dir")
    reference["diagnostics"].pop("checkpoint_epochs")
    assert candidate["diagnostics"].pop("checkpoint_epochs") == list(range(1, 41))
    assert candidate == reference
    grouped_reference = yaml.safe_load(
        (ROOT / "configs/motion-learning-rate-grouped.yaml").read_text(),
    )
    grouped_candidate = yaml.safe_load(
        (ROOT / "configs/motion-training-budget-grouped.yaml").read_text(),
    )
    assert grouped_reference.pop("output_dir") != grouped_candidate.pop("output_dir")
    assert grouped_reference.pop("base_model_config") == "motion-learning-rate.yaml"
    assert grouped_candidate.pop("base_model_config") == "motion-training-budget.yaml"
    assert grouped_candidate == grouped_reference


def test_budget_notebook_explains_cells_and_disables_opt_in_operations() -> None:
    """Keep fresh report execution descriptive, output-free, and unable to train."""
    notebook = json.loads((ROOT / "notebooks/40_motion_training_budget.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for flag in ("RUN_SETUP", "RUN_PREPARATION", "RUN_CHECK", "RUN_TRAINING"):
        assert f"{flag} = False" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert notebook["cells"][index - 1]["cell_type"] == "markdown"
        explanation = "".join(notebook["cells"][index - 1]["source"])
        assert "What" in explanation and "Why" in explanation
        assert not cell["outputs"] and cell["execution_count"] is None
        code = "".join(cell["source"])
        if any(token in code for token in (
            "if RUN_SETUP:", "if RUN_PREPARATION:", "if RUN_CHECK:", "if RUN_TRAINING:",
            "scripts/execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"].get("tags", [])
    assert "PrefixBudgetReport" in source
    assert "original_notebook_39_restored" not in source
