import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_recipe_changes_only_observation_controls_and_output_location() -> None:
    """Keep the diagnostic model identical to the measured seed-42 dropout recipe."""
    reference = yaml.safe_load((ROOT / "configs/transformer-dropout-02.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-diagnostics.yaml").read_text())
    diagnostics = candidate.pop("diagnostics")
    reference.pop("output_dir")
    candidate.pop("output_dir")

    assert candidate == reference
    assert diagnostics["enabled"] and diagnostics["evaluate_training"]
    assert diagnostics["save_predictions"]
    assert diagnostics["checkpoint_epochs"] == [13, 14, 15, 16, 17]
    grouped = yaml.safe_load((ROOT / "configs/transformer-diagnostics-grouped.yaml").read_text())
    assert grouped["folds"] == 5
    assert grouped["split_seed"] == 42
    assert grouped["selected_folds"] == ["fold-01", "fold-04"]


def test_diagnostic_notebook_explains_every_code_cell_and_gates_optimization() -> None:
    """Require self-contained explanations and safe report-only execution defaults."""
    notebook = json.loads((ROOT / "notebooks/20_transformer_diagnostics.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        explanation = notebook["cells"][index - 1]
        assert index > 0 and explanation["cell_type"] == "markdown"
        text = "".join(explanation["source"]).lower()
        assert "what" in text and "why" in text
        assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "".join(cell["source"])
        if "if RUN_TRAINING:" in source or "if RUN_PREPARATION:" in source:
            assert "skip-execution" in cell["metadata"]["tags"]
        if "scripts/execute_notebook.py" in source:
            assert "skip-execution" in cell["metadata"]["tags"]
    configuration = "".join(notebook["cells"][2]["source"])
    assert "RUN_TRAINING = False" in configuration
    assert "RUN_PREPARATION = False" in configuration
    assert "RUN_SETUP = False" in configuration
