import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_learning_rate_candidate_changes_only_rate_and_output_location() -> None:
    """Keep the model and evidence controls fixed for the isolated optimization test."""
    reference = yaml.safe_load((ROOT / "configs/transformer-diagnostics.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-learning-rate.yaml").read_text())
    assert reference.pop("learning_rate") == 0.001
    assert candidate.pop("learning_rate") == 0.0003
    reference.pop("output_dir")
    candidate.pop("output_dir")
    assert candidate == reference

    grouped = yaml.safe_load((ROOT / "configs/transformer-learning-rate-grouped.yaml").read_text())
    assert grouped["folds"] == 5 and grouped["split_seed"] == 42
    assert grouped["selected_folds"] == ["fold-01", "fold-04"]


def test_learning_rate_notebook_explains_cells_and_protects_report_execution() -> None:
    """Require What/Why explanations, clean outputs, and explicit optimization gates."""
    notebook = json.loads((ROOT / "notebooks/21_transformer_learning_rate.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert cell["outputs"] == [] and cell["execution_count"] is None
        if not source.strip():
            continue
        assert index > 0
        previous = notebook["cells"][index - 1]
        assert previous["cell_type"] == "markdown"
        explanation = "".join(previous["source"]).lower()
        assert "what" in explanation and "why" in explanation
        if any(marker in source for marker in ["if RUN_TRAINING:", "if RUN_PREPARATION:",
                                               "scripts/execute_notebook.py"]):
            assert "skip-execution" in cell["metadata"]["tags"]
        if "RUN_TRAINING =" in source:
            assert "RUN_TRAINING = False" in source
            assert "RUN_PREPARATION = False" in source
            assert "RUN_SETUP = False" in source
