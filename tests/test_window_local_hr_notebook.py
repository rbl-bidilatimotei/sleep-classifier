import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_window_local_hr_recipe_changes_only_transform_and_output_directory() -> None:
    """Freeze the original Adam/P3 controls while adding one causal feature family."""
    reference = yaml.safe_load((ROOT / "configs/transformer-patch.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    assert candidate.pop("window_features") == {
        "name": "heart_rate_deviation", "history_epochs": 10, "min_history_epochs": 4,
    }
    assert candidate.pop("output_dir") != reference.pop("output_dir")
    assert candidate == reference
    assert candidate["context_epochs"] == 30 and candidate["seed"] == 42
    assert candidate["epochs"] == 40 and candidate["patience"] == 12
    assert not {"optimizer", "optimizer_agent", "loss", "lr_scheduler"} & candidate.keys()

    grouped = yaml.safe_load(
        (ROOT / "configs/transformer-window-local-hr-grouped.yaml").read_text()
    )
    original = yaml.safe_load((ROOT / "configs/transformer-patch-grouped.yaml").read_text())
    assert grouped.pop("base_model_config") == "transformer-window-local-hr.yaml"
    assert original.pop("base_model_config") == "transformer-patch.yaml"
    assert grouped.pop("output_dir") != original.pop("output_dir")
    assert grouped == original
    assert grouped["folds"] == 5 and grouped["split_seed"] == 42
    assert "selected_folds" not in grouped


def test_window_local_hr_notebook_explains_cells_and_gates_training() -> None:
    """Keep reports safe and explain the raw-window and normalization boundaries."""
    notebook = json.loads((ROOT / "notebooks/29_window_local_heart_rate.ipynb").read_text())
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        sources.append(source)
        assert cell["outputs"] == [] and cell["execution_count"] is None
        assert index > 0 and notebook["cells"][index - 1]["cell_type"] == "markdown"
        explanation = "".join(notebook["cells"][index - 1]["source"]).lower()
        assert "what" in explanation and "why" in explanation
        if any(marker in source for marker in (
            "if RUN_SETUP:", "if RUN_PREPARATION:", "if RUN_TRAINING:",
            "scripts/execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"]["tags"]
    combined = "\n".join(sources)
    assert "RUN_TRAINING = False" in combined
    assert "RUN_PREPARATION = False" in combined and "RUN_SETUP = False" in combined
    assert combined.count("--train") == 1
    assert 'comparison="window_features"' in combined
    assert 'excluded = {"window_features", "output_dir"}' in combined
    assert '"raw_channels": 6, "effective_channels": 8' in combined
    assert "prior" in "".join(notebook["cells"][7]["source"]).lower()
    assert "forty epochs" in "".join(notebook["cells"][0]["source"])
