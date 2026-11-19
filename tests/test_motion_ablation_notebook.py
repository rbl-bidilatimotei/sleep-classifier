import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_motion_ablation_keeps_all_other_recipe_settings_fixed() -> None:
    """Permit only the added motion source and a fresh output directory."""
    reference = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/fixed-lag-motion.yaml").read_text())
    assert candidate.pop("motion_features") == {
        "name": "fixed_lag_motion", "audit_config": "motion-grid-audit.yaml",
    }
    assert candidate.pop("output_dir") != reference.pop("output_dir")
    assert candidate == reference
    grouped = yaml.safe_load((ROOT / "configs/fixed-lag-motion-grouped.yaml").read_text())
    assert grouped["folds"] == 5 and grouped["split_seed"] == 42
    assert grouped["source_prepare_config"] == "prepare.yaml"


def test_motion_ablation_notebook_explains_cells_and_gates_execution() -> None:
    """Keep source reports output-free and training under explicit operator control."""
    notebook = json.loads((ROOT / "notebooks/36_fixed_lag_motion.ipynb").read_text())
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        sources.append(source)
        assert not cell["outputs"] and cell["execution_count"] is None
        previous = notebook["cells"][index - 1]
        assert previous["cell_type"] == "markdown"
        assert all(word in "".join(previous["source"]).lower() for word in ("what", "why"))
        if any(marker in source for marker in (
            "if RUN_SETUP:", "if RUN_PREPARATION:", "if RUN_TRAINING:", "execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"]["tags"]
    source = "\n".join(sources)
    assert all(f"{flag} = False" in source for flag in (
        "RUN_SETUP", "RUN_PREPARATION", "RUN_TRAINING",
    ))
    assert "experiment.validate_training_start()" in source
    assert 'comparison="motion_features"' in source
    assert "grouped_config = experiment.config_path" in source
