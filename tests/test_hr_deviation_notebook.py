import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_deviation_control_changes_only_the_availability_mask() -> None:
    """Preserve the measured full-HR recipe and all participant-fold controls."""
    reference = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    control = yaml.safe_load((ROOT / "configs/hr-deviation-only.yaml").read_text())
    assert control["window_features"].pop("masked_channels") == ["heart_history_available"]
    assert control.pop("output_dir") != reference.pop("output_dir")
    assert control == reference
    assert control["seed"] == 42 and control["context_epochs"] == 30
    assert control["epochs"] == 40 and control["patience"] == 12

    reference_group = yaml.safe_load(
        (ROOT / "configs/transformer-window-local-hr-grouped.yaml").read_text()
    )
    control_group = yaml.safe_load(
        (ROOT / "configs/hr-deviation-only-grouped.yaml").read_text()
    )
    reference_group.pop("base_model_config")
    assert control_group.pop("base_model_config") == "hr-deviation-only.yaml"
    assert control_group.pop("output_dir") != reference_group.pop("output_dir")
    assert control_group == reference_group
    assert control_group["folds"] == 5 and control_group["split_seed"] == 42
    assert "selected_folds" not in control_group



def test_deviation_notebook_explains_and_gates_training() -> None:
    """Retain explicit opt-in, output-free source, and an unmodified full-HR reference."""
    notebook = json.loads((ROOT / "notebooks/32_hr_deviation_only.ipynb").read_text())
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
    assert 'comparison="window_feature_ablation"' in source
    assert '"masked_channels": ["heart_history_available"]' in source
    assert '"reference_parameters": 2852' in source and '"candidate_parameters": 2852' in source
    assert "short_history_epochs=4" in source
    assert 'reference.model_settings.get("window_features") != expected_window_features' in source
    assert source.count("--train") == 1
