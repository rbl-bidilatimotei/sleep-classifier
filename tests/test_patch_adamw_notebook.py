import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_adamw_recipe_changes_only_optimizer_and_output_directory() -> None:
    """Keep the replicated patch recipe fixed while enabling one explicit decay policy."""
    reference = yaml.safe_load((ROOT / "configs/transformer-patch.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-patch-adamw.yaml").read_text())

    assert candidate.pop("optimizer") == {
        "name": "adamw", "weight_decay": 0.01, "exclude_bias_and_norm": True,
    }
    assert candidate.pop("output_dir") != reference.pop("output_dir")
    assert candidate == reference
    assert candidate["seed"] == 42 and candidate["context_epochs"] == 30
    assert candidate["patch_size"] == 3
    assert candidate["epochs"] == 40 and candidate["patience"] == 12
    assert candidate["learning_rate"] == 0.001

    reference_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-patch-grouped.yaml").read_text()
    )
    candidate_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-patch-adamw-grouped.yaml").read_text()
    )
    assert candidate_grouped.pop("base_model_config") == "transformer-patch-adamw.yaml"
    assert reference_grouped.pop("base_model_config") == "transformer-patch.yaml"
    assert candidate_grouped.pop("output_dir") != reference_grouped.pop("output_dir")
    assert candidate_grouped == reference_grouped
    assert candidate_grouped["folds"] == 5 and candidate_grouped["split_seed"] == 42
    assert "selected_folds" not in candidate_grouped


def test_adamw_notebook_explains_cells_and_requires_a_fresh_zero_control() -> None:
    """Keep training opt-in and verify optimizer equivalence before actual candidate updates."""
    notebook = json.loads((ROOT / "notebooks/26_patch_adamw.ipynb").read_text())
    sources = []

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])
        sources.append(source)
        assert cell["outputs"] == [] and cell["execution_count"] is None
        if not source.strip():
            continue

        assert index > 0 and notebook["cells"][index - 1]["cell_type"] == "markdown"
        description = "".join(notebook["cells"][index - 1]["source"]).lower()
        assert "what" in description and "why" in description
        if any(marker in source for marker in (
            "if RUN_SETUP:", "if RUN_PREPARATION:", "if RUN_TRAINING:",
            "scripts/execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"]["tags"]

    source = "\n".join(sources)
    for setting in ("RUN_SETUP = False", "RUN_PREPARATION = False", "RUN_TRAINING = False"):
        assert setting in source
    assert source.count("--train") == 1
    assert 'excluded = {"optimizer", "output_dir"}' in source
    assert 'comparison="weight_decay"' in source
    assert 'configs/transformer-patch-grouped.yaml' in source
    assert 'TemporalSliceReport.from_grouped(' in source
    assert 'control_devices = list(dict.fromkeys(["cpu", resolved_device]))' in source
    assert 'if not all(result["passed"] for result in zero_decay_controls):' in source

    control_cell = next(source for source in sources if 'zero_decay_controls = [' in source)
    preparation_cell = next(source for source in sources if 'if RUN_PREPARATION:' in source)
    training_cell = next(source for source in sources if 'if RUN_TRAINING:' in source)
    assert sources.index(control_cell) < sources.index(preparation_cell)
    assert (
        'ZeroDecayControl(experiment.model_settings, device=resolved_device).run()'
        in training_cell
    )
    assert training_cell.index('if not launch_control["passed"]:') < training_cell.index('--train')
    assert '--config configs/transformer-patch-adamw-grouped.yaml --train' in training_cell
