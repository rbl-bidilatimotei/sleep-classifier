import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_history_notebook() -> dict[str, Any]:
    """Read the output-free history-horizon notebook source."""
    return json.loads((ROOT / "notebooks/34_hr_history_horizon.ipynb").read_text())


def history_control_guard() -> str:
    """Extract the actual notebook guard for the frozen model controls."""
    source = next(
        "".join(cell["source"]) for cell in load_history_notebook()["cells"]
        if cell["cell_type"] == "code" and "excluded = " in "".join(cell["source"])
    )
    return source[source.index("excluded = "):source.index("artifact_fields = ")]


def test_history_candidate_changes_only_nested_history_and_output() -> None:
    """Preserve the measured unmasked recipe and every grouped-fold control."""
    reference = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/hr-history-20.yaml").read_text())
    assert candidate.pop("output_dir") == "../artifacts/hr-history-20-single"
    reference.pop("output_dir")
    assert candidate["window_features"].pop("history_epochs") == 20
    assert reference["window_features"].pop("history_epochs") == 10
    assert candidate == reference
    assert candidate["window_features"] == {
        "name": "heart_rate_deviation", "min_history_epochs": 4,
    }
    assert candidate["context_epochs"] == 30 and candidate["patch_size"] == 3
    assert candidate["transformer"]["d_model"] == 16
    assert candidate["transformer"]["dropout"] == 0.2
    assert candidate["learning_rate"] == 0.001 and candidate["seed"] == 42
    assert candidate["epochs"] == 40 and candidate["patience"] == 12
    assert not any(candidate.get(key) for key in (
        "optimizer", "optimizer_agent", "lr_scheduler", "loss",
    ))

    grouped = yaml.safe_load((ROOT / "configs/hr-history-20-grouped.yaml").read_text())
    reference_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-window-local-hr-grouped.yaml").read_text()
    )
    assert grouped.pop("base_model_config") == "hr-history-20.yaml"
    reference_grouped.pop("base_model_config")
    assert grouped.pop("output_dir") == "../artifacts/hr-history-20-grouped-validation"
    reference_grouped.pop("output_dir")
    assert grouped == reference_grouped
    assert grouped["source_prepare_config"] == "prepare.yaml"
    assert grouped["folds"] == 5 and grouped["split_seed"] == 42
    assert grouped["folds"] * candidate["epochs"] == 200
    assert "selected_folds" not in grouped


@pytest.mark.parametrize("nested_change", [
    None, {"min_history_epochs": 5}, {"masked_channels": ["heart_history_available"]},
    {"name": "different_transform"}, {"undeclared_setting": True},
])
def test_notebook_history_guard_rejects_other_nested_changes(
    nested_change: dict[str, Any] | None,
) -> None:
    """Allow only the declared horizon without dropping other nested controls.

    Args:
        nested_change (dict[str, Any] | None): Additional candidate transform changes.
    """
    reference = yaml.safe_load((ROOT / "configs/transformer-window-local-hr.yaml").read_text())
    candidate = copy.deepcopy(reference)
    candidate["window_features"]["history_epochs"] = 20
    candidate["output_dir"] = "../artifacts/hr-history-20-single"
    if nested_change:
        candidate["window_features"].update(nested_change)
    unchanged_reference = copy.deepcopy(reference)
    unchanged_candidate = copy.deepcopy(candidate)
    namespace = {
        "reference": SimpleNamespace(model_settings=reference),
        "experiment": SimpleNamespace(model_settings=candidate),
    }
    if nested_change:
        with pytest.raises(ValueError, match="Another model, optimization, feature"):
            exec(history_control_guard(), namespace)
    else:
        exec(history_control_guard(), namespace)
    assert reference == unchanged_reference
    assert candidate == unchanged_candidate


def test_history_notebook_explains_and_gates_operations() -> None:
    """Keep explicit opt-in and describe the matched history-only comparison."""
    notebook = load_history_notebook()
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
    assert "RUN_CHECK = True" in source
    assert 'BRANCH = "experiment/hr-history-20"' in source
    assert source.count('comparison="window_history"') == 2
    assert source.count("--train") == 1
    assert '"raw_channels": 6, "effective_channels": 8' in source
    assert '"reference_parameters": 2852' in source and '"candidate_parameters": 2852' in source
    assert "short_history_epochs=4" in source
    assert 'experiment.validate_training_start()' in source
    assert "ConfusionMatrixDisplay" in source and "validation_nll" in source
    assert "[50.0] * 10 + [60.0] * 10 + [70.0]" in source
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "markdown")
    assert "../docs/hr-history-horizon.md" in markdown
    assert "raw context never extends" in markdown
    assert "valid-pair count" in markdown and "standard deviation may" in markdown
    assert "No candidate conclusion" in markdown


def test_history_example_retains_flag_and_changes_only_raw_baseline() -> None:
    """Verify the notebook's raw-BPM example without importing training modules."""
    source = next(
        "".join(cell["source"]) for cell in load_history_notebook()["cells"]
        if cell["cell_type"] == "code" and "example_heart_rate = " in "".join(cell["source"])
    )
    namespace: dict[str, Any] = {}
    exec(source[:source.index("display(")], namespace)
    rows = namespace["example_rows"]
    assert [row["history_epochs"] for row in rows] == [10, 20]
    assert [row["strictly_past_mean_bpm"] for row in rows] == [60.0, 55.0]
    assert [row["raw_deviation_bpm"] for row in rows] == [10.0, 15.0]
    assert [row["availability"] for row in rows] == [1, 1]
