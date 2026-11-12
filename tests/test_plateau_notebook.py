import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_plateau_recipe_adds_only_the_agreed_scheduler() -> None:
    """Keep the stronger reference unchanged apart from scheduling and output storage."""
    reference = yaml.safe_load((ROOT / "configs/transformer-diagnostics.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-plateau.yaml").read_text())
    controls = candidate.pop("lr_scheduler")
    reference.pop("output_dir")
    candidate.pop("output_dir")
    assert candidate == reference
    assert controls == {"kind": "reduce_on_plateau", "monitor": "validation_macro_f1",
                        "mode": "max", "factor": 0.3, "patience": 3,
                        "threshold": 0.0001, "threshold_mode": "abs", "cooldown": 0,
                        "min_lr": 0.00003, "eps": 1e-8}
    grouped = yaml.safe_load((ROOT / "configs/transformer-plateau-grouped.yaml").read_text())
    assert grouped["folds"] == 5 and grouped["split_seed"] == 42
    assert grouped["selected_folds"] == ["fold-01", "fold-04"]


def test_plateau_notebook_explains_each_cell_and_gates_training() -> None:
    """Require safe defaults, clean outputs, and What/Why text before executable cells."""
    notebook = json.loads((ROOT / "notebooks/22_transformer_plateau.ipynb").read_text())
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


@pytest.mark.parametrize("reduced", [True, False])
def test_plateau_notebook_displays_actual_rate_history_and_warns_if_unused(
    tmp_path: Path, reduced: bool,
) -> None:
    """Inspect saved rate transitions and identify a schedule with no observed reduction.

    Args:
        tmp_path (Path): Temporary saved scheduler-history directory.
        reduced (bool): Whether the synthetic optimizer reduced its next-epoch rate.
    """
    upcoming = [0.0003] if reduced else [0.001]
    directory = tmp_path / "fold-01/results"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps({"history": [
        {"epoch": 1, "learning_rates_used": [0.001],
         "learning_rates_next_epoch": upcoming, "scheduler_reduced": reduced},
        {"epoch": 2, "learning_rates_used": upcoming,
         "learning_rates_next_epoch": None, "scheduler_reduced": False},
    ]}))
    notebook = json.loads((ROOT / "notebooks/22_transformer_plateau.ipynb").read_text())
    namespace = {"plan": {"folds": [{"name": "fold-01"}]},
                 "experiment": SimpleNamespace(output_dir=tmp_path), "json": json,
                 "pd": pd, "logger": Mock(), "display": Mock()}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("scheduler_rows = []"):
            exec(source, namespace)
    assert len(namespace["scheduler_rows"]) == 2
    assert namespace["scheduler_rows"][0]["learning_rates_used"] == [0.001]
    assert namespace["scheduler_rows"][1]["learning_rates_next_epoch"] is None
    assert namespace["logger"].warning.called is (not reduced)
