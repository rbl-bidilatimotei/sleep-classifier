import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_patch_size_recipe_changes_only_resolution_and_output_paths() -> None:
    """Keep seed-42 controls fixed while changing three-epoch patches to two epochs."""
    reference = yaml.safe_load((ROOT / "configs/transformer-patch.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/transformer-patch-size-2.yaml").read_text())

    assert reference.pop("patch_size") == 3
    assert candidate.pop("patch_size") == 2
    assert reference.pop("output_dir") != candidate.pop("output_dir")
    assert candidate == reference
    assert candidate["seed"] == 42 and candidate["context_epochs"] == 30
    assert candidate["epochs"] == 40 and candidate["patience"] == 12

    reference_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-patch-grouped.yaml").read_text()
    )
    candidate_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-patch-size-2-grouped.yaml").read_text()
    )
    assert candidate_grouped.pop("base_model_config") == "transformer-patch-size-2.yaml"
    assert reference_grouped.pop("base_model_config") == "transformer-patch.yaml"
    assert candidate_grouped.pop("output_dir") != reference_grouped.pop("output_dir")
    assert candidate_grouped == reference_grouped
    assert candidate_grouped["folds"] == 5 and candidate_grouped["split_seed"] == 42
    assert "selected_folds" not in candidate_grouped


def test_patch_size_notebook_explains_cells_and_requires_training_opt_in() -> None:
    """Keep report-only execution safe and preserve matched temporal diagnostics."""
    notebook = json.loads((ROOT / "notebooks/25_patch_size_comparison.ipynb").read_text())
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
    assert "excluded = {\"patch_size\", \"output_dir\"}" in source
    assert "configs/transformer-patch-grouped.yaml" in source
    assert "TemporalSliceReport.from_grouped(" in source
    assert "PairedArchitectureReport(" in source
    assert "slice_class_rows" in source
    assert "--config configs/transformer-patch-size-2-grouped.yaml --train" in source
