import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_label_smoothing_recipe_changes_only_training_loss_and_output_directory() -> None:
    """Keep the replicated Adam recipe fixed while enabling one training-loss intervention."""
    reference = yaml.safe_load((ROOT / "configs/transformer-patch.yaml").read_text())
    candidate = yaml.safe_load(
        (ROOT / "configs/transformer-patch-label-smoothing.yaml").read_text()
    )

    assert candidate.pop("loss") == {"name": "cross_entropy", "label_smoothing": 0.05}
    assert candidate.pop("output_dir") != reference.pop("output_dir")
    assert candidate == reference
    assert candidate["seed"] == 42 and candidate["context_epochs"] == 30
    assert candidate["patch_size"] == 3
    assert candidate["epochs"] == 40 and candidate["patience"] == 12
    assert candidate["learning_rate"] == 0.001
    assert "optimizer" not in candidate and "lr_scheduler" not in candidate

    reference_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-patch-grouped.yaml").read_text()
    )
    candidate_grouped = yaml.safe_load(
        (ROOT / "configs/transformer-patch-label-smoothing-grouped.yaml").read_text()
    )
    assert candidate_grouped.pop("base_model_config") == "transformer-patch-label-smoothing.yaml"
    assert reference_grouped.pop("base_model_config") == "transformer-patch.yaml"
    assert candidate_grouped.pop("output_dir") != reference_grouped.pop("output_dir")
    assert candidate_grouped == reference_grouped
    assert candidate_grouped["folds"] == 5 and candidate_grouped["split_seed"] == 42
    assert "selected_folds" not in candidate_grouped


def test_label_smoothing_notebook_explains_losses_and_gates_training() -> None:
    """Keep the new experiment safe to export and preserve its hard-label evaluation contract."""
    notebook = json.loads((ROOT / "notebooks/27_patch_label_smoothing.ipynb").read_text())
    sources = []
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(set(identifiers)) == len(identifiers)

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
    assert 'excluded = {"loss", "output_dir"}' in source
    assert 'comparison="label_smoothing"' in source
    assert 'configs/transformer-patch-grouped.yaml' in source
    assert 'TemporalSliceReport.from_grouped(' in source
    assert 'measured_recipe != expected_measured_recipe' in source
    assert '"name": "cross_entropy", "label_smoothing": 0.05' in source
    assert 'ZeroDecayControl' not in source

    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "0.9625" in text and "0.0125" in text
    assert (
        "validation and evaluation-mode training NLL use ordinary hard-label cross-entropy" in text
    )
    assert "Do not interpret its distance from validation NLL" in text
    assert '--config configs/transformer-patch-label-smoothing-grouped.yaml --train' in source
