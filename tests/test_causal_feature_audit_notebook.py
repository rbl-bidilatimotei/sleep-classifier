import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_causal_feature_audit_recipe_uses_only_completed_development_evidence() -> None:
    """Freeze the audit inputs and causal windows without introducing a model recipe."""
    config = yaml.safe_load((ROOT / "configs/causal-feature-audit.yaml").read_text())
    assert config["reference_summary"] == (
        "../artifacts/transformer-patch-grouped-validation/summary.json"
    )
    assert set(config["comparisons"]) == {"patch_size_2", "adamw", "label_smoothing"}
    assert {name: case["comparison"] for name, case in config["comparisons"].items()} == {
        "patch_size_2": "architecture", "adamw": "weight_decay",
        "label_smoothing": "label_smoothing",
    }
    assert config["heart_history_epochs"] == 10
    assert config["min_heart_history_epochs"] == 4
    assert config["max_motion_pair_gap_seconds"] == 0.2
    assert not {"epochs", "learning_rate", "optimizer", "train", "model_type"} & config.keys()


def test_causal_audit_notebook_explains_cells_and_gates_raw_observation_access() -> None:
    """Keep automatic reports output-free and unable to launch the raw audit implicitly."""
    notebook = json.loads((ROOT / "notebooks/28_causal_feature_audit.ipynb").read_text())
    identifiers = [cell["id"] for cell in notebook["cells"]]
    assert len(identifiers) == len(set(identifiers))
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
        explanation = "".join(notebook["cells"][index - 1]["source"]).lower()
        assert "what" in explanation and "why" in explanation
        if any(marker in source for marker in (
            "if RUN_SETUP:", "if RUN_AUDIT:", "scripts/execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"]["tags"]

    source = "\n".join(sources)
    assert "RUN_SETUP = False" in source and "RUN_AUDIT = False" in source
    assert "--train" not in source and "--prepare" not in source
    assert "torch" not in source and "tensorflow" not in source
    assert source.count("--run") == 1
    raw_cell = next(text for text in sources if "--run" in text)
    assert "if RUN_AUDIT:" in raw_cell and raw_cell.index("if RUN_AUDIT:") < raw_cell.index("--run")
    assert "experiment.load_report()" in source
    assert 'plan["test_observations_read"] is not False' in source
    assert 'plan["model_training_performed"] is not False' in source
    assert 'plan["candidate_features_integrated"] is not False' in source
    assert "observed & reserved_test" in source


def test_causal_audit_notebook_links_the_immutable_dashboard_without_rewriting_it() -> None:
    """Display verified snapshot evidence without replacing manifest-linked artifacts."""
    notebook = json.loads((ROOT / "notebooks/28_causal_feature_audit.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code")
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "FileLink(str(dashboard)" in source
    assert "write_html" not in source and "write_text" not in source
    assert "visualization.plot_distributions()" in source
    assert "visualization.plot_persistent_errors()" in source
    assert 'summary["training_fold_stage_summary"]' in source
    assert 'summary["candidate_availability"]' in source
    assert 'summary["matched_error_summary"]' in source
    assert "strictly previous" in text and "segment boundaries" in text
    assert "does not compress the timeline" in text
    assert "not independent evidence" in text
