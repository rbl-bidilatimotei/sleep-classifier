import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_motion_learning_rate_configs_change_only_rate_and_output() -> None:
    """Keep the working MPS motion recipe fixed outside the constant rate."""
    reference = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/motion-learning-rate.yaml").read_text())
    assert candidate.pop("learning_rate") == 0.0003
    assert reference.pop("learning_rate") == 0.001
    assert candidate.pop("output_dir") != reference.pop("output_dir")
    assert candidate == reference
    old = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps-grouped.yaml").read_text())
    new = yaml.safe_load((ROOT / "configs/motion-learning-rate-grouped.yaml").read_text())
    assert new.pop("base_model_config") == "motion-learning-rate.yaml"
    old.pop("base_model_config")
    assert new.pop("output_dir") != old.pop("output_dir")
    assert new == old


def test_motion_learning_rate_notebook_is_report_safe_and_self_explanatory() -> None:
    """Preserve opt-in execution and the strict fixed-motion rate comparison."""
    notebook = json.loads((ROOT / "notebooks/39_motion_learning_rate.ipynb").read_text())
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
        "RUN_SETUP", "RUN_PREPARATION", "RUN_CHECK", "RUN_TRAINING",
    ))
    assert 'comparison="learning_rate"' in source
    assert "allow_device_mismatch=True" not in source
    assert "comparison[\"device_comparison\"]" not in source
    assert 'comparison["fixed_motion_inputs"]' in source
    assert "grouped_config = experiment.config_path" in source
