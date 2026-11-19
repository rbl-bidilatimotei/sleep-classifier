import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]



def test_motion_grid_audit_config_preserves_reference_and_development_boundaries() -> None:
    """Declare only sensor diagnostics and the completed full-HR/horizon comparison."""
    config = yaml.safe_load((ROOT / "configs/motion-grid-audit.yaml").read_text())
    assert config["reference_summary"] == (
        "../artifacts/transformer-window-local-hr-grouped-validation/summary.json"
    )
    assert config["comparisons"] == {"history_20": {
        "summary": "../artifacts/hr-history-20-grouped-validation/summary.json",
        "comparison": "window_history",
    }}
    assert config["motion_grid"] == {
        "step_seconds": 1.0, "max_sample_age_seconds": 0.2,
        "support_fractions": [0.5, 0.8, 0.95],
    }
    assert config["heart_history_epochs"] == 10 and config["min_heart_history_epochs"] == 4
    assert config["output_dir"] == "../artifacts/motion-grid-audit"



def test_motion_grid_notebook_gates_raw_reads_and_explains_each_cell() -> None:
    """Keep cached reporting free from raw reads, model training, and recursive export."""
    notebook = json.loads((ROOT / "notebooks/35_motion_grid_audit.ipynb").read_text())
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        sources.append(source)
        assert not cell["outputs"] and cell["execution_count"] is None
        previous = notebook["cells"][index - 1]
        assert previous["cell_type"] == "markdown"
        text = "".join(previous["source"]).lower()
        assert "what" in text and "why" in text
        if any(marker in source for marker in (
            "if RUN_SETUP:", "if RUN_AUDIT:", "scripts/execute_notebook.py",
        )):
            assert "skip-execution" in cell["metadata"]["tags"]
    source = "\n".join(sources)
    assert "RUN_SETUP = False" in source and "RUN_AUDIT = False" in source
    assert "--train" not in source and "RUN_TRAINING" not in source
    assert "audit_config = experiment.settings.config_path" in source
    assert "experiment.load_report()" in source
    assert "FixedLagMotionFeatures" in source and "return_before_next_tick" in source
