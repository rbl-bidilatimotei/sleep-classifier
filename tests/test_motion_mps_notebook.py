import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_motion_mps_repeat_changes_only_device_and_output_paths() -> None:
    """Require explicit MPS while preserving the original seed-42 motion recipe."""
    previous = yaml.safe_load((ROOT / "configs/fixed-lag-motion.yaml").read_text())
    repeat = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps.yaml").read_text())
    assert repeat.pop("device") == "mps"
    previous.pop("device")
    assert repeat.pop("output_dir") != previous.pop("output_dir")
    assert repeat == previous
    old_group = yaml.safe_load((ROOT / "configs/fixed-lag-motion-grouped.yaml").read_text())
    new_group = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps-grouped.yaml").read_text())
    assert new_group.pop("base_model_config") == "fixed-lag-motion-mps.yaml"
    old_group.pop("base_model_config")
    assert new_group.pop("output_dir") != old_group.pop("output_dir")
    assert new_group == old_group


def test_motion_mps_notebook_gates_training_and_requires_matching_devices() -> None:
    """Keep the MPS repeat self-explanatory and reject exploratory device exceptions."""
    notebook = json.loads((ROOT / "notebooks/37_fixed_lag_motion_mps.ipynb").read_text())
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
    assert "allow_device_mismatch=True" not in source and "ALLOW_DEVICE_MISMATCH" not in source
    assert source.count("allow_device_mismatch=False") == 2
    assert 'reference_manifest["device"] != "mps"' in source
    assert 'prepared["device"] != "mps"' in source
    assert 'comparison="motion_features"' in source
