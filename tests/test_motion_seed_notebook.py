import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_motion_seed_configs_change_only_seed_and_output_paths() -> None:
    """Keep the completed MPS candidate recipe fixed in both new candidates."""
    base = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps.yaml").read_text())
    for seed in (43, 44):
        config = ROOT / f"configs/fixed-lag-motion-mps-seed-{seed}.yaml"
        candidate = yaml.safe_load(config.read_text())
        assert candidate.pop("seed") == seed
        assert candidate.pop("output_dir") != base["output_dir"]
        expected = {key: value for key, value in base.items() if key not in {"seed", "output_dir"}}
        assert candidate == expected
        assert candidate["device"] == "mps"
    study = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps-seeds.yaml").read_text())
    assert study["comparison"] == "motion_features"
    assert [pair["seed"] for pair in study["pairs"]] == [42, 43, 44]
    assert all(pair["reuse_reference"] for pair in study["pairs"])
    assert [pair["reuse_candidate"] for pair in study["pairs"]] == [True, False, False]
    assert study["pairs"][0]["candidate_config"] == "fixed-lag-motion-mps-grouped.yaml"


def test_motion_seed_notebook_explains_cells_and_preserves_execution_gates() -> None:
    """Make cached seed reporting safe without enabling training or device exceptions."""
    notebook = json.loads((ROOT / "notebooks/38_motion_mps_seeds.ipynb").read_text())
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
    assert "study.validate_reused_results()" in source
    assert 'experiment.validate_training_start(expected_device="mps")' in source
    assert "allow_device_mismatch=True" not in source
    assert "grouped_config = experiment.config_path" in source
