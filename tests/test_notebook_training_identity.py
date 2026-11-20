import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("IPython")

from IPython.core.inputtransformer2 import TransformerManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class RecordingNotebookShell:
    """Record notebook shell commands without launching any process."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        """Keep interpolation variables and a successful synthetic command status.

        Args:
            namespace (dict[str, Any]): Variables available to the notebook cell.
        """
        self.user_ns = namespace
        self.user_ns["_exit_code"] = 0
        self.commands: list[str] = []


    def system(self, command: str) -> None:
        """Record an expanded shell command without executing it.

        Args:
            command (str): IPython shell input with optional notebook variables.
        """
        self.commands.append(command.format_map(self.user_ns))


    def run_line_magic(self, name: str, arguments: str) -> None:
        """Accept the notebook directory change without altering test isolation.

        Args:
            name (str): IPython magic name.
            arguments (str): Magic arguments from the source cell.
        """
        assert name == "cd"


@pytest.mark.parametrize("notebook_name", [
    "31_hr_availability_control.ipynb", "32_hr_deviation_only.ipynb",
    "34_hr_history_horizon.ipynb", "36_fixed_lag_motion.ipynb",
    "37_fixed_lag_motion_mps.ipynb",
])
def test_training_cell_uses_the_validated_experiment_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notebook_name: str,
) -> None:
    """Prevent a stale cell from launching a different hardcoded experiment.

    Args:
        tmp_path (Path): Isolated notebook working directory.
        monkeypatch (pytest.MonkeyPatch): Scoped working-directory replacement.
        notebook_name (str): Notebook whose actual training cell is exercised.
    """
    monkeypatch.chdir(tmp_path)
    active_config = ("hr-deviation-only-grouped.yaml" if notebook_name.startswith("31_")
                     else "hr-availability-control-grouped.yaml")
    masked_channel = ("heart_history_available" if notebook_name.startswith("31_")
                      else "heart_rate_delta_past")
    config = tmp_path / "configs" / active_config
    config.parent.mkdir()
    config.write_text("output_dir: ../candidate\n")
    summary = tmp_path / "reference.json"
    plan = {"folds": []}
    summary.write_text(json.dumps({"plan": plan}))
    experiment = SimpleNamespace(
        config_path=config.resolve(), output_dir=tmp_path / "candidate",
        model_settings={"window_features": {"masked_channels": [masked_channel],
                                            "history_epochs": 10}},
        validate_training_start=Mock(),
    )
    namespace = {
        "ROOT": tmp_path, "RUN_TRAINING": True, "Path": Path, "json": json,
        "experiment": experiment, "reference_summary": summary,
        "reference": SimpleNamespace(plan=Mock(return_value=plan)),
        "PairedArchitectureReport": Mock(), "logger": Mock(),
        "reference_manifest": {"device": "cpu"},
    }
    shell = RecordingNotebookShell(namespace)
    namespace["get_ipython"] = Mock(return_value=shell)
    notebook = json.loads((ROOT / "notebooks" / notebook_name).read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "if RUN_TRAINING:" in "".join(cell["source"]))
    exec(TransformerManager().transform_cell(source), namespace)
    assert len(shell.commands) == 1
    arguments = shlex.split(shell.commands[0])
    invoked_config = Path(arguments[arguments.index("--config") + 1]).resolve()
    assert invoked_config == experiment.config_path
    if notebook_name.startswith(("36_", "37_")):
        experiment.validate_training_start.assert_called_once_with(expected_device="cpu")
    else:
        experiment.validate_training_start.assert_called_once_with()


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("notebook_name", [
    "33_hr_deviation_only_seeds.ipynb", "38_motion_mps_seeds.ipynb",
])
def test_seed_training_cell_binds_each_validated_config_and_checks_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool, notebook_name: str,
) -> None:
    """Use each resolved experiment target and stop blocked runs before any CLI command.

    Args:
        tmp_path (Path): Isolated notebook working directory.
        monkeypatch (pytest.MonkeyPatch): Scoped working-directory replacement.
        blocked (bool): Whether the first readiness check rejects existing results.
        notebook_name (str): Source notebook containing the training cell.
    """
    monkeypatch.chdir(tmp_path)
    runs = [{"seed": seed, "config": str(tmp_path / f"alias-{seed}.yaml")}
            for seed in (43, 44)]
    experiments = [SimpleNamespace(
        config_path=tmp_path / f"resolved-{seed}.yaml", output_dir=tmp_path / str(seed),
        model_settings={"window_features": {"masked_channels": ["heart_history_available"]}},
        validate_training_start=Mock(),
    ) for seed in (43, 44)]
    if blocked:
        experiments[0].validate_training_start.side_effect = FileExistsError(
            "Retain existing results",
        )
    factory = Mock(side_effect=experiments)
    namespace = {
        "ROOT": tmp_path, "RUN_TRAINING": True, "Path": Path, "logger": Mock(),
        "study": SimpleNamespace(validate_reused_results=Mock(), new_runs=Mock(return_value=runs)),
        "GroupedValidationExperiment": factory,
    }
    shell = RecordingNotebookShell(namespace)
    namespace["get_ipython"] = Mock(return_value=shell)
    notebook = json.loads((ROOT / "notebooks" / notebook_name).read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "if RUN_TRAINING:" in "".join(cell["source"]))
    transformed = TransformerManager().transform_cell(source)
    if blocked:
        with pytest.raises(FileExistsError, match="Retain existing results"):
            exec(transformed, namespace)
        assert shell.commands == []
        experiments[1].validate_training_start.assert_not_called()
    else:
        exec(transformed, namespace)
        assert len(shell.commands) == 2
        for command, experiment in zip(shell.commands, experiments, strict=True):
            arguments = shlex.split(command)
            assert Path(arguments[arguments.index("--config") + 1]) == experiment.config_path
            if notebook_name.startswith("38_"):
                experiment.validate_training_start.assert_called_once_with(expected_device="mps")
            else:
                experiment.validate_training_start.assert_called_once_with()
    namespace["study"].validate_reused_results.assert_called_once_with()


@pytest.mark.parametrize("blocked", [False, True])
def test_motion_audit_cell_uses_validated_config_and_preserves_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool,
) -> None:
    """Bind the actual raw-audit cell to its active config without launching processes.

    Args:
        tmp_path (Path): Isolated notebook working directory.
        monkeypatch (pytest.MonkeyPatch): Scoped directory replacement.
        blocked (bool): Whether a valid existing snapshot must skip the raw command.
    """
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "different-audit.yaml"
    output = tmp_path / "cached-audit"
    if blocked:
        output.mkdir()
    namespace = {
        "RUN_AUDIT": True, "ROOT": tmp_path, "Path": Path, "logger": Mock(),
        "experiment": SimpleNamespace(settings=SimpleNamespace(config_path=config),
                                      output_dir=output, load_report=Mock()),
    }
    shell = RecordingNotebookShell(namespace)
    namespace["get_ipython"] = Mock(return_value=shell)
    notebook = json.loads((ROOT / "notebooks/35_motion_grid_audit.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "if RUN_AUDIT:" in "".join(cell["source"]))
    transformed = TransformerManager().transform_cell(source)
    if blocked:
        exec(transformed, namespace)
        namespace["experiment"].load_report.assert_called_once_with()
        assert shell.commands == []
    else:
        exec(transformed, namespace)
        assert len(shell.commands) == 1
        arguments = shlex.split(shell.commands[0])
        assert Path(arguments[arguments.index("--config") + 1]) == config
        assert "--run" in arguments and "--train" not in arguments


def test_motion_audit_cell_rejects_invalid_cache_without_raw_command(tmp_path: Path) -> None:
    """Preserve invalid cached evidence and surface its validation error.

    Args:
        tmp_path (Path): Isolated existing snapshot directory.
    """
    namespace = {
        "RUN_AUDIT": True, "logger": Mock(),
        "experiment": SimpleNamespace(
            settings=SimpleNamespace(config_path=tmp_path / "audit.yaml"), output_dir=tmp_path,
            load_report=Mock(side_effect=ValueError("Snapshot hash mismatch")),
        ),
    }
    shell = RecordingNotebookShell(namespace)
    namespace["get_ipython"] = Mock(return_value=shell)
    notebook = json.loads((ROOT / "notebooks/35_motion_grid_audit.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "if RUN_AUDIT:" in "".join(cell["source"]))

    with pytest.raises(ValueError, match="Snapshot hash mismatch"):
        exec(TransformerManager().transform_cell(source), namespace)

    assert shell.commands == []


def test_mps_repeat_training_cell_blocks_a_different_prepared_device(tmp_path: Path) -> None:
    """Stop the actual repeat cell before launching a mismatched prepared study.

    Args:
        tmp_path (Path): Synthetic experiment identity.
    """
    experiment = SimpleNamespace(
        config_path=tmp_path / "different-mps.yaml", output_dir=tmp_path / "different-output",
        model_settings={"window_features": {"history_epochs": 10}},
        validate_training_start=Mock(side_effect=ValueError("Prepared device cpu differs")),
    )
    namespace = {
        "RUN_TRAINING": True, "logger": Mock(), "experiment": experiment,
        "reference_manifest": {"device": "mps"},
    }
    shell = RecordingNotebookShell(namespace)
    namespace["get_ipython"] = Mock(return_value=shell)
    notebook = json.loads((ROOT / "notebooks/37_fixed_lag_motion_mps.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "if RUN_TRAINING:" in "".join(cell["source"]))

    with pytest.raises(ValueError, match="Prepared device cpu differs"):
        exec(TransformerManager().transform_cell(source), namespace)

    experiment.validate_training_start.assert_called_once_with(expected_device="mps")
    assert not shell.commands
