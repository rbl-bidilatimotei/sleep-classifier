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
    "34_hr_history_horizon.ipynb",
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
    experiment.validate_training_start.assert_called_once_with()


@pytest.mark.parametrize("blocked", [False, True])
def test_seed_training_cell_binds_each_validated_config_and_checks_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool,
) -> None:
    """Use each resolved experiment target and stop blocked runs before any CLI command.

    Args:
        tmp_path (Path): Isolated notebook working directory.
        monkeypatch (pytest.MonkeyPatch): Scoped working-directory replacement.
        blocked (bool): Whether the first readiness check rejects existing results.
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
    notebook = json.loads((ROOT / "notebooks/33_hr_deviation_only_seeds.ipynb").read_text())
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
            experiment.validate_training_start.assert_called_once_with()
    namespace["study"].validate_reused_results.assert_called_once_with()
