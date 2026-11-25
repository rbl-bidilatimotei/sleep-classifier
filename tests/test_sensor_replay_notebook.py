import json
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("IPython")

from IPython.core.inputtransformer2 import TransformerManager  # noqa: E402
from test_notebook_training_identity import RecordingNotebookShell  # noqa: E402

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/45_sensor_only_frozen_replay.ipynb"
)


@pytest.fixture
def notebook_cells() -> list[dict[str, Any]]:
    """Read the checked-in frozen replay notebook cells."""
    return json.loads(NOTEBOOK.read_text())["cells"]


@pytest.mark.parametrize("enabled,cached", [(False, False), (False, True),
                                            (True, False), (True, True)])
def test_replay_cell_checks_readiness_and_keeps_existing_output(
    notebook_cells: list[dict[str, Any]], tmp_path: Path, enabled: bool, cached: bool,
) -> None:
    """Exercise replay and report loading without launching inference.

    Args:
        notebook_cells (list[dict[str, Any]]): Cells from the source notebook.
        tmp_path (Path): Isolated checkout with a different active configuration path.
        enabled (bool): Explicit replay opt-in flag.
        cached (bool): Whether a retained report already exists.
    """
    output = tmp_path / "retained"
    cached_report = output / "report.json"
    retained_bytes = b"retained report"
    if cached:
        output.mkdir()
        cached_report.write_bytes(retained_bytes)

    report = {"status": "passed" if cached else "pending", "folds": []}
    study = Mock(
        config_path=tmp_path / "another replay.yaml", output_dir=output,
        read_report=Mock(return_value=report),
    )
    namespace = {"RUN_REPLAY": enabled, "study": study, "ROOT": tmp_path, "logger": Mock()}
    shell = RecordingNotebookShell(namespace)
    actions = Mock()
    actions.attach_mock(study.validate_run_start, "validate_run_start")
    actions.attach_mock(Mock(wraps=shell.system), "system")
    actions.attach_mock(study.read_report, "read_report")
    shell.system = actions.system
    namespace["get_ipython"] = Mock(return_value=shell)
    code_cells = ["".join(cell["source"]) for cell in notebook_cells if cell["cell_type"] == "code"]
    transform = TransformerManager().transform_cell

    replay = next(source for source in code_cells if source.startswith("if RUN_REPLAY"))
    load_report = next(source for source in code_cells if "report = study.read_report()" in source)
    exec(transform(replay), namespace)
    exec(transform(load_report), namespace)

    should_run = enabled and not cached
    expected_actions = ["validate_run_start", "system", "read_report"] if should_run else [
        "read_report",
    ]
    assert [action[0] for action in actions.mock_calls] == expected_actions
    assert len(shell.commands) == int(should_run)
    assert namespace["report"] is report
    assert namespace["report_ready"] is cached

    if should_run:
        command = shlex.split(shell.commands[0])
        assert command[command.index("--config") + 1] == str(study.config_path)
        assert command[command.index("-m") + 1] == "sleep_classifier.sensor_only_replay"
        assert "--run" in command and "--train" not in command
    if cached:
        assert cached_report.read_bytes() == retained_bytes
    else:
        assert not output.exists()


def test_notebook_explains_code_and_keeps_execution_opt_in(
    notebook_cells: list[dict[str, Any]],
) -> None:
    """Require explanatory cells, empty outputs, and opt-in execution.

    Args:
        notebook_cells (list[dict[str, Any]]): Cells from the source notebook.
    """
    code_cells = []
    for index, cell in enumerate(notebook_cells):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        assert notebook_cells[index - 1]["cell_type"] == "markdown"
        explanation = "".join(notebook_cells[index - 1]["source"])
        assert "**What:**" in explanation and "**Why:**" in explanation
        source = "".join(cell["source"])
        code_cells.append(source)
        if any(marker in source for marker in ("if RUN_SETUP", "if RUN_REPLAY",
                                                "execute_notebook.py")):
            assert "skip-execution" in cell["metadata"]["tags"]

    flag_code = next(source for source in code_cells if "RUN_SETUP =" in source)
    assert "RUN_SETUP = False" in flag_code and "RUN_REPLAY = False" in flag_code
