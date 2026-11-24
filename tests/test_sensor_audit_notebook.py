import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("IPython")

from IPython.core.inputtransformer2 import TransformerManager  # noqa: E402
from test_notebook_training_identity import RecordingNotebookShell  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("enabled, cached", [(False, False), (False, True),
                                            (True, False), (True, True)])
def test_audit_cell_binds_config_and_reuses_cache(
    tmp_path: Path, enabled: bool, cached: bool,
) -> None:
    """Exercise the actual notebook gate without launching a process.

    Args:
        tmp_path (Path): Isolated directory for a different active audit configuration.
        enabled (bool): Explicit raw-audit opt-in switch.
        cached (bool): Whether a retained snapshot directory exists.
    """
    output = tmp_path / "retained"
    if cached:
        output.mkdir()
    audit = SimpleNamespace(config_path=tmp_path / "another audit.yaml", output_dir=output)
    namespace = {"RUN_AUDIT": enabled, "audit": audit, "ROOT": tmp_path, "logger": Mock()}
    shell = RecordingNotebookShell(namespace)
    namespace["get_ipython"] = Mock(return_value=shell)
    notebook = json.loads((ROOT / "notebooks/44_label_free_sensor_audit.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                and "if RUN_AUDIT" in "".join(cell["source"]))
    assert "skip-execution" in cell["metadata"]["tags"]
    exec(TransformerManager().transform_cell("".join(cell["source"])), namespace)

    assert len(shell.commands) == int(enabled and not cached)
    if shell.commands:
        command = shlex.split(shell.commands[0])
        assert command[command.index("--config") + 1] == str(audit.config_path)
        assert "sleep_classifier.sensor_preparation_audit" in command
        assert "--run" in command and "--train" not in command


def test_all_report_cells_explain_purpose_and_keep_source_outputs_empty() -> None:
    """Keep the notebook explanatory and prevent source-output or recursive report execution."""
    notebook = json.loads((ROOT / "notebooks/44_label_free_sensor_audit.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None and cell["outputs"] == []
        explanation = "".join(notebook["cells"][index - 1]["source"])
        assert "**What:**" in explanation and "**Why:**" in explanation
        code = "".join(cell["source"])
        if "execute_notebook.py" in code or "if RUN_SETUP" in code:
            assert "skip-execution" in cell["metadata"]["tags"]
