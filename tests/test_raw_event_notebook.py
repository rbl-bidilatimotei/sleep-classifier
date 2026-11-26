import json
import shlex
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("IPython")

from IPython.core.inputtransformer2 import TransformerManager  # noqa: E402
from test_notebook_training_identity import RecordingNotebookShell  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/46_raw_event_adapter.ipynb"


@pytest.fixture
def notebook_cells() -> list[dict[str, Any]]:
    """Read the checked-in raw-event audit notebook cells."""
    return json.loads(NOTEBOOK.read_text())["cells"]


@pytest.mark.parametrize(
    "enabled,cached_status",
    [(False, None), (True, None), (False, "passed"), (True, "passed"), (True, "partial")],
)
def test_audit_cell_validates_fresh_runs_and_preserves_existing_output(
    notebook_cells: list[dict[str, Any]],
    tmp_path: Path,
    enabled: bool,
    cached_status: str | None,
) -> None:
    """Exercise the actual audit and report cells with a recording shell.

    Args:
        notebook_cells (list[dict[str, Any]]): Cells from the source notebook.
        tmp_path (Path): Isolated checkout with a different active configuration path.
        enabled (bool): Explicit audit opt-in flag.
        cached_status (str | None): Status of any retained report.
    """
    output = tmp_path / "retained"
    cached_report = output / "report.json"
    retained_bytes = b"retained report"
    if cached_status is not None:
        output.mkdir()
        cached_report.write_bytes(retained_bytes)

    report = {
        "status": cached_status or "pending",
        "training_performed": False,
        "test_observations_read": False,
    }
    audit = Mock(
        config_path=tmp_path / "another audit.yaml",
        output_dir=output,
        read_report=Mock(return_value=report),
    )
    namespace = {
        "RUN_AUDIT": enabled, "audit": audit, "ROOT": tmp_path, "logger": Mock(),
        "display": Mock(), "pd": Mock(),
    }
    shell = RecordingNotebookShell(namespace)
    actions = Mock()
    actions.attach_mock(audit.validate_run_start, "validate_run_start")
    actions.attach_mock(Mock(wraps=shell.system), "system")
    actions.attach_mock(audit.read_report, "read_report")
    shell.system = actions.system
    namespace["get_ipython"] = Mock(return_value=shell)
    code_cells = ["".join(cell["source"]) for cell in notebook_cells if cell["cell_type"] == "code"]
    transform = TransformerManager().transform_cell

    audit_cell = next(source for source in code_cells if source.startswith("if RUN_AUDIT"))
    load_report = next(source for source in code_cells if "report = audit.read_report()" in source)
    exec(transform(audit_cell), namespace)
    exec(transform(load_report), namespace)

    should_run = enabled and cached_status is None
    expected_actions = ["validate_run_start", "system", "read_report"] if should_run else [
        "read_report",
    ]
    assert [action[0] for action in actions.mock_calls] == expected_actions
    assert len(shell.commands) == int(should_run)
    assert namespace["report"] is report
    assert namespace["report_ready"] is (cached_status == "passed")

    if should_run:
        assert shlex.split(shell.commands[0]) == [
            "uv", "run", "--locked", "--inexact", "--group", "gru", "--group", "notebook",
            "python", "-u", "-m", "sleep_classifier.raw_event_audit",
            "--config", str(audit.config_path), "--run",
        ]
    if cached_status is not None:
        assert cached_report.read_bytes() == retained_bytes
    else:
        assert not output.exists()


def test_completed_report_renders_all_tables_and_figure(
    notebook_cells: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render the actual report cells with the published schema and synthetic data.

    Args:
        notebook_cells (list[dict[str, Any]]): Cells from the source notebook.
        tmp_path (Path): Isolated checkout path passed to the audit constructor.
        monkeypatch (pytest.MonkeyPatch): Scoped audit and display replacements.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    show = Mock()
    monkeypatch.setattr(plt, "show", show)
    plan = {
        "participants": ["100001", "100002"], "chunk_sizes": [128, 1024],
        "max_buffered_events": 8192, "device": "mps",
    }
    report = {
        "status": "passed", "plan": plan, "training_performed": False,
        "test_observations_read": False,
        "participants": [
            {
                "participant_id": person, "chunk_size": chunk,
                "epochs": 20, "eligible_epochs": 19,
                "feature_maximum_difference": 0.0, "feature_mismatches": 0,
                "historical_feature_maximum_difference": 1e-9, "historical_feature_mismatches": 0,
                "eligibility_matches": True, "segments_match": True,
                "event_conservation_matches": True, "peak_buffered_events": 1500 + chunk,
                "duplicate_events": 3, "discarded_tail_events": 26,
                "filtered_pre_session_events": 7, "filtered_after_session_events": 33,
                "elapsed_seconds": 0.001 * chunk,
            }
            for person in plan["participants"] for chunk in plan["chunk_sizes"]
        ],
        "prediction_parity": [{
            "fold": "fold-01", "epochs": 38, "maximum_probability_difference": 1e-7,
            "class_disagreements": 0, "history_lengths_match": True, "reset_reasons_match": True,
            "weights_unchanged": True, "saved_transforms_unchanged": True,
            "gradients_created": False, "evaluation_mode": True,
            "training_performed": False, "checkpoint_sha256": "a" * 64,
        }],
    }
    config = tmp_path / "configs/raw-event-adapter.yaml"
    audit = SimpleNamespace(
        config_path=config, output_dir=tmp_path / "output",
        plan=Mock(return_value=plan), read_report=Mock(return_value=report),
    )
    constructor = Mock(return_value=audit)
    monkeypatch.setitem(
        sys.modules, "sleep_classifier.raw_event_audit", SimpleNamespace(RawEventAudit=constructor),
    )
    display = Mock()
    namespace = {"ROOT": tmp_path, "logger": Mock(), "display": display}
    report_cells = [
        "".join(cell["source"]) for cell in notebook_cells
        if cell["cell_type"] == "code" and not cell["metadata"].get("tags")
    ]

    try:
        for source in report_cells[1:]:
            exec(source, namespace)

        constructor.assert_called_once_with(config)
        show.assert_called_once_with()
        scope, access, participants, predictions = [call.args[0] for call in display.call_args_list]
        assert scope.at[0, "development_participants"] == 2
        assert access.iloc[0].to_dict() == {
            "training_performed": False, "test_observations_read": False,
        }
        assert len(participants) == 4
        assert participants["event_conservation_matches"].all()
        assert participants["historical_feature_maximum_difference"].max() == 1e-9
        assert participants["historical_feature_mismatches"].sum() == 0
        assert predictions.at["fold-01", "checkpoint_sha256"] == "a" * 64

        with BytesIO() as rendered:
            namespace["fig"].savefig(rendered, format="png", backend="agg")
            assert rendered.getvalue().startswith(b"\x89PNG")
    finally:
        if "fig" in namespace:
            plt.close(namespace["fig"])


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
        if any(marker in source for marker in ("if RUN_SETUP", "if RUN_AUDIT",
                                                "execute_notebook.py")):
            assert "skip-execution" in cell["metadata"]["tags"]

    flag_code = next(source for source in code_cells if "RUN_SETUP =" in source)
    assert "RUN_SETUP = False" in flag_code and "RUN_AUDIT = False" in flag_code
