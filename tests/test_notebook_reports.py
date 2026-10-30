import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

spec = importlib.util.spec_from_file_location(
    "notebook_report_runner", Path(__file__).resolve().parents[1] / "scripts/execute_notebook.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
NotebookReportRunner = module.NotebookReportRunner


def test_runner_executes_notebook_and_writes_html_report(tmp_path):
    notebook_path = tmp_path / "report_sample.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('rendered notebook output')")]
    )
    nbformat.write(notebook, notebook_path)

    original_source = notebook_path.read_bytes()
    report_path = NotebookReportRunner(notebook_path, tmp_path / "reports").run()

    report = report_path.read_text(encoding="utf-8")
    assert report_path.name == "report_sample.executed.html"
    assert "rendered notebook output" in report
    assert "<html" in report
    assert notebook_path.read_bytes() == original_source
    assert not list(report_path.parent.glob("*.ipynb"))
