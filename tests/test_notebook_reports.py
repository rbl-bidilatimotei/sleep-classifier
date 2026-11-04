import importlib.util
import os
import sys
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


def test_report_runner_skips_tagged_training_cells(tmp_path):
    notebook_path = tmp_path / "training_guard.ipynb"
    training_cell = nbformat.v4.new_code_cell("raise RuntimeError('training must not run')")
    training_cell.metadata["tags"] = ["skip-execution"]
    notebook = nbformat.v4.new_notebook(
        cells=[
            training_cell,
            nbformat.v4.new_code_cell("print('report completed')"),
        ]
    )
    nbformat.write(notebook, notebook_path)

    report_path = NotebookReportRunner(notebook_path, tmp_path / "reports").run()

    assert "report completed" in report_path.read_text()


def test_baseline_notebook_guards_training_and_its_export_command():
    source = Path(__file__).resolve().parents[1] / "notebooks/05_baseline_experiment.ipynb"
    notebook = nbformat.read(source, as_version=4)
    configuration = next(cell for cell in notebook.cells if "RUN_TRAINING =" in cell.source)
    assert "RUN_TRAINING = False" in configuration.source

    guarded_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and ("--train" in cell.source or "scripts/execute_notebook.py" in cell.source)
    ]
    assert len(guarded_cells) == 2
    assert all("skip-execution" in cell.metadata.get("tags", []) for cell in guarded_cells)


def test_gru_notebook_keeps_training_operator_controlled():
    source = Path(__file__).resolve().parents[1] / "notebooks/07_causal_gru.ipynb"
    notebook = nbformat.read(source, as_version=4)
    configuration = next(cell for cell in notebook.cells if "RUN_TRAINING =" in cell.source)
    assert "RUN_TRAINING = False" in configuration.source
    training = [
        cell for cell in notebook.cells if cell.cell_type == "code" and "--train" in cell.source
    ]
    assert len(training) == 1
    assert "skip-execution" in training[0].metadata.get("tags", [])


def test_report_includes_descriptive_alt_text_and_local_ipc(tmp_path, caplog):
    notebook_path = tmp_path / "accessible_report.ipynb"
    cell = nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "from IPython import get_ipython\n"
        "import zmq\n"
        "if zmq.has('ipc'):\n"
        "    assert get_ipython().kernel.parent.transport == 'ipc'\n"
        "plt.plot([1, 2], [0.8, 0.5])\n"
        "plt.show()"
    )
    description = 'Training loss decreases from 0.8 to 0.5; label contains "quotes" & symbols'
    cell.metadata["alt"] = description
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), notebook_path)

    report_path = NotebookReportRunner(notebook_path, tmp_path / "reports").run()

    from bs4 import BeautifulSoup

    report = BeautifulSoup(report_path.read_text(), "html.parser")
    assert [image.get("alt") for image in report.select("img")] == [description]
    assert "Alternative text is missing" not in caplog.text
    assert not list(report_path.parent.glob("*.ipynb"))


@pytest.mark.parametrize(
    "source",
    sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")),
    ids=lambda source: source.name,
)
def test_shipped_notebooks_tag_training_and_report_export_cells(source):
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if "--train" in cell.source or "scripts/execute_notebook.py" in cell.source:
            assert "skip-execution" in cell.metadata.get("tags", []), source.name


def test_report_rejects_recursive_export_without_changing_parent_environment(tmp_path):
    notebook_path = tmp_path / "recursive_export.ipynb"
    output_dir = tmp_path / "nested_reports"
    script = Path(__file__).resolve().parents[1] / "scripts/execute_notebook.py"
    command = [
        sys.executable,
        str(script),
        "--notebook",
        str(notebook_path),
        "--output-dir",
        str(output_dir),
    ]
    cell = nbformat.v4.new_code_cell(
        "import subprocess\n"
        f"result = subprocess.run({command!r}, capture_output=True, text=True, timeout=30)\n"
        "assert result.returncode != 0\n"
        "assert 'Recursive notebook report execution is not allowed' in result.stderr\n"
        "print('Nested exporter rejected')"
    )
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), notebook_path)
    original_environment = os.environ.get(module.REPORT_EXECUTION_ENV)

    report = NotebookReportRunner(notebook_path, tmp_path / "reports").run()

    assert "Nested exporter rejected" in report.read_text()
    assert not output_dir.exists()
    assert os.environ.get(module.REPORT_EXECUTION_ENV) == original_environment
