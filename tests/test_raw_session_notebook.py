import ast
from pathlib import Path

import nbformat

from sleep_classifier.bundle_export import EvaluationBundleExporter
from sleep_classifier.raw_session_cli import SessionCLIConfig

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_defaults_preserve_operator_control() -> None:
    """Keep setup, publication, inference, and recursive export behind explicit gates."""
    notebook = nbformat.read(ROOT / "notebooks/47_raw_session_inference.ipynb", as_version=4)
    first_code = next(cell for cell in notebook.cells if cell.cell_type == "code")
    flags = {node.targets[0].id: ast.literal_eval(node.value)
             for node in ast.walk(ast.parse(first_code.source))
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id.startswith("RUN_")}
    assert flags == {"RUN_SETUP": False, "RUN_EXPORT": False, "RUN_DEMO": False}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert not cell.outputs and cell.execution_count is None
        previous = notebook.cells[index-1]
        assert previous.cell_type == "markdown"
        assert "**What:**" in previous.source and "**Why:**" in previous.source
        if any(line.lstrip().startswith("!") for line in cell.source.splitlines()):
            assert "skip-execution" in cell.metadata.get("tags", [])
            assert '_exit_code' in cell.source


def test_notebook_bundle_configs_share_a_fresh_identity() -> None:
    """Use one new bundle destination for export and session loading."""
    exporter = EvaluationBundleExporter(ROOT / "configs/raw-session-bundle.yaml")
    session = SessionCLIConfig.load(ROOT / "configs/raw-session.yaml")
    assert exporter.output_dir == session.bundle_dir.resolve()
    assert exporter.settings.bundle_id == "motion-seed42-fold01-raw-session-v1"
    assert exporter.settings.fold == "fold-01"
