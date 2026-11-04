import argparse
import json
import logging
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
import zmq
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from sleep_classifier.logging_utils import configure_logging

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
REPORT_EXECUTION_ENV = "SLEEP_CLASSIFIER_REPORT_ACTIVE"


class NotebookReportRunner:
    def __init__(self, notebook_path: Path, output_dir: Path) -> None:
        self.notebook_path = notebook_path
        self.output_dir = output_dir

    def run(self) -> Path:
        """Execute the source notebook and save a standalone HTML report."""
        if os.environ.get(REPORT_EXECUTION_ENV):
            raise RuntimeError(
                "Recursive notebook report execution is not allowed; "
                "tag export cells with skip-execution"
            )
        notebook = nbformat.read(self.notebook_path, as_version=4)
        self.execute(notebook)
        template = (ROOT / "scripts/templates/accessible.html.j2").read_text(encoding="utf-8")
        report, _ = HTMLExporter(raw_template=template).from_notebook_node(notebook)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"{self.notebook_path.stem}.executed.html"
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def execute(self, notebook: nbformat.NotebookNode) -> None:
        """Execute with a temporary kernel specification for this Python environment."""
        # Short paths keep Unix socket names below the macOS length limit
        directory = "/tmp" if os.name == "posix" else None
        with TemporaryDirectory(prefix="sc-kernel-", dir=directory) as temporary:
            kernel_dir = Path(temporary) / "sleep-classifier"
            kernel_dir.mkdir()
            (kernel_dir / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "Sleep classifier",
                        "language": "python",
                    }
                ),
                encoding="utf-8",
            )
            manager = KernelSpecManager(kernel_dirs=[temporary])
            client = NotebookClient(
                notebook,
                timeout=300,
                kernel_name="sleep-classifier",
                km=KernelManager(
                    kernel_name="sleep-classifier",
                    kernel_spec_manager=manager,
                    connection_file=str(Path(temporary) / "connection.json"),
                    transport="ipc" if zmq.has("ipc") else "tcp",
                ),
                resources={"metadata": {"path": str(ROOT)}},
            )
            environment = {**os.environ, REPORT_EXECUTION_ENV: "1"}
            client.execute(cleanup_kc=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a notebook and export a standalone HTML report."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/01_data_exploration.ipynb"),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/notebooks")
    args = parser.parse_args()
    configure_logging()
    source = (ROOT / args.notebook).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report_path = NotebookReportRunner(source, output_dir.resolve()).run()
    LOGGER.info("Executed notebook report written: path=%s", report_path)


if __name__ == "__main__":
    main()
