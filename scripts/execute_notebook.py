import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


class NotebookReportRunner:
    def __init__(self, notebook_path: Path, output_dir: Path) -> None:
        self.notebook_path = notebook_path
        self.output_dir = output_dir

    def run(self) -> Path:
        """Execute the source notebook and save a standalone HTML report."""
        notebook = nbformat.read(self.notebook_path, as_version=4)
        self.execute(notebook)
        report, _ = HTMLExporter().from_notebook_node(notebook)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"{self.notebook_path.stem}.executed.html"
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def execute(self, notebook: nbformat.NotebookNode) -> None:
        """Execute with a temporary kernel specification for this Python environment."""
        with TemporaryDirectory(prefix="sleep-classifier-kernel-") as temporary:
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
                km=KernelManager(kernel_name="sleep-classifier", kernel_spec_manager=manager),
                resources={"metadata": {"path": str(ROOT)}},
            )
            client.execute(cleanup_kc=True)


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
    source = (ROOT / args.notebook).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report_path = NotebookReportRunner(source, output_dir.resolve()).run()
    print(report_path)


if __name__ == "__main__":
    main()
