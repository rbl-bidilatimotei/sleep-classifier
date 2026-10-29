"""Execute the exploration notebook with the active Python environment."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / "notebooks/01_data_exploration.ipynb", as_version=4)
with TemporaryDirectory() as temporary:
    kernel_dir = Path(temporary) / "sleep-classifier"
    kernel_dir.mkdir()

    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Sleep classifier",
                "language": "python",
            }
        )
    )
    manager = KernelSpecManager(kernel_dirs=[temporary])
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="sleep-classifier",
        km=KernelManager(kernel_name="sleep-classifier", kernel_spec_manager=manager),
        resources={"metadata": {"path": str(root)}},
        # Explicit kernel spec prevents accidentally using a global Python kernel.
    ).execute(cleanup_kc=True)
output = root / "artifacts/notebooks/01_data_exploration.executed.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, output)
print(output)
