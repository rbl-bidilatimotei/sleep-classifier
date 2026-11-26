import base64
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "wheel_package_check", ROOT / "scripts/check_wheel.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
WheelPackageCheck = module.WheelPackageCheck
REVISION = "a" * 40
SCHEMAS = (
    "bundle_schemas/v1/manifest.schema.json",
    "bundle_schemas/v1/epoch-request.schema.json",
    "bundle_schemas/v1/epoch-response.schema.json",
    "session_schemas/v1/event.schema.json",
    "session_schemas/v1/output.schema.json",
)


def make_wheel(
    tmp_path: Path, additions: dict[str, bytes] | None = None,
    removals: tuple[str, ...] = (), identity_changes: dict[str, Any] | None = None,
    stale_record: bool = False,
) -> tuple[Path, Path]:
    """Build a tiny valid wheel without project dependencies or model observations.

    Args:
        tmp_path (Path): Temporary fixture directory.
        additions (dict[str, bytes] | None): Extra or replacement wheel files.
        removals (tuple[str, ...]): Wheel entries to omit.
        identity_changes (dict[str, Any] | None): Changes to the embedded runtime identity.
        stale_record (bool): Whether to corrupt a file after its RECORD digest is written.
    """
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.13"\n')
    identity = {"source_revision": REVISION, "project_version": "0.1.0",
                "python_requires": ">=3.11,<3.13"} | (identity_changes or {})
    entries = {
        "sleep_classifier/_bundle_runtime.json": json.dumps(identity).encode(),
        "sleep_classifier/__init__.py": b"",
        "sleep_classifier-0.1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: sleep-classifier\nVersion: 0.1.0\n"
            b"Requires-Python: <3.13,>=3.11\n"
        ),
        "sleep_classifier-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        **{f"sleep_classifier/{name}": json.dumps({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        }).encode() for name in SCHEMAS},
        **(additions or {}),
    }
    for name in removals:
        entries.pop(name)
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, data in entries.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow(("sleep_classifier-0.1.0.dist-info/RECORD", "", ""))
    entries["sleep_classifier-0.1.0.dist-info/RECORD"] = record.getvalue().encode()
    if stale_record:
        entries["sleep_classifier/__init__.py"] = b"changed = True\n"
    wheel = tmp_path / "sleep_classifier-0.1.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return wheel, project


def test_wheel_check_accepts_complete_identity_and_resources(tmp_path: Path) -> None:
    """Accept a self-contained synthetic package without importing model code.

    Args:
        tmp_path (Path): Temporary fixture directory.
    """
    wheel, project = make_wheel(tmp_path)
    report = WheelPackageCheck(wheel, REVISION, project).run()
    assert report["status"] == "passed" and report["packaged_schemas"] == 5
    assert report["dataset_payloads"] is False and report["model_payloads"] is False


@pytest.mark.parametrize("changes", [
    {"source_revision": "b" * 40}, {"project_version": "0.2.0"},
    {"python_requires": ">=3.14"}, {"extra": "unsupported"},
])
def test_wheel_check_rejects_wrong_runtime_identity(
    tmp_path: Path, changes: dict[str, Any],
) -> None:
    """Reject a wheel that cannot represent the expected runtime revision.

    Args:
        tmp_path (Path): Temporary fixture directory.
        changes (dict[str, Any]): Invalid identity changes.
    """
    wheel, project = make_wheel(tmp_path, identity_changes=changes)
    with pytest.raises(ValueError, match="identity"):
        WheelPackageCheck(wheel, REVISION, project).run()


@pytest.mark.parametrize("path", [
    "data/heart_rate.txt", "artifacts/checkpoint.pt", "notebooks/experiment.ipynb",
    ".env", "../outside.py", "/absolute.py", "sleep_classifier/model.pt",
    "sleep_classifier/.env", "sleep_classifier-0.1.0.dist-info/credentials.json",
])
def test_wheel_check_rejects_payloads_outside_runtime_contract(
    tmp_path: Path, path: str,
) -> None:
    """Keep datasets, models, credentials, and unsafe paths outside the runtime wheel.

    Args:
        tmp_path (Path): Temporary fixture directory.
        path (str): Unexpected archive payload name.
    """
    wheel, project = make_wheel(tmp_path, additions={path: b"unexpected"})
    with pytest.raises(ValueError):
        WheelPackageCheck(wheel, REVISION, project).run()


@pytest.mark.parametrize("missing", [
    "sleep_classifier/_bundle_runtime.json",
    "sleep_classifier/session_schemas/v1/event.schema.json",
    "sleep_classifier/bundle_schemas/v1/epoch-request.schema.json",
])
def test_wheel_check_rejects_missing_resources(tmp_path: Path, missing: str) -> None:
    """Reject wheels missing their runtime identity or packaged protocol schemas.

    Args:
        tmp_path (Path): Temporary fixture directory.
        missing (str): Required archive entry to omit.
    """
    wheel, project = make_wheel(tmp_path, removals=(missing,))
    with pytest.raises(ValueError):
        WheelPackageCheck(wheel, REVISION, project).run()


def test_wheel_check_rejects_stale_record_digest(tmp_path: Path) -> None:
    """Detect package-file changes that no longer match the wheel's recorded digest.

    Args:
        tmp_path (Path): Temporary fixture directory.
    """
    wheel, project = make_wheel(tmp_path, stale_record=True)
    with pytest.raises(ValueError, match="RECORD digest"):
        WheelPackageCheck(wheel, REVISION, project).run()


def test_workflow_uses_one_bounded_cpu_job_and_explicit_synthetic_tests() -> None:
    """Keep automatic CI free of training, datasets, matrices, and duplicate branch pushes."""
    # BaseLoader preserves the GitHub Actions `on` key under YAML 1.1 parsers
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert "pull_request" in workflow["on"] and "workflow_dispatch" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert len(workflow["jobs"]) == 1
    job = workflow["jobs"]["synthetic-checks"]
    assert int(job["timeout-minutes"]) <= 15 and "strategy" not in job
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    tests = next(step["run"] for step in job["steps"]
                 if step.get("name") == "Run only the synthetic contract tests")
    selected = {part for part in tests.split() if part.startswith("tests/")}
    assert selected == {
        "tests/test_bundle_loader.py", "tests/test_bundle_runtime.py",
        "tests/test_session_protocol.py", "tests/test_raw_session_runner.py",
        "tests/test_raw_session_cli.py", "tests/test_http_service.py",
        "tests/test_service_sessions.py", "tests/test_ci_contract.py",
    }
    assert "test ! -e data" in tests and "test ! -e artifacts" in tests
    docker_steps = [step for step in job["steps"] if "docker " in step.get("run", "")]
    assert len(docker_steps) == 2
    assert all("workflow_dispatch" in step["if"] and "inputs.check_docker" in step["if"]
               for step in docker_steps)


def test_wheel_checker_cli_uses_only_standard_library(tmp_path: Path) -> None:
    """Run the checker in isolated Python without importing the installed project.

    Args:
        tmp_path (Path): Temporary fixture directory.
    """
    wheel, project = make_wheel(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/check_wheel.py"), "--wheel", str(wheel),
         "--expected-revision", REVISION, "--project", str(project)],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout)["status"] == "passed"
