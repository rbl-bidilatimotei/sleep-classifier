import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sleep_classifier import bundle_runtime
from sleep_classifier.bundle_runtime import RuntimeIdentity

RUNTIME_IDENTITY = {
    "source_revision": "a" * 40,
    "project_version": "0.1.0",
    "python_requires": ">=3.11,<3.13",
}


def git(root: Path, *arguments: str) -> str:
    """Run Git in an isolated test repository.

    Args:
        root (Path): Temporary repository directory.
        *arguments (str): Git subcommand and arguments.
    """
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a tracked runtime and use its path for source discovery.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Attribute replacement fixture.
    """
    module = tmp_path / "src" / "sleep_classifier" / "bundle_runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("active = True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.13"\n', encoding="utf-8"
    )
    (tmp_path / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    (tmp_path / "uv.lock").touch()
    (tmp_path / "hatch_build.py").touch()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "manifest.json").write_text("{}\n", encoding="utf-8")
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c", "user.name=Test", "-c", "user.email=test@example.test", "-c", "commit.gpgsign=false",
        "commit", "--quiet", "-m", "Add runtime inputs",
    )
    monkeypatch.setattr(bundle_runtime, "__file__", str(module))
    return tmp_path


@pytest.fixture
def installed_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a synthetic identity resource without a Git checkout.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Attribute replacement fixture.
    """
    package = tmp_path / "site-packages" / "sleep_classifier"
    package.mkdir(parents=True)
    identity_path = package / bundle_runtime.IDENTITY_RESOURCE
    identity_path.write_text(json.dumps(RUNTIME_IDENTITY), encoding="utf-8")
    monkeypatch.setattr(bundle_runtime, "__file__", str(package / "bundle_runtime.py"))
    monkeypatch.setattr(bundle_runtime.resources, "files", Mock(return_value=package))
    monkeypatch.setattr(bundle_runtime.metadata, "version", Mock(return_value="0.1.0"))
    monkeypatch.setattr(
        bundle_runtime.subprocess, "run", Mock(side_effect=AssertionError("Git used"))
    )
    return identity_path


def test_checkout_reads_head_with_unrelated_changes(checkout: Path) -> None:
    """Ignore files outside the runtime paths and generated files in them.

    Args:
        checkout (Path): Temporary Git repository with runtime inputs.
    """
    (checkout / "tests").mkdir()
    (checkout / "tests" / "test_example.py").touch()
    (checkout / "docs").mkdir()
    (checkout / "docs" / "guide.md").touch()
    package = checkout / "src" / "sleep_classifier"
    (package / "module.ignored").touch()
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "bundle_runtime.cpython-311.pyc").touch()

    assert RuntimeIdentity.current() == {
        **RUNTIME_IDENTITY,
        "source_revision": git(checkout, "rev-parse", "HEAD"),
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/sleep_classifier/bundle_runtime.py",
        "src/sleep_classifier/added.py",
        "schemas/manifest.json",
        "pyproject.toml",
        "uv.lock",
        "hatch_build.py",
    ],
)
def test_checkout_rejects_changed_runtime_inputs(checkout: Path, relative_path: str) -> None:
    """Reject both tracked and untracked changes within the runtime paths.

    Args:
        checkout (Path): Temporary Git repository with runtime inputs.
        relative_path (str): Runtime path to change.
    """
    (checkout / relative_path).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted changes"):
        RuntimeIdentity.current()


def test_checkout_rejects_staged_runtime_changes(checkout: Path) -> None:
    """Require the index and worktree to match the committed runtime inputs.

    Args:
        checkout (Path): Temporary Git repository with runtime inputs.
    """
    (checkout / "uv.lock").write_text("updated\n", encoding="utf-8")
    git(checkout, "add", "uv.lock")
    with pytest.raises(ValueError, match="uncommitted changes"):
        RuntimeIdentity.current()


def test_source_does_not_use_parent_git_repository(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require a Git checkout at the project root.

    Args:
        checkout (Path): Temporary enclosing Git repository.
        monkeypatch (pytest.MonkeyPatch): Attribute replacement fixture.
    """
    nested = checkout / "nested"
    nested.mkdir()
    (nested / "pyproject.toml").write_bytes((checkout / "pyproject.toml").read_bytes())
    monkeypatch.setattr(
        bundle_runtime, "__file__", str(nested / "src" / "sleep_classifier" / "bundle_runtime.py")
    )
    with pytest.raises(ValueError, match="own Git checkout"):
        RuntimeIdentity.current()


def test_installed_identity_uses_baked_resource(installed_package: Path) -> None:
    """Read a matching installed identity without calling Git.

    Args:
        installed_package (Path): Identity resource in the simulated installation.
    """
    assert installed_package.is_file()
    assert RuntimeIdentity.current() == RUNTIME_IDENTITY
    RuntimeIdentity.verify({**RUNTIME_IDENTITY, "validated_device": "mps"})


@pytest.mark.parametrize("field", RUNTIME_IDENTITY)
def test_verify_rejects_manifest_mismatch(installed_package: Path, field: str) -> None:
    """Reject differences in each identity field.

    Args:
        installed_package (Path): Identity resource in the simulated installation.
        field (str): Manifest runtime field to change.
    """
    with pytest.raises(ValueError, match=field):
        RuntimeIdentity.verify({**RUNTIME_IDENTITY, field: "different"})


def test_installed_identity_checks_distribution_version(
    installed_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require the distribution version to match the baked resource.

    Args:
        installed_package (Path): Identity resource in the simulated installation.
        monkeypatch (pytest.MonkeyPatch): Attribute replacement fixture.
    """
    monkeypatch.setattr(bundle_runtime.metadata, "version", Mock(return_value="9.9"))
    with pytest.raises(ValueError, match="installed version"):
        RuntimeIdentity.current()


def test_installed_identity_requires_baked_resource(installed_package: Path) -> None:
    """Reject a wheel installation without a runtime identity.

    Args:
        installed_package (Path): Identity resource in the simulated installation.
    """
    installed_package.unlink()
    with pytest.raises(ValueError, match="Cannot read the installed bundle runtime identity"):
        RuntimeIdentity.current()


@pytest.mark.parametrize("version", [(3, 10, 14), (3, 13, 0)])
def test_verify_requires_supported_python(
    installed_package: Path, monkeypatch: pytest.MonkeyPatch, version: tuple[int, int, int]
) -> None:
    """Reject a matching manifest under an incompatible Python version.

    Args:
        installed_package (Path): Identity resource in the simulated installation.
        monkeypatch (pytest.MonkeyPatch): Attribute replacement fixture.
        version (tuple[int, int, int]): Simulated running Python version.
    """
    monkeypatch.setattr(bundle_runtime, "sys", SimpleNamespace(version_info=version))
    with pytest.raises(ValueError, match="requires Python"):
        RuntimeIdentity.verify(RUNTIME_IDENTITY)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_worktree_edits_cannot_reuse_the_committed_revision(
    checkout: Path, flag: str,
) -> None:
    """Bind the declared revision to actual source bytes rather than cached index status.

    Args:
        checkout (Path): Temporary repository with tracked runtime files.
        flag (str): Git index flag that hides worktree edits from ordinary status checks.
    """
    name = "src/sleep_classifier/bundle_runtime.py"
    git(checkout, "update-index", flag, name)
    (checkout / name).write_text("active = False\n")
    assert git(checkout, "status", "--porcelain", "--", name) == ""
    with pytest.raises(ValueError, match="uncommitted changes"):
        RuntimeIdentity.current()
