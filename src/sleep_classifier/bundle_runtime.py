from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from importlib import metadata, resources
from pathlib import Path
from typing import Any

PYTHON_REQUIRES = ">=3.11,<3.13"
IDENTITY_RESOURCE = "_bundle_runtime.json"
IDENTITY_FIELDS = ("source_revision", "project_version", "python_requires")
RUNTIME_PATHS = ("src/sleep_classifier", "pyproject.toml", "uv.lock", "schemas", "hatch_build.py")


class RuntimeIdentity:
    """Identify the runtime source used for inference bundles."""

    @classmethod
    def current(cls) -> dict[str, str]:
        """Read the identity from the current checkout or installed wheel."""
        package_path = Path(__file__).resolve().parent
        root = package_path.parent.parent
        if package_path == root / "src" / "sleep_classifier" and (
            (root / ".git").exists() or (root / "pyproject.toml").is_file()
        ):
            return cls.from_checkout(root)
        return cls._from_installation()

    @classmethod
    def verify(cls, record: dict[str, Any]) -> None:
        """Require the manifest identity to match the current runtime.

        Args:
            record (dict[str, Any]): Runtime fields from the bundle manifest.
        """
        identity = cls.current()
        for name in IDENTITY_FIELDS:
            if record.get(name) != identity[name]:
                raise ValueError(f"Bundle runtime {name} does not match the current runtime")

    @classmethod
    def from_checkout(cls, root: Path) -> dict[str, str]:
        """Read the identity of a clean runtime in the given Git checkout.

        Args:
            root (Path): Repository directory that contains the runtime inputs.
        """
        root = root.resolve()
        if not (root / ".git").exists() or Path(
            cls._git(root, "rev-parse", "--show-toplevel")
        ) != root:
            raise ValueError("Bundle runtime source must be in its own Git checkout")

        status = cls._git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--no-renames",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            *RUNTIME_PATHS,
        )
        for entry in status.split("\0"):
            if entry.startswith("?? ") and "__pycache__" in Path(entry[3:]).parts:
                continue
            if entry:
                raise ValueError(f"Bundle runtime source has uncommitted changes: {entry[3:]}")

        cls._verify_tracked_bytes(root)
        try:
            with (root / "pyproject.toml").open("rb") as stream:
                project = tomllib.load(stream)["project"]
            identity = {
                "source_revision": cls._git(root, "rev-parse", "--verify", "HEAD"),
                "project_version": project["version"],
                "python_requires": project["requires-python"],
            }
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
            raise ValueError("Cannot read bundle runtime project metadata") from error
        return cls._validate(identity)

    @classmethod
    def _verify_tracked_bytes(cls, root: Path) -> None:
        """Compare runtime bytes with Git objects even when index flags hide local changes.

        Args:
            root (Path): Source checkout already checked for staged and untracked changes.
        """
        tree = cls._git(root, "ls-tree", "-r", "-z", "HEAD", "--", *RUNTIME_PATHS)
        for entry in tree.split("\0"):
            if not entry:
                continue
            metadata_record, name = entry.split("\t", 1)
            mode, kind, expected = metadata_record.split(" ")
            path = root / name
            if mode not in ("100644", "100755") or kind != "blob" or path.is_symlink():
                raise ValueError("Runtime identity requires regular tracked source files")
            data = path.read_bytes()
            digest = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False,
            ).hexdigest()
            if digest != expected:
                raise ValueError(f"Bundle runtime source has uncommitted changes: {name}")

    @classmethod
    def _from_installation(cls) -> dict[str, str]:
        """Read the baked identity and compare the installed distribution version."""
        try:
            resource = resources.files("sleep_classifier").joinpath(IDENTITY_RESOURCE)
            identity = cls._validate(json.loads(resource.read_text(encoding="utf-8")))
            installed_version = metadata.version("sleep-classifier")
        except (
            OSError, UnicodeError, metadata.PackageNotFoundError, json.JSONDecodeError
        ) as error:
            raise ValueError("Cannot read the installed bundle runtime identity") from error
        if identity["project_version"] != installed_version:
            raise ValueError("Bundle runtime project_version does not match the installed version")
        return identity

    @staticmethod
    def _validate(identity: Any) -> dict[str, str]:
        """Check the identity fields and the running Python version.

        Args:
            identity (Any): Decoded checkout or installed runtime identity.
        """
        if not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS):
            raise ValueError("Invalid bundle runtime identity fields")
        if not all(isinstance(value, str) and value for value in identity.values()):
            raise ValueError("Bundle runtime identity fields must be nonempty strings")
        if re.fullmatch(r"[0-9a-f]{40}", identity["source_revision"]) is None:
            raise ValueError("Invalid bundle runtime source_revision")
        python_version = sys.version_info[:2]
        if identity["python_requires"] != PYTHON_REQUIRES or not (
            (3, 11) <= python_version < (3, 13)
        ):
            raise ValueError(f"Bundle runtime requires Python {PYTHON_REQUIRES}")
        return identity

    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        """Run a Git command against the given checkout.

        Args:
            root (Path): Expected Git repository directory.
            *arguments (str): Git subcommand and arguments.
        """
        try:
            result = subprocess.run(
                [
                    "git",
                    "--no-optional-locks",
                    "-c", "core.fsmonitor=false",
                    "-C", str(root),
                    *arguments,
                ],
                capture_output=True,
                check=True,
                encoding="utf-8",
                errors="surrogateescape",
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("Cannot determine bundle runtime Git identity") from error
        return result.stdout.removesuffix("\n")
