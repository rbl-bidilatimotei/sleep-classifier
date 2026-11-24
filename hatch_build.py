from __future__ import annotations

import json
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Embed the runtime identity in a wheel built from a clean checkout."""

    _identity_directory: TemporaryDirectory[str] | None = None

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Add build-generated runtime metadata to a standard wheel.

        Args:
            version (str): Hatch build mode.
            build_data (dict[str, Any]): Hatch options to update for the build.
        """
        if version == "editable":
            return

        root = Path(self.root)
        package = Path("sleep_classifier")
        runtime = runpy.run_path(str(root / "src" / package / "bundle_runtime.py"))
        identity = runtime["RuntimeIdentity"].from_checkout(root)
        filename = runtime["IDENTITY_RESOURCE"]

        self._identity_directory = TemporaryDirectory()
        identity_path = Path(self._identity_directory.name) / filename
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
        build_data["force_include"][str(identity_path)] = (package / filename).as_posix()

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """Remove the generated runtime metadata after the build.

        Args:
            version (str): Hatch build mode.
            build_data (dict[str, Any]): Hatch options used for the build.
            artifact_path (str): Path to the completed wheel.
        """
        if self._identity_directory is not None:
            self._identity_directory.cleanup()
            self._identity_directory = None
