import os
import shutil
from importlib import import_module
from pathlib import Path
from shlex import quote


class CodexRuntime:
    """Locate the external CLI and check OPAL's isolated login without inference."""

    @staticmethod
    def resolve_binary(configured: str | None = None) -> str:
        """Resolve an explicit binary, PATH entry, or bundled macOS CLI.

        Args:
            configured (str | None): Optional executable path or command name.
        """
        requested = configured or os.environ.get("CODEX_BINARY")
        if requested:
            binary = shutil.which(str(Path(requested).expanduser()))
            if binary:
                return binary
            raise RuntimeError("Configured Codex executable was not found: " + requested)

        binary = shutil.which("codex")
        if binary:
            return binary

        for directory in (Path("/Applications"), Path.home() / "Applications"):
            candidate = directory / "ChatGPT.app/Contents/Resources/codex-cli/bin/codex"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

        raise RuntimeError(
            "Codex CLI was not found. Install @openai/codex with npm or set CODEX_BINARY "
            "to an executable path. uv manages Python packages, not this external CLI."
        )

    @classmethod
    def check(cls, configured: str | None = None) -> dict:
        """Check version and isolated authentication without a model request.

        Args:
            configured (str | None): Optional executable path or command name.
        """
        binary = cls.resolve_binary(configured)
        backend = import_module("opal.backends.codex").CodexBackend(binary=binary)
        backend.preflight()
        logged_in = backend.login_status()
        result = {
            "binary": binary,
            "version": backend.version,
            "codex_home": str(backend.codex_home),
            "logged_in": logged_in,
        }
        if not logged_in:
            command = (
                f"CODEX_HOME={quote(str(backend.codex_home))} {quote(binary)} login --device-auth"
            )
            raise RuntimeError(
                "OPAL's isolated Codex login is missing. Run these commands in Terminal:\n"
                f"mkdir -p {quote(str(backend.codex_home))}\n" + command
            )
        return result
