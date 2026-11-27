from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ServiceConfig:
    """Fixed bundle location and bounded local HTTP session settings."""

    bundle_dir: Path | None = None
    max_buffered_events: int = 8192
    max_epochs_per_event: int = 128
    max_line_bytes: int = 4096
    max_session_events: int = 2_000_000
    idle_timeout_seconds: float = 600.0

    @classmethod
    def load(cls, path: Path) -> "ServiceConfig":
        """Resolve one operator-owned YAML file without accepting request-selected paths.

        Args:
            path (Path): Local service configuration file.
        """
        path = path.resolve()
        settings: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = set(cls.__dataclass_fields__)
        if type(settings) is not dict or settings.keys() != names:
            raise ValueError("Service config must contain exactly the documented settings")
        for key in names - {"bundle_dir", "idle_timeout_seconds"}:
            if type(settings[key]) is not int or not 0 < settings[key] <= 100_000_000:
                raise ValueError(f"{key} must be a positive integer at most 100000000")
        timeout = settings["idle_timeout_seconds"]
        if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
            raise ValueError("idle_timeout_seconds must be finite and at most 86400")
        location = settings["bundle_dir"]
        if location is not None:
            if type(location) is not str or not location.strip():
                raise ValueError("bundle_dir must be a path string or null")
            settings["bundle_dir"] = path.parent / location
        return cls(**settings)
