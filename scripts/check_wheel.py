import argparse
import base64
import csv
import hashlib
import io
import json
import logging
import re
import stat
import sys
import tomllib
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

LOGGER = logging.getLogger(__name__)
_REQUIRED_SCHEMAS = (
    "bundle_schemas/v1/manifest.schema.json",
    "bundle_schemas/v1/epoch-request.schema.json",
    "bundle_schemas/v1/epoch-response.schema.json",
    "session_schemas/v1/event.schema.json",
    "session_schemas/v1/output.schema.json",
)


class WheelPackageCheck:
    """Check the source identity and allowed contents of one runtime wheel."""

    def __init__(self, wheel: Path, expected_revision: str, project: Path) -> None:
        """Bind a trusted source revision and the corresponding project metadata.

        Args:
            wheel (Path): Built project wheel.
            expected_revision (str): Full commit hash expected in the wheel identity.
            project (Path): Source pyproject.toml with the expected version and Python range.
        """
        if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
            raise ValueError("Expected revision must be a full Git commit hash")
        self.wheel = wheel
        self.expected_revision = expected_revision
        with project.open("rb") as stream:
            self.project = tomllib.load(stream)["project"]


    def run(self) -> dict[str, Any]:
        """Verify runtime metadata, packaged schemas, and the wheel's payload boundary."""
        try:
            with ZipFile(self.wheel) as archive:
                names = self._check_paths(archive)
                identity = self._check_identity(archive)
                self._check_metadata(archive, names, identity)
                self._check_records(archive, names)
                for name in _REQUIRED_SCHEMAS:
                    record = json.loads(archive.read(f"sleep_classifier/{name}"))
                    if (not isinstance(record, dict)
                            or record.get("$schema") != "https://json-schema.org/draft/2020-12/schema"):
                        raise ValueError("Packaged resource is not a versioned JSON schema")
        except (BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("Wheel is incomplete or contains invalid metadata") from error
        return {
            "status": "passed", "source_revision": identity["source_revision"],
            "project_version": identity["project_version"],
            "packaged_schemas": len(_REQUIRED_SCHEMAS), "entries": len(names),
            "dataset_payloads": False, "model_payloads": False,
        }


    @staticmethod
    def _check_paths(archive: ZipFile) -> set[str]:
        """Reject duplicate, unsafe, and unexpected wheel paths.

        Args:
            archive (ZipFile): Open wheel archive.
        """
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        if len(names) != len(entries):
            raise ValueError("Wheel contains duplicate paths")
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or "\\" in name
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("Wheel contains an unsafe path")
            if not path.parts:
                raise ValueError("Wheel contains an empty path")
            root = path.parts[0]
            if root == "sleep_classifier":
                resource = "/".join(path.parts[1:])
                if not (resource.endswith(".py") or resource == "_bundle_runtime.json"
                        or resource in _REQUIRED_SCHEMAS
                        or resource == "templates/causal_audit.html"):
                    raise ValueError("Wheel contains an unexpected package payload")
            elif re.fullmatch(r"sleep_classifier-[^/]+\.dist-info", root):
                if len(path.parts) != 2 or path.name not in {
                    "METADATA", "WHEEL", "RECORD", "entry_points.txt",
                }:
                    raise ValueError("Wheel contains an unexpected metadata payload")
            else:
                raise ValueError("Wheel contains files outside the runtime distribution")
        return names


    def _check_identity(self, archive: ZipFile) -> dict[str, str]:
        """Require the embedded identity to match the clean source build.

        Args:
            archive (ZipFile): Open wheel archive.
        """
        identity = json.loads(archive.read("sleep_classifier/_bundle_runtime.json"))
        expected = {
            "source_revision": self.expected_revision,
            "project_version": self.project["version"],
            "python_requires": self.project["requires-python"],
        }
        if identity != expected:
            raise ValueError("Wheel runtime identity does not match the source build")
        return identity


    @staticmethod
    def _check_metadata(
        archive: ZipFile, names: set[str], identity: dict[str, str],
    ) -> None:
        """Require one distribution with matching wheel metadata.

        Args:
            archive (ZipFile): Open wheel archive.
            names (set[str]): Validated archive paths.
            identity (dict[str, str]): Verified runtime identity.
        """
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Wheel requires exactly one distribution metadata record")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if (metadata["Name"] != "sleep-classifier"
                or metadata["Version"] != identity["project_version"]
                or set(metadata.get("Requires-Python", "").split(","))
                != set(identity["python_requires"].split(","))):
            raise ValueError("Wheel distribution metadata does not match its runtime identity")
        directory = metadata_names[0].rsplit("/", 1)[0]
        if directory != f"sleep_classifier-{identity['project_version']}.dist-info":
            raise ValueError("Wheel metadata directory does not match its version")
        if not {f"{directory}/WHEEL", f"{directory}/RECORD"}.issubset(names):
            raise ValueError("Wheel is missing its format or record metadata")


    @staticmethod
    def _check_records(archive: ZipFile, names: set[str]) -> None:
        """Check every packaged file against its wheel RECORD digest and length.

        Args:
            archive (ZipFile): Open wheel archive.
            names (set[str]): Validated archive paths.
        """
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        if any(len(row) != 3 for row in rows):
            raise ValueError("Wheel RECORD requires three fields per row")
        paths = [row[0] for row in rows]
        if len(set(paths)) != len(paths) or set(paths) != names:
            raise ValueError("Wheel RECORD does not cover exactly the packaged files")
        for name, digest, size in rows:
            if name == record_name:
                if digest or size:
                    raise ValueError("Wheel RECORD must not hash itself")
                continue
            data = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if digest != f"sha256={expected}" or size != str(len(data)):
                raise ValueError("Wheel RECORD digest or length does not match a packaged file")


def main() -> None:
    """Check a built wheel without importing the model or reading observations."""
    parser = argparse.ArgumentParser(description="Check runtime wheel identity and payloads")
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    report = WheelPackageCheck(args.wheel, args.expected_revision, args.project).run()
    LOGGER.info("Wheel identity and %s schemas passed", report["packaged_schemas"])
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
