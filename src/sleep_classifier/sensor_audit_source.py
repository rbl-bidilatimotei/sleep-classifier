import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from sleep_classifier.data import load_sensor, sensor_path, sha256


class DevelopmentSensorSource:
    """Restrict raw sensor and saved preparation access to frozen development membership."""

    def __init__(self, prepared_dir: Path, raw_dir: Path, split_path: Path) -> None:
        """Validate partition metadata and file names before opening observations.

        Args:
            prepared_dir (Path): Existing train and validation preparation directory.
            raw_dir (Path): Publisher dataset containing checksum-verified sensors.
            split_path (Path): Frozen participant split with reserved final-test identifiers.
        """
        self.prepared_dir = prepared_dir.resolve()
        self.raw_dir = raw_dir.resolve()
        self.split_path = split_path.resolve()
        self.input_hashes: dict[str, str] = {}
        self.provenance = self._metadata(
            self._inside(self.prepared_dir / "provenance.json", self.prepared_dir),
        )
        self.split = self._metadata(self.split_path)
        if self.provenance["split"] != self.split:
            raise ValueError("Prepared and frozen participant splits differ")
        self.partitions = self._partitions(self.split)
        self.reserved = list(self.split["test"])
        self.rule = self.provenance["rule"]
        inventory = self.provenance["source_inventory"]["participants"]
        self.records = {row["participant_id"]: row for row in inventory}
        if len(self.records) != len(inventory):
            raise ValueError("Inventory participant identities must be unique")
        self._validate_inventory()


    def _metadata(self, path: Path) -> dict[str, Any]:
        """Read metadata and retain its identity for the snapshot.

        Args:
            path (Path): Explicit source metadata file.
        """
        self.input_hashes[str(path)] = sha256(path)
        return json.loads(path.read_text())


    @staticmethod
    def _partitions(split: dict[str, Any]) -> dict[str, str]:
        """Require safe, unique identifiers with disjoint development and test partitions.

        Args:
            split (dict[str, Any]): Frozen train, validation, and test membership.
        """
        identifiers = []
        for name in ("train", "validation", "test"):
            values = split.get(name)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) or not re.fullmatch(r"\d+", value)
                           for value in values)):
                raise ValueError("Participant partitions require numeric identifiers")
            identifiers.extend(values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Participant partitions must be unique and disjoint")
        return {person: partition for partition in ("train", "validation")
                for person in split[partition]}


    def _validate_inventory(self) -> None:
        """Reject unexpected derived tables and missing raw identities before observation reads."""
        if {path.name for path in self.prepared_dir.iterdir()} != {
            "provenance.json", "train", "validation",
        }:
            raise ValueError("Prepared directory contains unexpected entries")
        for partition in ("train", "validation"):
            directory = self.prepared_dir / partition
            expected = {f"{person}.csv" for person in self.split[partition]}
            if {path.name for path in directory.iterdir()} != expected:
                raise ValueError("Prepared tables differ from the development allowlist")
            for name in expected:
                path = self._inside(directory / name, self.prepared_dir)
                if not path.is_file():
                    raise ValueError("Prepared observations must be regular files")
        if not set(self.partitions) <= set(self.records):
            raise ValueError("Raw inventory lacks a development participant")
        for person in self.partitions:
            if not {"heart_rate", "motion"} <= set(self.records[person].get("sha256", {})):
                raise ValueError("Raw inventory lacks required sensor digests")


    @staticmethod
    def _inside(path: Path, root: Path) -> Path:
        """Reject path escapes and symbolic links before accessing observations.

        Args:
            path (Path): Expected source file.
            root (Path): Declared source directory.
        """
        if path.absolute() != path.resolve() or not path.resolve().is_relative_to(root):
            raise ValueError("Source files must remain inside their declared directory")
        return path


    def load_raw(self, participant_id: str, sensor: str) -> pd.DataFrame:
        """Verify and read one allowed development heart-rate or motion file.

        Args:
            participant_id (str): Frozen development participant identifier.
            sensor (str): Either heart_rate or motion; labels and steps are forbidden.
        """
        if participant_id not in self.partitions or sensor not in {"heart_rate", "motion"}:
            raise ValueError("Raw reads require development heart-rate or motion observations")
        path = self._inside(sensor_path(self.raw_dir, participant_id, sensor), self.raw_dir)
        expected = self.records[participant_id]["sha256"][sensor]
        if sha256(path) != expected:
            raise ValueError(f"Raw sensor checksum differs: {path.name}")
        self.input_hashes[str(path)] = expected
        return load_sensor(path, sensor)


    def load_reference(self, participant_id: str) -> pd.DataFrame:
        """Read saved labels and summaries only after label-free preparation completes.

        Args:
            participant_id (str): Frozen development participant identifier.
        """
        if participant_id not in self.partitions:
            raise ValueError("Prepared reads require development membership")
        path = self.prepared_dir / self.partitions[participant_id] / f"{participant_id}.csv"
        self._inside(path, self.prepared_dir)
        self.input_hashes[str(path)] = sha256(path)
        frame = pd.read_csv(path)
        if frame.time.duplicated().any():
            raise ValueError("Prepared epoch times must be unique")
        return frame
