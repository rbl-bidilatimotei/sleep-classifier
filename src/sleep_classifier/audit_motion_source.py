import re
from pathlib import Path

import pandas as pd

from sleep_classifier.data import load_sensor, manifest, sensor_path, sha256


class VerifiedMotionSource:
    """Read checksum-verified motion files for an explicit development allowlist."""

    def __init__(
        self, raw_dir: Path, participant_ids: list[str], reserved_test_ids: list[str],
    ) -> None:
        """Reject overlapping or unsafe participant identifiers before any sensor read.

        Args:
            raw_dir (Path): Publisher dataset directory containing SHA256SUMS.txt.
            participant_ids (list[str]): Development participants permitted for this audit.
            reserved_test_ids (list[str]): Reserved identifiers used only to enforce exclusion.
        """
        identifiers = participant_ids + reserved_test_ids
        if (not participant_ids or any(not isinstance(item, str) or not re.fullmatch(r"\d+", item)
                                       for item in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise ValueError("Audit participant identifiers must be unique, numeric, and disjoint")
        self.raw_dir = raw_dir.resolve()
        self.participant_ids = frozenset(participant_ids)
        self.reserved_test_ids = frozenset(reserved_test_ids)
        self.accessed_files: dict[str, str] = {}


    def load(self, participant_id: str) -> pd.DataFrame:
        """Verify and load one permitted motion file without reading labels or test files.

        Args:
            participant_id (str): Development identifier already present in the allowlist.
        """
        if participant_id not in self.participant_ids or participant_id in self.reserved_test_ids:
            raise ValueError("Motion access is restricted to development participants")
        path = sensor_path(self.raw_dir, participant_id, "motion")
        if path.resolve() != path or not path.resolve().is_relative_to(self.raw_dir):
            raise ValueError("Motion files must use their declared dataset paths")
        checksums = manifest(self.raw_dir)
        relative = path.relative_to(self.raw_dir).as_posix()
        expected = checksums.get(relative)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"Publisher checksum differs for {relative}")
        frame = load_sensor(path, "motion")
        self.accessed_files[str(path)] = expected
        return frame
