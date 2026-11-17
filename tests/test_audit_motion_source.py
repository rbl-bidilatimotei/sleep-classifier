import hashlib
from pathlib import Path

import pytest

from sleep_classifier.audit_motion_source import VerifiedMotionSource


def _raw_motion(root: Path) -> Path:
    """Write a permitted motion file and its publisher-style checksum metadata.

    Args:
        root (Path): Temporary dataset directory.
    """
    path = root / "motion/1_acceleration.txt"
    path.parent.mkdir(parents=True)
    path.write_text("0 1 0 0\n0.02 0 1 0\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "SHA256SUMS.txt").write_text(f"{digest} motion/1_acceleration.txt\n")
    return path


def test_motion_access_requires_allowlist_before_opening_files(tmp_path: Path) -> None:
    """Reject reserved and unknown identifiers even when no dataset files exist.

    Args:
        tmp_path (Path): Empty dataset directory.
    """
    source = VerifiedMotionSource(tmp_path, ["1"], ["2"])
    for participant in ("2", "3", "../2"):
        with pytest.raises(ValueError, match="development"):
            source.load(participant)
    assert not source.accessed_files


def test_motion_source_verifies_original_bytes(tmp_path: Path) -> None:
    """Load only the permitted checksum-verified sensor and reject later modifications.

    Args:
        tmp_path (Path): Temporary publisher dataset directory.
    """
    path = _raw_motion(tmp_path)
    source = VerifiedMotionSource(tmp_path, ["1"], ["2"])
    frame = source.load("1")
    assert list(frame.columns) == ["time", "x", "y", "z"]
    assert len(frame) == 2
    assert list(source.accessed_files) == [str(path)]
    path.write_text("0 9 9 9\n")
    with pytest.raises(ValueError, match="checksum"):
        source.load("1")


def test_motion_source_rejects_reserved_symlink_before_checksum_access(tmp_path: Path) -> None:
    """Reject an allowed filename that aliases a reserved participant's sensor.

    Args:
        tmp_path (Path): Temporary dataset directory.
    """
    directory = tmp_path / "motion"
    directory.mkdir()
    reserved = directory / "2_acceleration.txt"
    reserved.write_text("reserved sentinel")
    (directory / "1_acceleration.txt").symlink_to(reserved)
    source = VerifiedMotionSource(tmp_path, ["1"], ["2"])
    with pytest.raises(ValueError, match="declared.*paths"):
        source.load("1")
    assert not source.accessed_files


@pytest.mark.parametrize("allowed,reserved", [(["1"], ["1"]), (["../2"], []), ([], ["2"])])
def test_motion_source_rejects_unsafe_membership(
    tmp_path: Path, allowed: list[str], reserved: list[str],
) -> None:
    """Require disjoint development and reserved identifiers before any sensor access.

    Args:
        tmp_path (Path): Temporary dataset directory.
        allowed (list[str]): Invalid development allowlist.
        reserved (list[str]): Reserved participant identifiers.
    """
    with pytest.raises(ValueError):
        VerifiedMotionSource(tmp_path, allowed, reserved)
