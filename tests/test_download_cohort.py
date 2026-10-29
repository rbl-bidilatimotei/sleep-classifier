"""Verify archive integrity and preservation of existing files."""

import hashlib
import zipfile

import pytest

from sleep_classifier.download_cohort import extract_verified


def archive_at(path, contents=b"0 60\n", digest=None, relative="heart_rate/1_heartrate.txt"):
    digest = digest or hashlib.sha256(contents).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("release/SHA256SUMS.txt", f"{digest} {relative}\n")
        archive.writestr("release/" + relative, contents)
    return path


def test_archive_extracts_and_reuses_verified_files(tmp_path):
    archive = archive_at(tmp_path / "release.zip")
    target = tmp_path / "raw"
    extract_verified(archive, target)
    extract_verified(archive, target)
    file = target / "heart_rate/1_heartrate.txt"
    assert file.read_bytes() == b"0 60\n"
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing checksum"):
        extract_verified(archive, target)
    assert file.read_bytes() == b"changed"


def test_corrupt_archive_member_is_not_published(tmp_path):
    archive = archive_at(tmp_path / "release.zip", digest="0" * 64)
    with pytest.raises(ValueError, match="Archive checksum"):
        extract_verified(archive, tmp_path / "raw")
    assert not list((tmp_path / "raw").rglob("*.txt"))
    assert not list((tmp_path / "raw").rglob("*.part"))


def test_manifest_traversal_is_rejected(tmp_path):
    archive = archive_at(tmp_path / "release.zip", relative="../escape.txt")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_verified(archive, tmp_path / "raw")
    assert not (tmp_path / "escape.txt").exists()
