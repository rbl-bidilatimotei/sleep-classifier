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


def test_member_extraction_does_not_follow_deterministic_part_symlink(tmp_path):
    archive = archive_at(tmp_path / "release.zip")
    target = tmp_path / "raw"
    member = target / "heart_rate/1_heartrate.txt"
    member.parent.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep this file")
    (member.parent / "1_heartrate.txt.part").symlink_to(outside)

    extract_verified(archive, target)

    assert outside.read_text() == "keep this file"
    assert member.read_bytes() == b"0 60\n"


def test_manifest_symlink_is_rejected(tmp_path):
    contents = b"0 60\n"
    archive = archive_at(tmp_path / "release.zip", contents=contents)
    target = tmp_path / "raw"
    target.mkdir()
    outside = tmp_path / "outside-manifest.txt"
    manifest_contents = (
        f"{hashlib.sha256(contents).hexdigest()} heart_rate/1_heartrate.txt\n"
    )
    outside.write_text(manifest_contents)
    (target / "SHA256SUMS.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="manifest.*symlink"):
        extract_verified(archive, target)

    assert outside.read_text() == manifest_contents
