import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleep_classifier import bundle_publication
from sleep_classifier.bundle_publication import publish_directory

requires_exclusive_rename = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="Exclusive rename requires macOS or Linux",
)


@requires_exclusive_rename
def test_publish_directory_moves_complete_payload(tmp_path: Path) -> None:
    """Move the staged directory and its file in one rename.

    Args:
        tmp_path (Path): Temporary directory for staging and publication.
    """
    staging = tmp_path / ".stage"
    staging.mkdir()
    manifest = staging / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "bundle"
    directory_inode = staging.stat().st_ino
    manifest_inode = manifest.stat().st_ino

    publish_directory(staging, destination)

    assert not staging.exists()
    assert destination.stat().st_ino == directory_inode
    published = destination / "manifest.json"
    assert published.stat().st_ino == manifest_inode
    assert published.read_text(encoding="utf-8") == "{}\n"


@requires_exclusive_rename
@pytest.mark.parametrize("kind", ["empty directory", "nonempty directory", "symlink"])
def test_publish_directory_preserves_existing_destination(tmp_path: Path, kind: str) -> None:
    """Preserve an existing destination and the staged directory on failure.

    Args:
        tmp_path (Path): Temporary directory for staging and publication.
        kind (str): Existing destination type to exercise.
    """
    staging = tmp_path / ".stage"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "bundle"
    target = tmp_path / "missing"
    if kind == "symlink":
        destination.symlink_to(target, target_is_directory=True)
    else:
        destination.mkdir()
        if kind == "nonempty directory":
            (destination / "preserve").write_text("old", encoding="utf-8")
    destination_inode = destination.lstat().st_ino

    with pytest.raises(FileExistsError) as error:
        publish_directory(staging, destination)

    assert error.value.filename == str(staging)
    assert error.value.filename2 == str(destination)
    assert (staging / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert destination.lstat().st_ino == destination_inode
    if kind == "symlink":
        assert destination.readlink() == target
        assert not target.exists()
    elif kind == "nonempty directory":
        assert (destination / "preserve").read_text(encoding="utf-8") == "old"
    else:
        assert list(destination.iterdir()) == []


def test_publish_directory_rejects_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before a rename on a platform without a supported API.

    Args:
        tmp_path (Path): Temporary directory for staging and publication.
        monkeypatch (pytest.MonkeyPatch): Attribute replacement fixture.
    """
    staging = tmp_path / ".stage"
    staging.mkdir()
    destination = tmp_path / "bundle"
    monkeypatch.setattr(bundle_publication, "sys", SimpleNamespace(platform="freebsd"))

    with pytest.raises(NotImplementedError, match="requires macOS or Linux"):
        publish_directory(staging, destination)

    assert staging.is_dir()
    assert not destination.exists()
