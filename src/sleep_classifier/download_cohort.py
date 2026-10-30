import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

from sleep_classifier.data import fetch_file, sha256

# TODO: Move this URL to a configuration file or environment variable
ARCHIVE_URL = "https://physionet.org/content/sleep-accel/get-zip/1.0.0/"


# TODO: Split this into smaller functions for clarity and testability
def extract_verified(archive: Path, destination: Path) -> None:
    """Extract manifest-listed files after verifying their publisher checksums.

    Args:
        archive (Path): Local release ZIP file.
        destination (Path): Raw data directory.
    """
    with zipfile.ZipFile(archive) as source:
        manifests = [n for n in source.namelist() if n.endswith("/SHA256SUMS.txt")]
        if len(manifests) != 1:
            raise ValueError("Archive must contain exactly one publisher manifest")
        name = manifests[0]
        prefix = name.removesuffix("SHA256SUMS.txt")
        contents = source.read(name)
        destination.mkdir(parents=True, exist_ok=True)
        
        manifest_path = destination / "SHA256SUMS.txt"
        if manifest_path.is_symlink():
            raise ValueError("Existing manifest must not be a symlink")
        if manifest_path.exists():
            if manifest_path.read_bytes() != contents:
                raise ValueError("Existing manifest differs from the archive")
        
        for line in contents.decode().splitlines():
            expected, relative = line.split()
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe manifest path")
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"Existing checksum mismatch: {relative}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            digest = hashlib.sha256()
            try:
                with source.open(prefix + relative) as stream, tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{target.name}.",
                    suffix=".part",
                    dir=target.parent,
                    delete=False,
                ) as out:
                    temporary_path = Path(out.name)
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        out.write(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Archive checksum mismatch: {relative}")
                temporary_path.replace(target)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        
        if not manifest_path.exists():
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=".SHA256SUMS.",
                    suffix=".part",
                    dir=destination,
                    delete=False,
                ) as out:
                    temporary_path = Path(out.name)
                    out.write(contents)
                temporary_path.replace(manifest_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    fetch_file(ARCHIVE_URL, args.archive)
    extract_verified(args.archive, args.destination)
    print(f"Verified cohort files in {args.destination}")


if __name__ == "__main__":
    main()
