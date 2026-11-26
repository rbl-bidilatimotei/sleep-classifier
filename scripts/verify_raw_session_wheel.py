import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

_PROBE = r'''
import os
import sys
from pathlib import Path

runtime, config, events, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime))

class SourceAccessGuard:
    """Block original project observations, artifacts, source modules, and Git."""

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        """Reject source access during the isolated session.

        Args:
            event (str): Python audit event name.
            arguments (tuple[object, ...]): Audit event arguments.
        """
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(arguments[0])))
            for name in ("artifacts", "data", "src/sleep_classifier", "schemas"):
                if path.is_relative_to(source / name):
                    raise AssertionError("Original project input accessed")
        if event == "subprocess.Popen" and Path(os.fsdecode(arguments[0])).name == "git":
            raise AssertionError("Git executable accessed")

sys.addaudithook(SourceAccessGuard())
import sleep_classifier.raw_session_cli as module
assert Path(module.__file__).is_relative_to(runtime)
sys.argv = [str(module.__file__), "--config", str(config), "--input", str(events)]
module.main()
'''


def verify(
    bundle: Path, wheel: Path, events: Path, expected: Path,
) -> dict[str, object]:
    """Compare copied-bundle wheel responses with a trusted source-runtime response file.

    Args:
        bundle (Path): Immutable evaluation bundle to copy.
        wheel (Path): Trusted runtime wheel from the bundle's exact revision.
        events (Path): Synthetic JSONL event example.
        expected (Path): Successful source CLI response file for those inputs.
    """
    source = Path(__file__).resolve().parents[1]
    reference = [json.loads(line) for line in expected.read_text().splitlines()]
    directory = "/private/tmp" if sys.platform == "darwin" else None
    with TemporaryDirectory(prefix="raw-session-wheel-", dir=directory) as temporary:
        root = Path(temporary).resolve()
        runtime = root / "runtime"
        with ZipFile(wheel) as archive:
            archive.extractall(runtime)
        shutil.copytree(bundle, root / "bundle")
        shutil.copyfile(events, root / "events.jsonl")
        (root / "session.yaml").write_text(
            "bundle_dir: bundle\nmax_buffered_events: 8192\n"
            "max_epochs_per_event: 128\nmax_line_bytes: 4096\n",
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", _PROBE, str(runtime), str(root / "session.yaml"),
             str(root / "events.jsonl"), str(source)],
            cwd=root, text=True, capture_output=True, timeout=120, check=True,
        )
        actual = [json.loads(line) for line in result.stdout.splitlines()]
    if actual != reference or not actual or actual[-1]["type"] != "session_finished":
        raise ValueError("Isolated wheel responses differ from the completed source session")
    return {"status": "passed", "response_records": len(actual),
            "prediction_records": sum(row["type"] == "prediction" for row in actual),
            "responses_exactly_equal": True, "source_access_blocked": True,
            "git_access_blocked": True, "loaded_module_from_wheel": True,
            "packaged_session_schemas_used": True, "training_performed": False,
            "test_observations_read": False, "session_summary": actual[-1]}


def main() -> None:
    """Verify the synthetic session through a matching isolated runtime wheel."""
    parser = argparse.ArgumentParser(description="Verify isolated wheel raw-session inference")
    for name in ("bundle", "wheel", "events", "expected"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.bundle.resolve(), args.wheel.resolve(), args.events.resolve(),
                    args.expected.resolve())
    sys.stdout.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
