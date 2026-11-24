import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.bundle_validation import sha256_bytes

_PROBE = r'''
import json
import os
import sys
from pathlib import Path

runtime, bundle, source, expected = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime))

class SourceAccessGuard:
    """Reject source inputs and Git in the isolated inference process."""

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        """Check filesystem and subprocess audit events.

        Args:
            event (str): Python audit event name.
            arguments (tuple[object, ...]): Arguments supplied by the audited operation.
        """
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(arguments[0])))
            for name in ("artifacts", "data", "src/sleep_classifier"):
                if path.is_relative_to(source / name):
                    raise AssertionError("Original project input accessed: " + str(path))
        if event == "subprocess.Popen" and Path(os.fsdecode(arguments[0])).name == "git":
            raise AssertionError("Git executable accessed")

sys.addaudithook(SourceAccessGuard())
import sleep_classifier.bundle_loader as module
assert Path(module.__file__).is_relative_to(runtime)
loaded = module.EvaluationBundle(bundle)
stream = loaded.predictor()
before = stream.integrity()
records = json.loads(expected.read_text())
responses = [stream.predict(row["request"]) for row in records]
maximum = max(abs(a - b) for response, row in zip(responses, records, strict=True)
              for a, b in zip(response["probabilities"],
                              row["response"]["probabilities"], strict=True))
assert maximum <= 2e-6
assert all(response["prediction"] == row["response"]["prediction"]
           and response["history_epochs"] == row["response"]["history_epochs"]
           for response, row in zip(responses, records, strict=True))
assert stream.integrity() == before
assert before["evaluation_mode"] and not before["gradients_created"]
sys.stdout.write(json.dumps({"status": "passed", "synthetic_epochs": len(records),
 "maximum_probability_difference": maximum, "class_disagreements": 0,
 "loaded_module_from_wheel": True, "source_access_blocked": True, "git_access_blocked": True,
 "weights_unchanged": True, "gradients_created": False,
 "runtime": loaded.manifest["runtime"]}) + "\n")
'''


class WheelBundleVerifier:
    """Compare a copied bundle's installed-wheel inference with its matching source runtime."""

    def __init__(self, bundle: Path, wheel: Path) -> None:
        """Retain the approved bundle and wheel supplied by the operator.

        Args:
            bundle (Path): Complete immutable evaluation bundle.
            wheel (Path): Trusted runtime wheel built at the bundle's exact source revision.
        """
        self.bundle = bundle.resolve()
        self.wheel = wheel.resolve()
        self.source = Path(__file__).resolve().parents[1]


    def _reference(self) -> list[dict[str, object]]:
        """Predict synthetic epochs through context eviction without reading source observations."""
        loaded = EvaluationBundle(self.bundle)
        stream = loaded.predictor()
        features = {"heart_rate_mean": 60.0, "heart_rate_std": 2.0, "heart_rate_min": 57.0,
                    "heart_rate_max": 63.0, "motion_magnitude_mean": 1.0,
                    "motion_magnitude_std": .01, "motion_grid_change_mean": .02}
        rows = []
        for index in range(33):
            request = {"participant_id": "synthetic-replay", "segment_id": 0,
                       "time": float(index * 30), "features": features}
            rows.append({"request": request, "response": stream.predict(request)})
        return rows


    def run(self) -> dict[str, object]:
        """Run the wheel in a separate process with original inputs and Git blocked."""
        records = self._reference()
        directory = "/private/tmp" if sys.platform == "darwin" else None
        with TemporaryDirectory(prefix="bundle-wheel-", dir=directory) as temporary:
            root = Path(temporary).resolve()
            runtime = root / "wheel"
            with ZipFile(self.wheel) as archive:
                for entry in archive.infolist():
                    if not (runtime / entry.filename).resolve().is_relative_to(runtime):
                        raise ValueError("Wheel entry escapes its extraction directory")
                archive.extractall(runtime)
            copied = root / "bundle"
            shutil.copytree(self.bundle, copied, symlinks=True)
            expected = root / "expected.json"
            expected.write_text(json.dumps(records, allow_nan=False))
            probe = root / "probe.py"
            probe.write_text(_PROBE)
            process = subprocess.run(
                [sys.executable, "-I", str(probe), str(runtime), str(copied), str(self.source),
                 str(expected)], cwd=root, text=True, capture_output=True, check=True,
            )
            result = json.loads(process.stdout)
        return result | {"wheel_sha256": sha256_bytes(self.wheel.read_bytes()),
                         "bundle_manifest_sha256": sha256_bytes(
                             (self.bundle / "manifest.json").read_bytes(),
                         ), "training_performed": False, "test_observations_read": False}


def main() -> None:
    """Verify approved local artifacts and save a fresh portability report."""
    parser = argparse.ArgumentParser(
        description="Verify self-contained evaluation bundle wheel loading",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Portability evidence already exists; preserve it")
    result = WheelBundleVerifier(args.bundle, args.wheel).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
