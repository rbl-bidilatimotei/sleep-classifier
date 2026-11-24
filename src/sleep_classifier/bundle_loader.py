import hashlib
import os
import pickle
import stat
import tomllib
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from sleep_classifier.bundle_runtime import RuntimeIdentity
from sleep_classifier.bundle_validation import (
    sha256_bytes,
    strict_json_loads,
    validate_checkpoint,
    validate_schema,
)
from sleep_classifier.streaming_inference import StreamingEpochPredictor

_FILES = {
    "checkpoint": "checkpoint.pt",
    "model_card": "MODEL_CARD.md",
    "environment_lock": "uv.lock",
}
_DIRECTORY_ENTRIES = {"manifest.json", *_FILES.values()}


def _open_directory(path: Path) -> int:
    """Open the bundle directory without following symlinks in its path.

    Args:
        path (Path): Requested bundle directory.
    """
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_file(directory: int, name: str) -> bytes:
    """Read a regular file relative to the open bundle directory.

    Args:
        directory (int): Bundle directory descriptor.
        name (str): Fixed bundle filename.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        raise ValueError(f"Bundle entry {name} must be a regular file without symlinks") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Bundle entry {name} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_verified_file(directory: int, name: str, expected_digest: str) -> bytes:
    """Read a regular bundle file and verify its recorded SHA-256 digest.

    Args:
        directory (int): Bundle directory descriptor.
        name (str): Fixed bundle filename.
        expected_digest (str): SHA-256 digest from the validated manifest.
    """
    data = _read_regular_file(directory, name)
    if sha256_bytes(data) != expected_digest:
        raise ValueError(f"Bundle SHA-256 differs for {name}")
    return data


def _verify_environment_lock(data: bytes, runtime: dict[str, Any]) -> None:
    """Match the packaged lock's Python requirement and project version to the runtime.

    Args:
        data (bytes): Verified environment lock bytes.
        runtime (dict[str, Any]): Validated runtime record from the manifest.
    """
    try:
        lock = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Bundle uv.lock must contain valid UTF-8 TOML") from error

    requirement = lock.get("requires-python")
    if not isinstance(requirement, str) or sorted(requirement.replace(" ", "").split(",")) != (
        sorted(runtime["python_requires"].split(","))
    ):
        raise ValueError("Bundle uv.lock requires-python differs from the runtime")
    packages = lock.get("package")
    if not isinstance(packages, list) or not all(isinstance(item, dict) for item in packages):
        raise ValueError("Bundle uv.lock must contain package records")
    project = [item for item in packages if item.get("name") == "sleep-classifier"]
    if len(project) != 1 or project[0].get("version") != runtime["project_version"]:
        raise ValueError("Bundle uv.lock project version differs from the runtime")


def _read_bundle_files(bundle_dir: Path) -> tuple[dict[str, Any], bytes]:
    """Verify the bundle layout, schema, lock metadata, and recorded file digests.

    Args:
        bundle_dir (Path): Directory containing a version-one evaluation bundle.
    """
    try:
        descriptor = _open_directory(bundle_dir)
    except OSError as error:
        raise ValueError("Bundle path must be a directory without symlinks") from error

    try:
        names = set(os.listdir(descriptor))
        if names != _DIRECTORY_ENTRIES:
            missing = sorted(_DIRECTORY_ENTRIES - names)
            unexpected = sorted(names - _DIRECTORY_ENTRIES)
            raise ValueError(f"Invalid bundle entries. Missing {missing}. Unexpected {unexpected}.")

        manifest = strict_json_loads(_read_regular_file(descriptor, "manifest.json"))
        validate_schema(manifest, "manifest")

        file_records = manifest["files"]
        _read_verified_file(descriptor, _FILES["model_card"], file_records["model_card"]["sha256"])
        lock = _read_verified_file(
            descriptor, _FILES["environment_lock"], file_records["environment_lock"]["sha256"],
        )
        _verify_environment_lock(lock, manifest["runtime"])
        checkpoint_bytes = _read_verified_file(
            descriptor, _FILES["checkpoint"], file_records["checkpoint"]["sha256"],
        )
        return manifest, checkpoint_bytes
    finally:
        os.close(descriptor)


def inspect_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Inspect file integrity without restoring weights or approving runtime compatibility.

    Args:
        bundle_dir (Path): Directory containing a version-one evaluation bundle.

    Returns:
        dict[str, Any]: Manifest with verified file hashes, without checkpoint semantic validation.
    """
    manifest, _ = _read_bundle_files(bundle_dir)
    return deepcopy(manifest)


class EvaluationBundle:
    """Verify and retain a self-contained evaluation-fold inference checkpoint."""

    def __init__(self, bundle_dir: Path) -> None:
        """Load a bundle from its verified checkpoint bytes.

        Args:
            bundle_dir (Path): Directory with exactly the files required by the contract.
        """
        manifest, checkpoint_bytes = _read_bundle_files(bundle_dir)
        RuntimeIdentity.verify(manifest["runtime"])
        try:
            checkpoint = torch.load(
                BytesIO(checkpoint_bytes), weights_only=True, map_location="cpu",
            )
        except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid bundle checkpoint: {error}") from error
        validate_checkpoint(checkpoint, manifest)
        self._manifest = manifest
        self._checkpoint = checkpoint


    @property
    def manifest(self) -> dict[str, Any]:
        """Return a copy of the verified bundle manifest."""
        return deepcopy(self._manifest)


    @property
    def motion_imputation_mean(self) -> float:
        """Return the checkpoint's saved training-only motion imputation mean."""
        return float(self._checkpoint["motion_features"]["imputation_mean"])


    def predictor(self, device: str = "mps") -> "BundlePredictor":
        """Create an empty stream on the device validated for this bundle.

        Args:
            device (str): Explicit device. It must match the manifest's validated backend.
        """
        validated = self._manifest["runtime"]["validated_device"]
        if device != validated:
            raise ValueError(f"Bundle prediction requires the validated device {validated!r}")
        digest = self._manifest["files"]["checkpoint"]["sha256"]
        return BundlePredictor(StreamingEpochPredictor.from_checkpoint(
            self._checkpoint, digest, device=device,
        ))


class BundlePredictor:
    """Validate version-one requests and maintain one stream's history."""

    def __init__(self, stream: StreamingEpochPredictor) -> None:
        """Wrap a stream backed by a verified checkpoint.

        Args:
            stream (StreamingEpochPredictor): Predictor with the bundle's model and preprocessing.
        """
        self._stream = stream


    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        """Classify one completed epoch from a decoded label-free request.

        Args:
            request (dict[str, Any]): Participant, segment, epoch start time, and prepared features.
        """
        validate_schema(request, "epoch-request")
        response = self._stream.predict_epoch(
            participant_id=request["participant_id"], segment_id=int(request["segment_id"]),
            time=request["time"], features=request["features"],
        )
        self._validate_response(response, request)
        return response


    def integrity(self) -> dict[str, str | bool]:
        """Report model-state identity, evaluation mode, and gradient presence."""
        model = self._stream.model
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
                          .numpy().tobytes())
        return {"state_sha256": digest.hexdigest(), "evaluation_mode": not model.training,
                "gradients_created": any(parameter.grad is not None
                                         for parameter in model.parameters())}


    def reset(self) -> None:
        """Clear retained epochs for this stream."""
        self._stream.reset()


    @staticmethod
    def _validate_response(response: dict[str, Any], request: dict[str, Any]) -> None:
        """Check normalized probabilities, class selection, and epoch identity.

        Args:
            response (dict[str, Any]): Predicted epoch with class and probabilities.
            request (dict[str, Any]): Validated request that produced the prediction.
        """
        validate_schema(response, "epoch-response")
        probabilities = response["probabilities"]
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Prediction probabilities must sum to one within 1e-6")
        if response["prediction"] != probabilities.index(max(probabilities)):
            raise ValueError("Prediction must match the first probability argmax")
        if (
            response["participant_id"] != request["participant_id"]
            or response["time"] != float(request["time"])
        ):
            raise ValueError("Prediction participant and epoch time must match the request")
