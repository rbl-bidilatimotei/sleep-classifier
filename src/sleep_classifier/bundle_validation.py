import hashlib
import json
import math
import re
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator, ValidationError

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.motion_feature_input import MOTION_DEFINITION, MOTION_FEATURE
from sleep_classifier.window_feature_settings import WindowFeatureSettings

CHECKPOINT_FIELDS = frozenset({
    "state_dict", "model_type", "model_config", "features", "model_features", "context_epochs",
    "scaler_mean", "scaler_scale", "window_feature_state", "motion_features",
})
_RAW_FEATURES = [*FEATURE_COLUMNS, MOTION_FEATURE]
_WINDOW_FEATURES = {
    "name": "heart_rate_deviation", "history_epochs": 10, "min_history_epochs": 4,
}
_CONSTRUCTOR = {
    "input_size": 7, "num_classes": 4, "d_model": 16, "nhead": 2, "num_layers": 1,
    "feedforward_dim": 32, "dropout": 0.2, "max_context": 30, "patch_size": 3,
    "motion_features": MOTION_DEFINITION, "window_features": _WINDOW_FEATURES,
}
_MOTION_COUNTS = (
    "training_observed_targets", "training_missing_targets",
    "validation_observed_targets", "validation_missing_targets",
)
_MOTION_HASHES = ("audit_config_sha256", "snapshot_manifest_sha256")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of data.

    Args:
        data (bytes): Bytes to hash.
    """
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object and reject duplicate keys.

    Args:
        pairs (list[tuple[str, Any]]): Decoded key and value pairs.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _finite_json_number(token: str) -> float:
    """Decode a JSON fractional number and reject nonfinite values.

    Args:
        token (str): Number token, including nonstandard constants from the decoder.
    """
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


def strict_json_loads(data: bytes | str) -> Any:
    """Decode JSON with unique keys and finite numbers.

    Args:
        data (bytes | str): Encoded JSON document.
    """
    return json.loads(
        data, object_pairs_hook=_unique_object, parse_float=_finite_json_number,
        parse_constant=_finite_json_number,
    )


@cache
def _validator(schema_name: str) -> Draft202012Validator:
    """Load a schema from package resources or an editable source checkout.

    Args:
        schema_name (str): Version-one schema basename without the suffix.
    """
    if schema_name not in ("manifest", "epoch-request", "epoch-response"):
        raise ValueError(f"Unknown bundle schema {schema_name!r}")
    filename = f"{schema_name}.schema.json"
    resource = files("sleep_classifier").joinpath("bundle_schemas", "v1", filename)
    if not resource.is_file():
        root = Path(__file__).resolve().parents[2]
        resource = root / "schemas" / "inference-bundle" / "v1" / filename
    return Draft202012Validator(strict_json_loads(resource.read_bytes()))


def _validate_json_values(value: Any) -> None:
    """Check the JSON types of a schema-valid decoded value.

    Args:
        value (Any): Decoded value from the Python API.
    """
    kind = type(value)
    if kind is dict:
        for entry in value.values():
            _validate_json_values(entry)
    elif kind is list:
        for entry in value:
            _validate_json_values(entry)
    elif kind is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
    elif value is not None and kind not in (bool, int, str):
        raise ValueError("Bundle JSON values must use JSON types")


def validate_schema(instance: Any, schema_name: str) -> None:
    """Validate a decoded JSON value against a version-one schema.

    Args:
        instance (Any): Decoded bundle manifest, epoch request, or epoch response.
        schema_name (str): Version-one schema basename without the suffix.
    """
    try:
        _validator(schema_name).validate(instance)
    except ValidationError as error:
        field = ".".join(str(part) for part in error.absolute_path)
        location = f" at {field}" if field else ""
        raise ValueError(f"Invalid {schema_name}{location}: {error.message}") from error
    _validate_json_values(instance)


def _matches_profile(actual: Any, expected: Any) -> bool:
    """Match fixed checkpoint settings with exact keys and scalar types.

    Args:
        actual (Any): Checkpoint setting.
        expected (Any): Version-one profile value.
    """
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _matches_profile(actual[key], value) for key, value in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _matches_profile(item, value) for item, value in zip(actual, expected)
        )
    return actual == expected


def _finite_number(value: Any) -> bool:
    """Check whether a metadata value is a finite scalar.

    Args:
        value (Any): Checkpoint metadata value.
    """
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _validate_scaler(checkpoint: dict[str, Any]) -> None:
    """Check saved scaler dimensions and values.

    Args:
        checkpoint (dict[str, Any]): Checkpoint with fitted scaler vectors.
    """
    for name in ("scaler_mean", "scaler_scale"):
        values = checkpoint[name]
        if (
            type(values) is not list or len(values) != len(_RAW_FEATURES)
            or not all(_finite_number(value) for value in values)
        ):
            raise ValueError(f"Checkpoint {name} must have one finite value per raw feature")
    if any(value <= 0 for value in checkpoint["scaler_scale"]):
        raise ValueError("Checkpoint scaler_scale values must be positive")


def _validate_motion_record(record: Any) -> None:
    """Check the fixed motion definition and saved training-only imputation metadata.

    Args:
        record (Any): Checkpoint motion feature record.
    """
    fixed = {
        "name": "fixed_lag_motion", "definition": MOTION_DEFINITION,
        "scaling_fit": "training_targets_only", "missing_policy": "training_mean",
    }
    keys = {*fixed, *_MOTION_HASHES, *_MOTION_COUNTS, "imputation_mean"}
    if (
        type(record) is not dict or record.keys() != keys
        or not all(_matches_profile(record[key], value) for key, value in fixed.items())
    ):
        raise ValueError("Checkpoint motion_features definition or imputation policy differs")
    for key in _MOTION_HASHES:
        if type(record[key]) is not str or re.fullmatch("[0-9a-f]{64}", record[key]) is None:
            raise ValueError("Checkpoint motion feature source hashes must be SHA-256 digests")
    for key in _MOTION_COUNTS:
        if type(record[key]) is not int or record[key] < 0:
            raise ValueError("Checkpoint motion support counts must be nonnegative integers")
    if not record["training_observed_targets"] or not _finite_number(record["imputation_mean"]):
        raise ValueError("Checkpoint motion imputation requires finite observed training data")


def _validate_weights(state_dict: Any) -> None:
    """Check that checkpoint weights contain only finite CPU float32 tensors.

    Args:
        state_dict (Any): Checkpoint weight and buffer mapping.
    """
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint state_dict must be a mapping")
    for key, tensor in state_dict.items():
        if (
            type(key) is not str or type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu" or tensor.dtype != torch.float32
            or tensor.layout != torch.strided or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Checkpoint state_dict must contain finite CPU float32 tensors")
    attributes = getattr(state_dict, "__dict__", {})
    metadata = attributes.get("_metadata", {})
    if attributes.keys() - {"_metadata"} or not isinstance(metadata, dict):
        raise ValueError("Checkpoint state_dict has unexpected metadata")
    for name, record in metadata.items():
        if (
            type(name) is not str or re.fullmatch(r"[a-zA-Z_0-9.]*", name) is None
            or type(record) is not dict or record.keys() != {"version"}
            or type(record["version"]) is not int or record["version"] < 0
        ):
            raise ValueError("Checkpoint state_dict has unexpected module metadata")


def validate_checkpoint(checkpoint: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Validate a sanitized CPU checkpoint against the manifest and version-one model profile.

    Args:
        checkpoint (dict[str, Any]): Loaded weights and retained metadata.
        manifest (dict[str, Any]): Schema-validated bundle manifest.
    """
    if type(checkpoint) is not dict or checkpoint.keys() != CHECKPOINT_FIELDS:
        raise ValueError("Checkpoint must contain exactly the approved inference fields")

    model = manifest["model"]
    effective = WindowFeatureSettings(_WINDOW_FEATURES).model_features(_RAW_FEATURES)
    if (
        not _matches_profile(checkpoint["model_type"], "patch_transformer")
        or checkpoint["model_type"] != model["model_type"]
        or not _matches_profile(checkpoint["context_epochs"], 30)
        or checkpoint["context_epochs"] != model["context_epochs"]
        or not _matches_profile(checkpoint["features"], _RAW_FEATURES)
        or checkpoint["features"] != model["raw_features"]
        or not _matches_profile(checkpoint["model_features"], effective)
        or checkpoint["model_features"] != model["effective_features"]
        or not _matches_profile(checkpoint["model_config"], _CONSTRUCTOR)
    ):
        raise ValueError("Checkpoint constructor or features differ from the version-one profile")

    _validate_scaler(checkpoint)
    _validate_motion_record(checkpoint["motion_features"])
    _validate_weights(checkpoint["state_dict"])
    with torch.random.fork_rng(devices=[]):
        try:
            model_on_cpu = ModelFactory.from_checkpoint(checkpoint)
        except (OverflowError, RuntimeError) as error:
            raise ValueError(f"Invalid checkpoint weights or normalization: {error}") from error
    count = sum(parameter.numel() for parameter in model_on_cpu.parameters())
    if count != 2900 or count != model["parameter_count"]:
        raise ValueError("Checkpoint parameter count differs from the version-one model profile")
    transform = model_on_cpu.transform
    if transform.heart_rate_scale.item() <= 0 or transform.delta_scale.item() <= 0:
        raise ValueError("Window normalization scales must stay positive in float32")
