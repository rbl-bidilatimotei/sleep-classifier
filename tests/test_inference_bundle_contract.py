import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator, ValidationError

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.motion_feature_input import MOTION_FEATURE

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas/inference-bundle/v1"
EXAMPLES = ROOT / "examples/inference-bundle"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one contract or example without accessing model or participant artifacts.

    Args:
        path (Path): Versioned JSON documentation file.
    """
    return json.loads(path.read_text())


def _validator(name: str) -> Draft202012Validator:
    """Build the local validator for one self-contained contract schema.

    Args:
        name (str): Manifest, request, or response schema basename.
    """
    return Draft202012Validator(_read_json(SCHEMAS / f"{name}.schema.json"))


@pytest.mark.parametrize("name", ["manifest", "epoch-request", "epoch-response"])
def test_bundle_schemas_are_valid_draft_2020_12(name: str) -> None:
    """Check the published schema syntax without resolving remote schema resources.

    Args:
        name (str): Versioned contract schema basename.
    """
    Draft202012Validator.check_schema(_read_json(SCHEMAS / f"{name}.schema.json"))


@pytest.mark.parametrize("name", ["manifest", "epoch-request", "epoch-response"])
def test_bundle_examples_match_their_schemas(name: str) -> None:
    """Keep the documentation examples consistent with their declared contract.

    Args:
        name (str): Contract and corresponding example basename.
    """
    filename = "manifest.example.json" if name == "manifest" else f"{name}.json"
    _validator(name).validate(_read_json(EXAMPLES / filename))


@pytest.mark.parametrize(("keys", "value"), [
    (("schema_version",), "2.0.0"),
    (("bundle_id",), "example\n"),
    (("files", "checkpoint", "sha256"), "a" * 64 + "\n"),
    (("runtime", "source_revision"), "a" * 40 + "\n"),
    (("provenance", "fold_id"), "fold-01\n"),
    (("feature_contract_version",), "1.1.0"),
    (("artifact_role",), "deployment_model"),
    (("files", "checkpoint", "path"), "../checkpoint.pt"),
    (("files", "checkpoint", "path"), "/tmp/checkpoint.pt"),
    (("files", "checkpoint", "sha256"), "not-a-hash"),
    (("files", "environment_lock", "path"), "environment.yaml"),
    (("model", "raw_features"), list(reversed((*FEATURE_COLUMNS, MOTION_FEATURE)))),
    (("model", "class_names"), ["wake", "light", "REM", "deep"]),
    (("model", "context_epochs"), 10),
    (("model", "parameter_count"), 2852),
    (("runtime", "source_revision"), "main"),
    (("runtime", "validated_device"), "cpu"),
    (("provenance", "final_test_used"), True),
    (("provenance", "fold_id"), "../fold-01"),
    (("provenance", "training_seed"), True),
])
def test_manifest_rejects_incompatible_or_unsafe_metadata(
    keys: tuple[str, ...], value: Any,
) -> None:
    """Reject version, schema, identity, and provenance mismatches structurally.

    Args:
        keys (tuple[str, ...]): Nested manifest field to replace.
        value (Any): Deliberately incompatible replacement value.
    """
    manifest = copy.deepcopy(_read_json(EXAMPLES / "manifest.example.json"))
    container = manifest
    for key in keys[:-1]:
        container = container[key]
    container[keys[-1]] = value

    with pytest.raises(ValidationError):
        _validator("manifest").validate(manifest)


@pytest.mark.parametrize(("name", "field"), [
    ("manifest", "files"), ("manifest", "provenance"),
    ("epoch-request", "features"), ("epoch-response", "probabilities"),
])
def test_required_contract_fields_cannot_disappear(name: str, field: str) -> None:
    """Reject incomplete records instead of inventing a default or missing model input.

    Args:
        name (str): Contract basename.
        field (str): Required top-level field to remove.
    """
    filename = "manifest.example.json" if name == "manifest" else f"{name}.json"
    record = _read_json(EXAMPLES / filename)
    record.pop(field)
    with pytest.raises(ValidationError):
        _validator(name).validate(record)


@pytest.mark.parametrize("mutation", ["target", "derived", "missing", "null", "boolean", "order"])
def test_epoch_request_requires_label_free_named_features(mutation: str) -> None:
    """Reject labels and incompatible inputs while preserving dictionary-order independence.

    Args:
        mutation (str): Input inconsistency or harmless object-order change.
    """
    request = _read_json(EXAMPLES / "epoch-request.json")
    if mutation == "target":
        request["target"] = 3
    elif mutation == "derived":
        request["features"]["heart_history_available"] = 1.0
    elif mutation == "missing":
        request["features"].pop("heart_rate_mean")
    elif mutation == "null":
        request["features"]["motion_grid_change_mean"] = None
    elif mutation == "boolean":
        request["segment_id"] = True
    else:
        request["features"] = dict(reversed(list(request["features"].items())))
        _validator("epoch-request").validate(request)
        return

    with pytest.raises(ValidationError):
        _validator("epoch-request").validate(request)


@pytest.mark.parametrize(("field", "value"), [
    ("prediction", 4), ("probabilities", [0.5, 0.5]),
    ("probabilities", [-0.1, 0.6, 0.3, 0.2]),
    ("history_epochs", 31), ("reset_reason", "unknown"),
])
def test_epoch_response_rejects_invalid_shapes_and_bounds(field: str, value: Any) -> None:
    """Keep response shapes and scalar bounds compatible with the current predictor.

    Args:
        field (str): Response field to replace.
        value (Any): Invalid replacement value.
    """
    response = _read_json(EXAMPLES / "epoch-response.json")
    response[field] = value
    with pytest.raises(ValidationError):
        _validator("epoch-response").validate(response)


def test_contract_profile_matches_current_recipe_and_runtime_schema() -> None:
    """Detect documented feature-order or recipe drift without loading research weights."""
    manifest = _read_json(EXAMPLES / "manifest.example.json")
    recipe = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps.yaml").read_text())
    raw = [*FEATURE_COLUMNS, MOTION_FEATURE]
    assert manifest["model"]["raw_features"] == raw
    assert manifest["model"]["effective_features"] == [
        *raw, "heart_rate_delta_past", "heart_history_available",
    ]
    assert manifest["model"]["context_epochs"] == recipe["context_epochs"] == 30
    assert manifest["model"]["model_type"] == recipe["model_type"]
    assert manifest["provenance"]["training_seed"] == recipe["seed"]
    assert manifest["runtime"]["validated_device"] == recipe["device"]
    assert manifest["model"]["class_names"] == ["wake", "light", "deep", "REM"]


@pytest.mark.parametrize("name", ["manifest", "epoch-request", "epoch-response"])
def test_contract_records_reject_unknown_fields(name: str) -> None:
    """Reject undeclared metadata rather than assuming forward compatibility.

    Args:
        name (str): Contract and corresponding example basename.
    """
    filename = "manifest.example.json" if name == "manifest" else f"{name}.json"
    record = _read_json(EXAMPLES / filename)
    record["unknown_field"] = "unexpected"
    with pytest.raises(ValidationError):
        _validator(name).validate(record)


def test_request_schema_and_fixed_profile_match_current_preparation() -> None:
    """Detect unversioned changes to the named inputs, support rule, or HR history."""
    request = _read_json(SCHEMAS / "epoch-request.schema.json")
    features = request["properties"]["features"]
    assert features["required"] == [*FEATURE_COLUMNS, MOTION_FEATURE]
    assert set(features["properties"]) == set(features["required"])
    coverage = yaml.safe_load((ROOT / "configs/coverage.yaml").read_text())
    selected = next(row for row in coverage["candidates"] if row["name"] == "moderate")
    assert coverage["selected_rule"] == "moderate"
    assert selected == {
        "name": "moderate", "max_heart_gap_seconds": 15, "min_heart_readings": 3,
        "min_motion_bin_fraction": 0.8, "max_motion_gap_seconds": 5,
    }
    recipe = yaml.safe_load((ROOT / "configs/fixed-lag-motion-mps.yaml").read_text())
    assert recipe["patch_size"] == 3
    assert recipe["window_features"] == {
        "name": "heart_rate_deviation", "history_epochs": 10, "min_history_epochs": 4,
    }
