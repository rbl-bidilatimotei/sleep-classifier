import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("jsonschema")

from bundle_fixtures import make_checkpoint  # noqa: E402

from sleep_classifier.bundle_loader import BundlePredictor, EvaluationBundle  # noqa: E402
from sleep_classifier.bundle_runtime import RuntimeIdentity  # noqa: E402
from sleep_classifier.bundle_validation import sha256_bytes  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.motion_feature_input import (  # noqa: E402
    MOTION_FEATURE,
    MotionFeatureInput,
)
from sleep_classifier.streaming_inference import StreamingEpochPredictor  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_LOCK = (
    b'version = 1\nrequires-python = ">=3.11, <3.13"\n'
    b'[[package]]\nname = "sleep-classifier"\nversion = "0.1.0"\n'
)


@pytest.fixture
def checkpoint() -> dict[str, Any]:
    """Provide a synthetic checkpoint for loader tests."""
    return make_checkpoint()

@pytest.fixture
def bundle_dir(tmp_path: Path, checkpoint: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Path:
    """Assemble a synthetic bundle and isolate runtime identity from the working checkout.

    Args:
        tmp_path (Path): Temporary directory for generated bundle files.
        checkpoint (dict[str, Any]): Synthetic model weights and metadata.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing dependency boundaries.
    """
    bundle = tmp_path.resolve() / "bundle"
    bundle.mkdir()
    torch.save(checkpoint, bundle / "checkpoint.pt")
    (bundle / "MODEL_CARD.md").write_text("# Synthetic weights\n", encoding="utf-8")
    (bundle / "uv.lock").write_bytes(_LOCK)
    example = _ROOT / "examples" / "inference-bundle" / "manifest.example.json"
    manifest = json.loads(example.read_bytes())
    for record in manifest["files"].values():
        record["sha256"] = sha256_bytes((bundle / record["path"]).read_bytes())
    _write_manifest(bundle, manifest)
    monkeypatch.setattr(RuntimeIdentity, "verify", Mock())
    return bundle


def _read_manifest(bundle: Path) -> dict[str, Any]:
    """Read the temporary fixture manifest.

    Args:
        bundle (Path): Temporary bundle directory.
    """
    return json.loads((bundle / "manifest.json").read_bytes())


def _write_manifest(bundle: Path, manifest: dict[str, Any]) -> None:
    """Replace the fixture manifest with modified JSON.

    Args:
        bundle (Path): Temporary bundle directory.
        manifest (dict[str, Any]): Synthetic manifest to write.
    """
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _refresh_digest(bundle: Path, role: str) -> None:
    """Update a fixture digest after modifying the associated file.

    Args:
        bundle (Path): Temporary bundle directory.
        role (str): Manifest file role to update.
    """
    manifest = _read_manifest(bundle)
    record = manifest["files"][role]
    record["sha256"] = sha256_bytes((bundle / record["path"]).read_bytes())
    _write_manifest(bundle, manifest)


def _request(index: int) -> dict[str, Any]:
    """Generate one finite, label-free epoch request.

    Args:
        index (int): Epoch position in a synthetic session.
    """
    features = dict.fromkeys((*FEATURE_COLUMNS, MOTION_FEATURE), 0.5)
    features["heart_rate_mean"] = 60.0 + index
    return {"participant_id": "p1", "segment_id": 0.0, "time": index * 30, "features": features}


@pytest.mark.parametrize("filename", ["checkpoint.pt", "MODEL_CARD.md", "uv.lock"])
def test_hash_mismatch_precedes_deserialization(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch, filename: str,
) -> None:
    """Reject changed bytes before deserializing the checkpoint.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        monkeypatch (pytest.MonkeyPatch): Fixture for observing checkpoint deserialization.
        filename (str): Verified file to replace.
    """
    (bundle_dir / filename).write_bytes(b"changed")
    load = Mock()
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        EvaluationBundle(bundle_dir)
    load.assert_not_called()


@pytest.mark.parametrize("change", ["extra", "missing", "directory", "symlink", "root", "ancestor"])
def test_invalid_bundle_paths(bundle_dir: Path, change: str) -> None:
    """Reject unexpected entries and symbolic links to otherwise valid content.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        change (str): Invalid filesystem layout to construct.
    """
    card = bundle_dir / "MODEL_CARD.md"
    if change == "extra":
        (bundle_dir / "unlisted").mkdir()
    elif change == "missing":
        card.unlink()
    elif change == "directory":
        card.unlink()
        card.mkdir()
    elif change == "symlink":
        target = card.rename(bundle_dir.parent / "card.md")
        card.symlink_to(target)
    elif change == "root":
        link = bundle_dir.parent / "alias"
        link.symlink_to(bundle_dir, target_is_directory=True)
        bundle_dir = link
    elif change == "ancestor":
        parent = bundle_dir.parent
        target = parent / "real"
        target.mkdir()
        bundle_dir.rename(target / "bundle")
        link = parent / "alias"
        link.symlink_to(target, target_is_directory=True)
        bundle_dir = link / "bundle"

    with pytest.raises(ValueError, match="entries|regular file|without symlinks"):
        EvaluationBundle(bundle_dir)


@pytest.mark.parametrize("field, value", [
    ("schema_version", "2.0.0"),
    ("feature_contract_version", "1.1.0"),
    ("artifact_role", "deployment_model"),
    ("unexpected", "unknown"),
])
def test_unknown_manifest_contract(bundle_dir: Path, field: str, value: str) -> None:
    """Reject roles, versions, and manifest fields outside the supported contract.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        field (str): Manifest field to change.
        value (str): Unsupported value to write.
    """
    manifest = _read_manifest(bundle_dir)
    manifest[field] = value
    _write_manifest(bundle_dir, manifest)
    with pytest.raises(ValueError, match="Invalid manifest"):
        EvaluationBundle(bundle_dir)


@pytest.mark.parametrize("encoded, message", [
    (b"{", "property name"),
    (b'{"x": {"key": 1, "key": 2}}', "Duplicate JSON key"),
    (b'{"x": 1e999}', "JSON numbers must be finite"),
    (b'{"x": NaN}', "JSON numbers must be finite"),
    (b'{"x": Infinity}', "JSON numbers must be finite"),
])
def test_strict_manifest_json(bundle_dir: Path, encoded: bytes, message: str) -> None:
    """Reject malformed or ambiguous manifest input at JSON decoding.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        encoded (bytes): Invalid manifest content.
        message (str): Expected error text.
    """
    (bundle_dir / "manifest.json").write_bytes(encoded)
    with pytest.raises(ValueError, match=message):
        EvaluationBundle(bundle_dir)


@pytest.mark.parametrize("before, after, message", [
    (b"<3.13", b"<3.12", "requires-python"),
    (b"0.1.0", b"0.2.0", "project version"),
    (b"[[package]]", b"[[package]", "valid UTF-8 TOML"),
])
def test_lock_metadata(
    bundle_dir: Path, before: bytes, after: bytes, message: str,
) -> None:
    """Reject a lock whose metadata disagrees even when its hash matches.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        before (bytes): Lock text to replace.
        after (bytes): Invalid replacement text.
        message (str): Expected error text.
    """
    (bundle_dir / "uv.lock").write_bytes(_LOCK.replace(before, after))
    _refresh_digest(bundle_dir, "environment_lock")
    with pytest.raises(ValueError, match=message):
        EvaluationBundle(bundle_dir)


def test_runtime_rejection_propagates(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop loading when the runtime verifier rejects a dirty checkout.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        monkeypatch (pytest.MonkeyPatch): Fixture for simulating a dirty runtime.
    """
    verify = Mock(side_effect=ValueError("runtime source has uncommitted changes"))
    monkeypatch.setattr(RuntimeIdentity, "verify", verify)
    load = Mock()
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="uncommitted changes"):
        EvaluationBundle(bundle_dir)
    verify.assert_called_once_with(_read_manifest(bundle_dir)["runtime"])
    load.assert_not_called()


@pytest.mark.parametrize("field, name, value, message", [
    ("training_config", None, {}, "approved inference fields"),
    ("model_config", "local_path", "/home/research/source.csv", "constructor"),
    ("model_config", "d_model", 32, "constructor"),
    ("features", 0, "motion_grid_change_mean", "features"),
    ("scaler_mean", 0, float("nan"), "finite"),
    ("scaler_scale", 0, 0.0, "positive"),
    ("scaler_scale", 0, True, "finite"),
    ("motion_features", "imputation_mean", float("inf"), "imputation"),
    ("motion_features", "snapshot_manifest_sha256", "../source.csv", "SHA-256"),
    ("window_feature_state", "delta_mean", 3.0, "buffers differ"),
])
def test_checkpoint_profile(
    bundle_dir: Path, checkpoint: dict[str, Any], field: str, name: str | int | None,
    value: Any, message: str,
) -> None:
    """Reject incompatible weights or metadata after the checkpoint hash is verified.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        checkpoint (dict[str, Any]): Valid weights and metadata to modify.
        field (str): Top-level checkpoint field name.
        name (str | int | None): Mapping key or vector position to replace, if nested.
        value (Any): Invalid metadata value.
        message (str): Expected error text.
    """
    if name is None:
        checkpoint[field] = value
    else:
        checkpoint[field][name] = value
    torch.save(checkpoint, bundle_dir / "checkpoint.pt")
    _refresh_digest(bundle_dir, "checkpoint")
    with pytest.raises(ValueError, match=message):
        EvaluationBundle(bundle_dir)


def test_in_memory_streams(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use verified weights after the checkpoint on disk changes.

    Args:
        bundle_dir (Path): Valid synthetic bundle directory.
        monkeypatch (pytest.MonkeyPatch): Fixture for simulating MPS calls on CPU.
    """
    monkeypatch.chdir(bundle_dir)
    bundle = EvaluationBundle(Path("."))
    assert bundle.motion_imputation_mean == 0.125
    assert bundle.manifest["runtime"]["validated_device"] == "mps"
    with pytest.raises(ValueError, match="validated device"):
        bundle.predictor("cpu")

    (bundle_dir / "checkpoint.pt").write_bytes(b"no longer contains weights")
    failure = Mock(side_effect=AssertionError("unexpected source read"))
    monkeypatch.setattr(torch, "load", failure)
    monkeypatch.setattr(MotionFeatureInput, "_load_sensor", failure)
    resolve = Mock(return_value=torch.device("cpu"))
    monkeypatch.setattr(StreamingEpochPredictor, "_resolve_device", resolve)
    stream = bundle.predictor()
    resolve.assert_called_once_with("mps")
    integrity = stream.integrity()
    assert len(integrity["state_sha256"]) == 64
    assert integrity["evaluation_mode"] is True
    assert integrity["gradients_created"] is False

    first = stream.predict(_request(0))
    assert first["history_epochs"] == 1
    assert first["reset_reason"] is None
    bad = _request(1)
    bad["label"] = 0
    with pytest.raises(ValueError, match="Invalid epoch-request"):
        stream.predict(bad)
    bad = _request(1)
    bad["features"]["heart_rate_mean"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        stream.predict(bad)
    with pytest.raises(ValueError, match="Invalid epoch-request"):
        stream.predict(_request(1) | {"segment_id": True})
    for index in range(1, 32):
        result = stream.predict(_request(index))
        assert result["history_epochs"] == min(30, index + 1)
    expected = bundle.predictor().predict(_request(0))
    assert first == expected
    assert stream.predict(_request(33))["reset_reason"] == "time_gap"
    assert stream.predict(_request(34) | {"segment_id": 2.0})["reset_reason"] == "segment_change"
    stream.reset()
    assert stream.predict(_request(0)) == expected
    assert integrity == stream.integrity()
    failure.assert_not_called()


@pytest.mark.parametrize("update, message", [
    ({"probabilities": [0.5, 0.2, 0.2, 0.2]}, "sum to one"),
    ({"prediction": 1}, "first probability argmax"),
    ({"participant_id": "p2"}, "must match the request"),
    ({"time": 30}, "must match the request"),
])
def test_response_semantics(update: dict[str, Any], message: str) -> None:
    """Reject schema-valid responses that violate normalization or epoch identity.

    Args:
        update (dict[str, Any]): Invalid fields to replace on the prediction response.
        message (str): Expected error text.
    """
    request = _request(0)
    response = {
        "participant_id": "p1", "time": 0, "prediction": 0, "history_epochs": 1,
        "reset_reason": None, "probabilities": [0.4, 0.4, 0.1, 0.1],
    }
    model = Mock()
    model.predict_epoch.return_value = response | update
    with pytest.raises(ValueError, match=message):
        BundlePredictor(model).predict(request)
    assert type(model.predict_epoch.call_args.kwargs["segment_id"]) is int
