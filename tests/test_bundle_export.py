import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from bundle_fixtures import make_checkpoint  # noqa: E402

from sleep_classifier import bundle_export, bundle_export_source, bundle_parity  # noqa: E402
from sleep_classifier.bundle_export import BundleExportConfig, EvaluationBundleExporter  # noqa: E402
from sleep_classifier.bundle_export_source import GitBundleDocumentation  # noqa: E402
from sleep_classifier.bundle_loader import EvaluationBundle  # noqa: E402
from sleep_classifier.bundle_parity import EvaluationBundleParity  # noqa: E402
from sleep_classifier.bundle_runtime import RuntimeIdentity  # noqa: E402
from sleep_classifier.bundle_validation import sha256_bytes  # noqa: E402
from sleep_classifier.streaming_inference import StreamingEpochPredictor  # noqa: E402

_RUNTIME = {
    "source_revision": "4" * 40, "project_version": "0.1.0", "python_requires": ">=3.11,<3.13",
}
_LOCK = (
    b'version = 1\nrequires-python = ">=3.11, <3.13"\n'
    b'[[package]]\nname = "sleep-classifier"\nversion = "0.1.0"\n'
)
_CARD = b"# Synthetic evaluation checkpoint\n"
_FOLD = "fold-02"
_BUNDLE_ID = "synthetic-evaluation-v1"


@dataclass
class ExportFixture:
    """Expose the synthetic source evidence and isolated export boundaries."""

    exporter: EvaluationBundleExporter
    source_path: Path
    metrics_path: Path
    report: dict[str, Any]
    read_report: Mock
    parity: Mock
    runtime: Mock


def _training_source(directory: Path) -> tuple[Path, Path]:
    """Write an original synthetic training checkpoint and matching metrics.

    Args:
        directory (Path): Temporary results directory for the chosen fold.
    """
    directory.mkdir(parents=True)
    checkpoint = make_checkpoint()
    config = {"seed": 43, "learning_rate": 0.001, "output_dir": str(directory)}
    metrics = {
        name: deepcopy(checkpoint[name]) for name in (
            "model_type", "model_config", "features", "model_features", "motion_features",
            "window_feature_state",
        )
    }
    provenance = {"test_observations_read": False, "split": {
        "train": ["synthetic-train"], "validation": ["synthetic-validation"],
        "test": ["synthetic-reserved"],
    }}
    features = directory.parent / "features"
    features.mkdir()
    (features / "provenance.json").write_text(json.dumps(provenance))
    metrics.update(config=config, device="mps", parameter_count=2900, provenance=provenance)
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    source_path = directory / "model_checkpoint.pt"
    torch.save(
        checkpoint | {"training_config": config, "optimizer_state_dict": {"state": {}}},
        source_path,
    )
    return source_path, metrics_path


def _report(source_path: Path, metrics_path: Path) -> dict[str, Any]:
    """Build completed synthetic audit evidence for the explicit source fold.

    Args:
        source_path (Path): Synthetic source checkpoint.
        metrics_path (Path): Synthetic fold metrics.
    """
    digest = sha256_bytes(source_path.read_bytes())
    comparison = {"max_abs_probability_difference": 1e-7, "prediction_disagreements": 0}
    fold = {
        "fold": _FOLD, "status": "passed", "checkpoint_sha256": digest, "device": "mps",
        "absolute_tolerance": 2e-6, "context_epochs": 30, "history_lengths_match": True,
        "weights_unchanged": True, "gradients_created": False, "training_performed": False,
        "stream_vs_offline": comparison.copy(), "offline_vs_saved": comparison.copy(),
    }
    return {
        "status": "passed", "training_performed": False, "test_observations_read": False,
        "plan": {
            "folds": [_FOLD], "device": "mps", "reference_manifest_sha256": "2" * 64,
            "sources": {_FOLD: {
                "metrics.json": sha256_bytes(metrics_path.read_bytes()),
                "model_checkpoint.pt": digest,
            }},
        },
        "folds": [fold],
    }


def _write_config(root: Path, overrides: dict[str, str] | None = None) -> Path:
    """Write a bundle export configuration with temporary destinations.

    Args:
        root (Path): Temporary directory containing the export configuration.
        overrides (dict[str, str] | None): Alternate settings for a rejection case.
    """
    settings = {
        "audit_config": "audit.yaml", "source_repository": ".", "output_dir": "bundles",
        "verification_dir": "verification", "bundle_id": _BUNDLE_ID, "fold": _FOLD,
    }
    path = root / "export.yaml"
    path.write_text(yaml.safe_dump(settings | (overrides or {})))
    return path


@pytest.fixture
def export_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExportFixture:
    """Isolate provenance, runtime identity, documentation, and MPS parity.

    Args:
        tmp_path (Path): Temporary directory for synthetic inputs and outputs.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing external boundaries.
    """
    root = tmp_path.resolve()
    reference = root / "reference"
    source, metrics = _training_source(reference / _FOLD / "results")
    report = _report(source, metrics)

    study = Mock(output_dir=root / "audit", read_report=Mock(return_value=report))
    study.reference.output_dir = reference
    study.reference.source_dir = root / "source-features"
    study.reference.model_settings = {}
    monkeypatch.setattr(bundle_export_source, "ChronologicalReplayStudy", Mock(return_value=study))
    runtime = Mock(return_value=_RUNTIME.copy())
    monkeypatch.setattr(RuntimeIdentity, "current", runtime)
    documentation = Mock()
    documentation.payload.return_value = {"MODEL_CARD.md": _CARD, "uv.lock": _LOCK}
    monkeypatch.setattr(bundle_export, "GitBundleDocumentation", Mock(return_value=documentation))
    monkeypatch.setattr(StreamingEpochPredictor, "_resolve_device", Mock())
    parity = Mock(return_value={
        "status": "passed", "maximum_probability_difference": 1e-7, "prediction_disagreements": 0,
        "history_lengths_match": True,
    })
    monkeypatch.setattr(EvaluationBundleParity, "build", parity)

    exporter = EvaluationBundleExporter(_write_config(root))
    return ExportFixture(exporter, source, metrics, report, study.read_report, parity, runtime)


def test_export_preserves_original_and_retained_values(
    export_case: ExportFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish the sanitized checkpoint and match the original tensor bits and metadata.

    Args:
        export_case (ExportFixture): Synthetic source, parity, and exporter.
        monkeypatch (pytest.MonkeyPatch): Fixture for observing publication writes.
    """
    original_bytes = export_case.source_path.read_bytes()
    original_metrics = export_case.metrics_path.read_bytes()
    publish = Mock(wraps=bundle_export.publish_directory)
    monkeypatch.setattr(bundle_export, "publish_directory", publish)

    published = export_case.exporter.export()

    assert published == export_case.exporter.output_dir
    files = {"checkpoint.pt", "MODEL_CARD.md", "uv.lock", "manifest.json"}
    assert {entry.name for entry in published.iterdir()} == files
    publish.assert_called_once()
    assert publish.call_args.args[1] == published
    assert not list(published.parent.glob(".bundle-stage-*"))
    assert export_case.source_path.read_bytes() == original_bytes
    assert export_case.metrics_path.read_bytes() == original_metrics
    assert (published / "MODEL_CARD.md").read_bytes() == _CARD
    assert (published / "uv.lock").read_bytes() == _LOCK

    original = torch.load(export_case.source_path, weights_only=True, map_location="cpu")
    retained = torch.load(published / "checkpoint.pt", weights_only=True, map_location="cpu")
    assert set(retained) == set(original) - {"training_config", "optimizer_state_dict"}
    for name in set(retained) - {"state_dict"}:
        assert retained[name] == original[name]
    assert retained["state_dict"].keys() == original["state_dict"].keys()
    for name, tensor in retained["state_dict"].items():
        expected = original["state_dict"][name]
        assert expected.shape == tensor.shape and expected.dtype == tensor.dtype
        assert torch.equal(expected.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(
            torch.uint8,
        ))
    assert str(export_case.source_path.parent).encode() not in (
        published / "checkpoint.pt"
    ).read_bytes()

    bundle = EvaluationBundle(published)
    assert bundle.manifest["provenance"] == {
        "fold_id": _FOLD, "training_seed": 43, "grouped_manifest_sha256": "2" * 64,
        "metrics_sha256": sha256_bytes(original_metrics), "final_test_used": False,
        "source_checkpoint_sha256": sha256_bytes(original_bytes),
    }
    packaged = sha256_bytes((published / "checkpoint.pt").read_bytes())
    assert bundle.manifest["files"]["checkpoint"]["sha256"] == packaged
    assert packaged != sha256_bytes(original_bytes)
    evidence = json.loads(export_case.exporter.verification_path.read_bytes())
    assert evidence["manifest_sha256"] == sha256_bytes((published / "manifest.json").read_bytes())
    assert evidence["sanitized_checkpoint_fields"] == sorted(retained)
    assert evidence["parity"]["status"] == "passed"
    export_case.parity.assert_called_once()


def test_mandatory_parity_failure(export_case: ExportFixture) -> None:
    """Leave no published bundle or acceptance evidence when staged replay fails.

    Args:
        export_case (ExportFixture): Synthetic source and exporter with mock parity.
    """
    export_case.parity.side_effect = ValueError("Bundle parity failed")
    with pytest.raises(ValueError, match="Bundle parity failed"):
        export_case.exporter.export()

    assert not export_case.exporter.output_dir.exists()
    assert not export_case.exporter.verification_path.exists()
    assert not list(export_case.exporter.settings.output_dir.glob(".bundle-stage-*"))


def test_failed_parity_status_blocks_publication(export_case: ExportFixture) -> None:
    """Require a passing replay report before exposing a bundle.

    Args:
        export_case (ExportFixture): Synthetic source and exporter with mock parity.
    """
    export_case.parity.return_value = {"status": "failed"}
    with pytest.raises(ValueError, match="requires successful full-fold parity"):
        export_case.exporter.export()
    assert not export_case.exporter.output_dir.exists()
    assert not export_case.exporter.verification_path.exists()


@pytest.mark.parametrize("changed", ["source", "runtime"])
def test_drift_aborts_publication(export_case: ExportFixture, changed: str) -> None:
    """Abort after staged validation if either frozen identity changes.

    Args:
        export_case (ExportFixture): Synthetic source with mutable read boundaries.
        changed (str): Identity that drifts during staged parity validation.
    """
    if changed == "source":
        updated = deepcopy(export_case.report)
        updated["plan"]["reference_manifest_sha256"] = "9" * 64
        export_case.read_report.side_effect = [export_case.report, updated]
    else:
        updated = _RUNTIME | {"source_revision": "9" * 40}
        export_case.runtime.side_effect = [_RUNTIME, _RUNTIME, updated]
    with pytest.raises(ValueError, match="Source or runtime changed"):
        export_case.exporter.export()
    export_case.parity.assert_called_once()
    assert not export_case.exporter.output_dir.exists()
    assert not export_case.exporter.verification_path.exists()


@pytest.mark.parametrize("existing", ["bundle", "evidence"])
def test_never_replace_existing_destination(export_case: ExportFixture, existing: str) -> None:
    """Preserve an already occupied bundle identifier or verification filename.

    Args:
        export_case (ExportFixture): Synthetic source and exporter.
        existing (str): Destination to reserve before export.
    """
    exporter = export_case.exporter
    path = (
        exporter.output_dir / "old" if existing == "bundle" else exporter.verification_path
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"already exists")
    with pytest.raises(FileExistsError):
        exporter.export()
    assert path.read_bytes() == b"already exists"
    export_case.parity.assert_not_called()


def test_failed_atomic_publication_keeps_no_bundle_or_new_report(
    export_case: ExportFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roll back this attempt's evidence if atomic bundle publication fails.

    Args:
        export_case (ExportFixture): Synthetic source and exporter.
        monkeypatch (pytest.MonkeyPatch): Scoped failure at the publication boundary.
    """
    exporter = export_case.exporter
    monkeypatch.setattr(bundle_export, "publish_directory",
                        Mock(side_effect=OSError("synthetic publication failure")))
    with pytest.raises(OSError, match="synthetic publication failure"):
        exporter.export()
    assert not exporter.output_dir.exists()
    assert not exporter.verification_path.exists()
    assert not list(exporter.settings.output_dir.glob(".bundle-stage-*"))


def test_report_creation_failure_cannot_publish_a_bundle(
    export_case: ExportFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep report-write errors before the irreversible successful publication boundary.

    Args:
        export_case (ExportFixture): Synthetic source and exporter.
        monkeypatch (pytest.MonkeyPatch): Scoped evidence-publication failure.
    """
    monkeypatch.setattr(bundle_export.os, "link",
                        Mock(side_effect=PermissionError("report blocked")))
    with pytest.raises(PermissionError, match="report blocked"):
        export_case.exporter.export()
    assert not export_case.exporter.output_dir.exists()
    assert not export_case.exporter.verification_path.exists()


@pytest.mark.parametrize("fold", ["best", "fold-06", "../fold-01"])
def test_config_requires_explicit_fold(tmp_path: Path, fold: str) -> None:
    """Refuse an unsupported fold selector before opening any source artifacts.

    Args:
        tmp_path (Path): Temporary configuration directory.
        fold (str): Invalid selection.
    """
    with pytest.raises(ValueError, match="explicit planned evaluation fold"):
        BundleExportConfig.load(_write_config(tmp_path, {"fold": fold}))


@pytest.mark.parametrize("change, message", [
    ("audit_failed", "passed chronological audit"),
    ("fold_not_planned", "planned fold"),
    ("final_test", "final-test"),
    ("history_mismatch", "frozen parity policy"),
    ("oversized_delta", "frozen parity tolerance"),
])
def test_source_requires_complete_passed_audit(
    export_case: ExportFixture, change: str, message: str,
) -> None:
    """Reject incomplete, failed, or incompatible evidence for the selected fold.

    Args:
        export_case (ExportFixture): Synthetic source with an initially passing report.
        change (str): Invalid report modification.
        message (str): Expected failure reason.
    """
    report = export_case.report
    if change == "audit_failed":
        report["status"] = "failed"
    elif change == "fold_not_planned":
        report["plan"]["folds"] = ["fold-01"]
    elif change == "final_test":
        report["test_observations_read"] = True
    elif change == "history_mismatch":
        report["folds"][0]["history_lengths_match"] = False
    elif change == "oversized_delta":
        report["folds"][0]["stream_vs_offline"]["max_abs_probability_difference"] = 3e-6

    with pytest.raises(ValueError, match=message):
        export_case.exporter.source.snapshot()


@pytest.mark.parametrize("role", ["checkpoint", "metrics", "fold row"])
def test_recorded_hash_required(
    export_case: ExportFixture, role: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a source file or selected fold row that disagrees with the audit hash.

    Args:
        export_case (ExportFixture): Valid synthetic source evidence.
        role (str): Recorded source identity to corrupt after the snapshot.
        monkeypatch (pytest.MonkeyPatch): Fixture for observing checkpoint deserialization.
    """
    source = export_case.exporter.source
    snapshot = source.snapshot()
    if role == "checkpoint":
        export_case.source_path.write_bytes(b"changed checkpoint")
    elif role == "metrics":
        export_case.metrics_path.write_bytes(b"changed metrics")
    elif role == "fold row":
        snapshot["fold"]["checkpoint_sha256"] = "0" * 64
    load = Mock()
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="differ from the completed audit"):
        source.checkpoint(snapshot)
    load.assert_not_called()


def test_checkpoint_must_match_metrics(export_case: ExportFixture) -> None:
    """Reject contradictory metrics even when the report records their new hash.

    Args:
        export_case (ExportFixture): Synthetic checkpoint and matching recorded metrics.
    """
    metrics = json.loads(export_case.metrics_path.read_bytes())
    metrics["model_config"]["d_model"] = 32
    data = json.dumps(metrics).encode()
    export_case.metrics_path.write_bytes(data)
    export_case.report["plan"]["sources"][_FOLD]["metrics.json"] = sha256_bytes(data)

    source = export_case.exporter.source
    with pytest.raises(ValueError, match="differs from metrics at model_config"):
        source.checkpoint(source.snapshot())


def _git(repository: Path, *arguments: str) -> str:
    """Execute a local Git operation for a temporary documentation fixture.

    Args:
        repository (Path): Temporary repository.
        *arguments (str): Local Git command and arguments.
    """
    result = subprocess.run(
        ["git", "-C", str(repository), "-c", "commit.gpgsign=false", *arguments],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _documentation_repo(repository: Path, card: str) -> str:
    """Commit a model card, evidence target, and lock without a network operation.

    Args:
        repository (Path): Temporary repository directory.
        card (str): Markdown with link cases to test.
    """
    docs = repository / "docs"
    docs.mkdir()
    (docs / "model-card.md").write_text(card)
    (docs / "study.md").write_text("# Experiment\n")
    (repository / "uv.lock").write_bytes(_LOCK)
    _git(repository, "init")
    _git(repository, "remote", "add", "origin", "git@github.com:example/synthetic.git")
    _git(repository, "add", "--", "docs", "uv.lock")
    _git(repository, "-c", "user.name=Bundle Test", "-c", "user.email=test@example.com",
         "commit", "-m", "Synthetic documentation")
    return _git(repository, "rev-parse", "HEAD")


def test_pinned_model_card_links(tmp_path: Path) -> None:
    """Read committed documentation and pin internal links to the exact revision.

    Args:
        tmp_path (Path): Temporary repository for committed documentation inputs.
    """
    card = (
        "# Model\n[study](study.md#experiment) [lock](../uv.lock)\n"
        "[section](#model) [external](https://example.com/study)\n"
    )
    revision = _documentation_repo(tmp_path, card)
    (tmp_path / "docs" / "model-card.md").write_text("uncommitted change")
    with pytest.raises(ValueError, match="Commit model-card"):
        GitBundleDocumentation(tmp_path, revision).payload()
    (tmp_path / "docs" / "model-card.md").write_text(card)
    (tmp_path / "uv.lock").write_bytes(b"uncommitted change")
    payload = GitBundleDocumentation(tmp_path, revision).payload()
    prefix = f"https://github.com/example/synthetic/blob/{revision}"
    expected = (
        f"# Model\n[study]({prefix}/docs/study.md#experiment) [lock]({prefix}/uv.lock)\n"
        "[section](#model) [external](https://example.com/study)\n"
    )
    assert payload == {"MODEL_CARD.md": expected.encode(), "uv.lock": _LOCK}


@pytest.mark.parametrize("target, message", [
    ("missing.md", "documentation revision"),
    ("../../outside.md", "escapes the source repository"),
])
def test_model_card_link_must_exist_at_revision(tmp_path: Path, target: str, message: str) -> None:
    """Reject links to uncommitted or out-of-repository targets.

    Args:
        tmp_path (Path): Temporary repository for committed documentation inputs.
        target (str): Invalid committed link target.
        message (str): Expected failure reason.
    """
    revision = _documentation_repo(tmp_path, f"[evidence]({target})\n")
    (tmp_path / "docs" / "missing.md").write_text("uncommitted evidence")
    with pytest.raises(ValueError, match=message):
        GitBundleDocumentation(tmp_path, revision).payload()


@pytest.mark.parametrize("saved, message", [
    ([float("nan"), 0.2, 0.2, 0.2], "Saved validation probabilities"),
    ([0.7, 0.2, 0.2, 0.2], "Saved validation probabilities"),
    ([0.41, 0.19, 0.2, 0.2], "Bundle parity failed"),
])
def test_replay_requires_finite_matching_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: list[float], message: str,
) -> None:
    """Reject nonfinite, unnormalized, or numerically different saved probabilities.

    Args:
        tmp_path (Path): Temporary synthetic saved prediction directory.
        monkeypatch (pytest.MonkeyPatch): Fixture for isolating inputs and model inference.
        saved (list[float]): Stored probabilities for one validation target.
        message (str): Expected failure reason.
    """
    results = tmp_path / "results"
    results.mkdir()
    epoch = {"participant_id": "p1", "time": 0.0, "target": 0, "segment_id": 0}
    values = {f"probability_{index}": value for index, value in enumerate(saved)}
    pd.DataFrame([epoch | {"prediction": 0} | values]).to_csv(
        results / "validation_predictions.csv", index=False,
    )
    audit = Mock()
    audit._load.return_value = (None, None, pd.DataFrame([epoch]))
    monkeypatch.setattr(bundle_parity, "CheckpointReplayAudit", Mock(return_value=audit))
    replay = EvaluationBundleParity(tmp_path, Mock())
    monkeypatch.setattr(replay, "_predict", Mock(return_value=[{
        "probabilities": [0.4, 0.2, 0.2, 0.2], "history_epochs": 1,
    }]))
    with pytest.raises(ValueError, match=message):
        replay.build()


def test_cached_export_matches_current_source(export_case: ExportFixture) -> None:
    """Reuse verified immutable bytes without publishing or reading checkpoint tensors again.

    Args:
        export_case (ExportFixture): Isolated source and exporter with synthetic model metadata.
    """
    export_case.exporter.export()
    export_case.parity.reset_mock()
    bundle = export_case.exporter.read_existing()
    assert bundle.manifest["bundle_id"] == _BUNDLE_ID
    export_case.parity.assert_not_called()


def test_export_refuses_changed_prepared_provenance(export_case: ExportFixture) -> None:
    """Keep source metrics and the frozen development preparation attached to the same fold.

    Args:
        export_case (ExportFixture): Isolated source and exporter with synthetic model metadata.
    """
    prepared = export_case.source_path.parent.parent / "features/provenance.json"
    prepared.write_text('{"test_observations_read": true}')
    with pytest.raises(ValueError, match="development provenance"):
        export_case.exporter.export()
    assert not export_case.exporter.output_dir.exists()


@pytest.mark.parametrize("destination", ["data/generated", "src/sleep_classifier/generated"])
def test_export_cannot_publish_into_data_or_runtime(
    export_case: ExportFixture, destination: str,
) -> None:
    """Keep generated bundles outside immutable observations and the code identity inputs.

    Args:
        export_case (ExportFixture): Isolated source repository and dependency boundaries.
        destination (str): Forbidden export collection path.
    """
    config = _write_config(export_case.exporter.config_path.parent, {"output_dir": destination})
    with pytest.raises(ValueError, match="outside source"):
        EvaluationBundleExporter(config)
