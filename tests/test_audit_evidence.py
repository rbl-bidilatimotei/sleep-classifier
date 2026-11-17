import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from sleep_classifier.audit_evidence import AuditEvidenceLoader
from sleep_classifier.features import FEATURE_COLUMNS

ORIGINAL_OPEN = Path.open


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one fixture object with parent directories.

    Args:
        path (Path): Fixture destination.
        value (dict[str, Any]): JSON-compatible fixture metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_study(root: Path, candidate: bool = False) -> Path:
    """Build two honest development folds with reserved observations outside the allowed paths.

    Args:
        root (Path): Temporary grouped artifact directory.
        candidate (bool): Correct the candidate's REM prediction only.
    """
    folds = [{"name": "fold-01", "train": ["b"], "validation": ["a"]},
             {"name": "fold-02", "train": ["a"], "validation": ["b"]}]
    plan = {
        "folds": folds, "development": ["a", "b"], "reserved_test": ["reserved"],
        "split_seed": 42, "training_seed": 42, "maximum_epochs": 4,
        "checkpoint_selection": "best macro-F1 on development fold",
        "independent_outer_test": False,
    }
    results = []
    hashes = {}
    recipe = {"seed": 42, "epochs": 2, "batch_size": 2, "context_epochs": 30,
              "learning_rate": 0.001, "patience": 1, "gradient_clip": 1, "device": "cpu"}
    for fold in folds:
        directory = root / fold["name"]
        split = {"train": fold["train"], "validation": fold["validation"], "test": ["reserved"]}
        provenance = {
            "split": split, "feature_columns": list(FEATURE_COLUMNS),
            "participants": [{"participant_id": person, "partition": partition,
                              "eligible_epochs": 4}
                             for partition in ("train", "validation")
                             for person in fold[partition]],
        }
        _write_json(directory / "features/provenance.json", provenance)
        paths = [directory / "features/provenance.json"]
        for partition in ("train", "validation"):
            file = directory / "features" / partition / f"{fold[partition][0]}.csv"
            file.parent.mkdir(parents=True)
            frame = pd.DataFrame({name: [0.0, 1.0, 2.0, 3.0] for name in FEATURE_COLUMNS})
            frame.assign(time=[0, 30, 60, 90], target=[0, 1, 2, 3], eligible=True,
                         segment_id=0).to_csv(file, index=False)
            paths.append(file)
        for file in paths:
            hashes[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
        person = fold["validation"][0]
        predicted = [0, 1, 1, 3] if candidate else [0, 1, 1, 1]
        predictions = pd.DataFrame({
            "participant_id": [person] * 4, "time": [0, 30, 60, 90],
            "target": [0, 1, 2, 3], "prediction": predicted,
        })
        for stage in range(4):
            predictions[f"probability_{stage}"] = np.where(
                predictions.prediction == stage, 0.7, 0.1,
            )
        output = directory / "results"
        output.mkdir()
        predictions.to_csv(output / "validation_predictions.csv", index=False)
        metrics = {
            "accuracy": float(accuracy_score(predictions.target, predictions.prediction)),
            "macro_f1": float(f1_score(predictions.target, predictions.prediction,
                                       labels=range(4), average="macro", zero_division=0)),
            "best_epoch": 1, "features": list(FEATURE_COLUMNS), "device": "cpu",
            "participants": {person: {"eligible_epochs": 4}},
            "coverage": {"eligible_epochs": 4, "fraction": 1}, "config": recipe,
            "provenance": {"split": split, "feature_columns": list(FEATURE_COLUMNS),
                           "test_observations_read": False},
        }
        path = output / "metrics.json"
        _write_json(path, metrics)
        results.append({"fold": fold["name"], "metrics_path": str(path),
                        "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        **{key: metrics[key] for key in
                           ("accuracy", "macro_f1", "best_epoch", "participants")}})
    _write_json(root / "manifest.json", {"plan": plan, "derived_hashes": hashes})
    _write_json(root / "summary.json", {"status": "complete", "plan": plan, "results": results})
    (root / "reserved.csv").write_text("poison: final test observations must never be read")
    return root / "summary.json"


def _guard_reserved_open(self: Path, *args: Any, **kwargs: Any) -> Any:
    """Fail if any code opens a deliberately poisoned reserved observation file.

    Args:
        self (Path): Requested file path.
        args (Any): Remaining file-open positional arguments.
        kwargs (Any): Remaining file-open keyword arguments.
    """
    if self.name == "reserved.csv":
        raise AssertionError("Reserved test observations were opened")
    return ORIGINAL_OPEN(self, *args, **kwargs)


def test_audit_loads_verified_predictions_and_excludes_reserved_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Align saved model evidence and define persistence as an intersection of matching errors.

    Args:
        tmp_path (Path): Temporary grouped artifact root.
        monkeypatch (pytest.MonkeyPatch): Scoped reserved-file access guard.
    """
    reference = _write_study(tmp_path / "reference")
    candidate = _write_study(tmp_path / "candidate", candidate=True)
    monkeypatch.setattr(Path, "open", _guard_reserved_open)
    result = AuditEvidenceLoader(reference, {
        "candidate": {"summary": candidate, "comparison": "architecture"},
    }).load()
    rows = result["rows"]

    assert len(rows) == 8 and set(rows.participant_id) == {"a", "b"}
    assert result["reserved_test_ids"] == ["reserved"]
    assert set(rows.loc[rows.persistent_to_light, "target"]) == {2}
    assert rows.persistent_to_light.sum() == 2
    assert rows.reference_correct.sum() == 4
    assert np.allclose(rows.confidence_reference, 0.7)
    assert np.allclose(rows.loc[rows.target == 2, "p_true_reference"], 0.1)
    assert not any(Path(path).name == "reserved.csv" for path in result["input_hashes"])
    assert {"reference", "candidate"} == set(result["reports"])


def _forbid_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Reject every observation read during invalid-metadata tests.

    Args:
        args (Any): CSV reader arguments.
        kwargs (Any): CSV reader keyword arguments.
    """
    raise AssertionError("Observation data were accessed before metadata validation")


def test_reserved_participant_in_plan_is_rejected_before_observation_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a contaminated development plan before loading any CSV file.

    Args:
        tmp_path (Path): Temporary grouped root.
        monkeypatch (pytest.MonkeyPatch): Scoped observation-read rejection.
    """
    reference = _write_study(tmp_path)
    summary = json.loads(reference.read_text())
    summary["plan"]["development"].append("reserved")
    reference.write_text(json.dumps(summary))
    monkeypatch.setattr(pd, "read_csv", _forbid_csv)
    with pytest.raises(ValueError, match="reserved test"):
        AuditEvidenceLoader(reference, {}).load()


def test_prepared_provenance_cannot_redirect_to_reserved_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a changed prepared split even when its manifest digest is updated.

    Args:
        tmp_path (Path): Temporary grouped root.
        monkeypatch (pytest.MonkeyPatch): Scoped observation-read rejection.
    """
    reference = _write_study(tmp_path)
    path = tmp_path / "fold-01/features/provenance.json"
    provenance = json.loads(path.read_text())
    provenance["split"]["train"] = ["reserved"]
    path.write_text(json.dumps(provenance))
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["derived_hashes"][str(path.relative_to(tmp_path))] = hashlib.sha256(
        path.read_bytes(),
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(pd, "read_csv", _forbid_csv)
    with pytest.raises(ValueError, match="participant assignment"):
        AuditEvidenceLoader(reference, {}).load()


@pytest.mark.parametrize("mutation", ["duplicate", "target", "fold"])
def test_exact_epoch_join_rejects_duplicate_or_different_population(mutation: str) -> None:
    """Reject ambiguous or mismatched saved epoch identities.

    Args:
        mutation (str): Invalid epoch alignment condition.
    """
    left = pd.DataFrame({"participant_id": ["a"], "time": [0], "target": [2],
                         "fold": ["fold-01"], "prediction_reference": [1]})
    right = left.drop(columns="prediction_reference").assign(prediction_candidate=1)
    if mutation == "duplicate":
        right = pd.concat([right, right])
    elif mutation == "target":
        right["target"] = 3
    else:
        right["fold"] = "fold-02"
    with pytest.raises((ValueError, pd.errors.MergeError)):
        AuditEvidenceLoader._merge_epochs(left, right)


@pytest.mark.parametrize("probabilities", [
    [0.2] * 4, [float("nan"), 0, 0, 1], [0.1, 0.7, 0.1, 0.1],
])
def test_probability_evidence_must_match_normalized_saved_decisions(
    probabilities: list[float],
) -> None:
    """Reject invalid distributions and probabilities inconsistent with the saved argmax.

    Args:
        probabilities (list[float]): Invalid probability vector.
    """
    frame = pd.DataFrame({"participant_id": ["a"], "time": [0], "target": [0],
                          "prediction_reference": [0]})
    raw = pd.DataFrame({"participant_id": ["a"], "time": [0], "target": [0],
                        **{f"probability_{k}": [value] for k, value in enumerate(probabilities)}})
    with pytest.raises(ValueError, match="probabilities"):
        AuditEvidenceLoader._attach_probabilities(
            frame, raw, [f"probability_{k}" for k in range(4)], "reference",
        )


def test_probability_evidence_cannot_be_reordered_onto_other_epochs() -> None:
    """Require exact epoch identity before attaching probabilities to a saved prediction."""
    frame = pd.DataFrame({"participant_id": ["a"], "time": [0], "target": [0],
                          "prediction_reference": [0]})
    raw = pd.DataFrame({"participant_id": ["a"], "time": [30], "target": [0],
                        **{f"probability_{k}": [0.7 if k == 0 else 0.1] for k in range(4)}})
    with pytest.raises(ValueError, match="epoch identities"):
        AuditEvidenceLoader._attach_probabilities(
            frame, raw, [f"probability_{k}" for k in range(4)], "reference",
        )


def test_feature_symlink_cannot_access_reserved_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a development filename redirected to reserved observations before any CSV read.

    Args:
        tmp_path (Path): Temporary grouped root.
        monkeypatch (pytest.MonkeyPatch): Scoped observation-read rejection.
    """
    reference = _write_study(tmp_path)
    feature = tmp_path / "fold-01/features/train/b.csv"
    feature.unlink()
    feature.symlink_to(tmp_path / "reserved.csv")
    monkeypatch.setattr(pd, "read_csv", _forbid_csv)
    monkeypatch.setattr(Path, "open", _guard_reserved_open)
    with pytest.raises(ValueError, match="declared study"):
        AuditEvidenceLoader(reference, {}).load()


def test_feature_checksum_change_is_rejected(tmp_path: Path) -> None:
    """Reject modified prepared values instead of associating predictions with changed features.

    Args:
        tmp_path (Path): Temporary grouped root.
    """
    reference = _write_study(tmp_path)
    feature = tmp_path / "fold-01/features/train/b.csv"
    frame = pd.read_csv(feature)
    frame.loc[0, FEATURE_COLUMNS[0]] = 99.0
    frame.to_csv(feature, index=False)
    with pytest.raises(ValueError, match="checksum"):
        AuditEvidenceLoader(reference, {}).load()
