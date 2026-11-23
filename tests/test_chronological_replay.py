import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sleep_classifier.chronological_replay import (  # noqa: E402
    ChronologicalReplayAudit,
    ChronologicalReplayStudy,
)
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.model_factory import ModelFactory  # noqa: E402
from sleep_classifier.motion_feature_input import MOTION_DEFINITION, MOTION_FEATURE  # noqa: E402
from sleep_classifier.sequences import SequenceBatchCollator  # noqa: E402
from sleep_classifier.streaming_inference import StreamingEpochPredictor  # noqa: E402

HISTORY_LENGTHS = [1, 2, 3, 3, 1, 2, 1, 2]
FEATURES = (*FEATURE_COLUMNS, MOTION_FEATURE)
TOLERANCE = 2e-6


def _digest(path: Path) -> str:
    """Hash a synthetic source or output artifact.

    Args:
        path (Path): File to hash.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    """Record the file set and digests in the temporary reference.

    Args:
        root (Path): Directory containing synthetic source artifacts.
    """
    return {str(path.relative_to(root)): _digest(path) for path in root.rglob("*")
            if path.is_file()}


@pytest.fixture
def fold_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Save an unfitted synthetic checkpoint with independently windowed predictions.

    Args:
        tmp_path (Path): Temporary root for the reference and mock motion source.
        monkeypatch (pytest.MonkeyPatch): Replace prepared-row and motion-snapshot reads.
    """
    fold = tmp_path / "reference" / "fold-01"
    results = fold / "results"
    results.mkdir(parents=True)
    frame = pd.DataFrame({
        name: np.arange(8, dtype=float) + position
        for position, name in enumerate(FEATURE_COLUMNS)
    }).assign(
        participant_id=["p1"] * 6 + ["p2"] * 2,
        segment_id=[0] * 4 + [1] * 2 + [0] * 2,
        time=[0.0, 30.0, 60.0, 90.0, 150.0, 180.0, 0.0, 30.0],
        target=[0, 1, 2, 3] * 2,
        eligible=True,
    )
    loaded = {"partitions": {"validation": frame.iloc[::-1]}}
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.PreparedFeatureDataset",
                        Mock(return_value=Mock(load=Mock(return_value=loaded))))

    motion = {"name": "fixed_lag_motion", "audit_config": "motion-source.yaml"}
    audit_source = fold / "motion-source.yaml"
    audit_source.write_text("{}\n")
    audit_manifest = fold / "motion-manifest.json"
    audit_manifest.write_text("{}\n")
    (fold / "gru.yaml").write_text(yaml.safe_dump({"motion_features": motion}))
    sensor = frame[["participant_id", "time"]].assign(
        **{MOTION_FEATURE: [0.5, 1.0, np.nan, 4.2, 3.0, 2.2, np.nan, 2.0]},
    )
    people = {"development_participants": ["p1", "p2"], "reserved_test_participants": ["held"]}
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.MotionFeatureInput._load_sensor",
                        Mock(return_value=(sensor, people, audit_manifest)))
    motion_state = {
        "definition": MOTION_DEFINITION, "audit_config_sha256": _digest(audit_source),
        "snapshot_manifest_sha256": _digest(audit_manifest), "missing_policy": "training_mean",
        "scaling_fit": "training_targets_only", "imputation_mean": 2.5,
    }
    checkpoint, probabilities = _offline_predictions(frame, sensor, motion_state)
    torch.save(checkpoint, results / "model_checkpoint.pt")
    predictions = frame[["participant_id", "time", "target"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    for stage in range(4):
        predictions[f"probability_{stage}"] = probabilities[:, stage]
    predictions.iloc[::-1].to_csv(results / "validation_predictions.csv", index=False)
    (results / "metrics.json").write_text("{}\n")
    return fold


def _offline_predictions(
    frame: pd.DataFrame, sensor: pd.DataFrame, motion_state: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate a tiny patch Transformer on manually indexed padded windows.

    Args:
        frame (pd.DataFrame): Ordered synthetic feature rows and target identities.
        sensor (pd.DataFrame): Aligned motion values with missing entries.
        motion_state (dict[str, Any]): Saved motion imputation and source identity.
    """
    factory = ModelFactory({
        "model_type": "patch_transformer", "context_epochs": 3, "patch_size": 2,
        "transformer": {"d_model": 4, "nhead": 2, "feedforward_dim": 8, "dropout": 0.2},
        "motion_features": MOTION_DEFINITION,
        "window_features": {
            "name": "heart_rate_deviation", "history_epochs": 2, "min_history_epochs": 1,
        },
    })
    scales = [2.0 + index for index in range(len(FEATURES))]
    means = [0.5 * index for index in range(len(FEATURES))]
    window_state = {
        "schema_version": 1, "heart_rate_scale": scales[FEATURES.index("heart_rate_mean")],
        "delta_mean": 0.5, "delta_scale": 2.0, "available_training_targets": 12,
        "scaling_fit": "training_targets_only",
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(63)
        model = factory.build(len(FEATURES), 4, window_state).eval()
    checkpoint = {
        "state_dict": model.state_dict(), "model_type": factory.model_type,
        "model_config": factory.checkpoint_config(len(FEATURES), 4), "features": list(FEATURES),
        "scaler_mean": means, "scaler_scale": scales, "context_epochs": 3,
        "motion_features": motion_state, "window_feature_state": window_state,
    }
    motion_values = sensor[MOTION_FEATURE].fillna(motion_state["imputation_mean"])
    raw = frame.assign(**{MOTION_FEATURE: motion_values}).loc[:, FEATURES].to_numpy()
    scaled = torch.tensor((raw - means) / scales, dtype=torch.float32)
    rows = [
        {"features": scaled[index + 1 - length:index + 1],
         "target": torch.tensor(frame.target.iloc[index]), "time": frame.time.iloc[index],
         "participant_id": frame.participant_id.iloc[index]}
        for index, length in enumerate(HISTORY_LENGTHS)
    ]
    batch = SequenceBatchCollator()(rows)
    with torch.inference_mode():
        probabilities = model(batch["features"], batch["lengths"]).softmax(dim=1).numpy()
    return checkpoint, probabilities


@pytest.fixture
def replay_study(
    fold_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> ChronologicalReplayStudy:
    """Register one completed synthetic development fold with immutable source files.

    Args:
        fold_dir (Path): Saved synthetic fold directory.
        tmp_path (Path): Temporary root for the replay config and output.
        monkeypatch (pytest.MonkeyPatch): Bind a synthetic grouped reference manifest.
    """
    config = tmp_path / "replay.yaml"
    config.write_text(yaml.safe_dump({
        "reference_grouped_config": "reference.yaml", "output_dir": "audit", "device": "cpu",
        "absolute_tolerance": TOLERANCE,
    }))
    plan = {
        "folds": [{"name": fold_dir.name}], "development": ["p1", "p2"], "reserved_test": ["held"],
    }
    manifest = {"plan": plan, "device": "cpu"}
    (fold_dir.parent / "manifest.json").write_text(json.dumps(manifest))
    metrics = fold_dir / "results/metrics.json"
    summary = {
        "status": "complete", "plan": plan,
        "results": [{"fold": fold_dir.name, "metrics_path": str(metrics),
                     "metrics_sha256": _digest(metrics)}],
    }
    (fold_dir.parent / "summary.json").write_text(json.dumps(summary))
    reference = Mock(
        output_dir=fold_dir.parent, _validate_manifest=Mock(return_value=manifest),
        model_settings={"device": "cpu", "seed": 42, "epochs": 1, "batch_size": 8,
                        "context_epochs": 3, "patience": 1, "learning_rate": 0.001,
                        "gradient_clip": 1.0, "hidden_size": 3},
    )
    monkeypatch.setattr("sleep_classifier.chronological_replay.GroupedValidationExperiment",
                        Mock(return_value=reference))
    return ChronologicalReplayStudy(config)


def test_replay_matches_offline_and_saved_probabilities(
    fold_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match saved predictions across warm-up, context eviction, and prepared boundaries.

    Args:
        fold_dir (Path): Synthetic checkpoint and offline predictions.
        monkeypatch (pytest.MonkeyPatch): Reject any transform fitting in the audit.
    """
    failed_fit = Mock(side_effect=AssertionError("Replay attempted to fit a transform"))
    monkeypatch.setattr("sleep_classifier.checkpoint_audit.StandardScaler.fit", failed_fit)
    monkeypatch.setattr("sleep_classifier.window_heart_rate.WindowLocalHeartRate.fit", failed_fit)
    before = _source_hashes(fold_dir)
    audit = ChronologicalReplayAudit(fold_dir, "cpu", TOLERANCE)
    report = audit.build()

    assert report["status"] == "passed"
    assert (report["epochs"], report["participants"]) == (8, 2)
    assert report["features"] == list(FEATURES)
    for key in ("offline_vs_saved", "stream_vs_offline"):
        assert report[key]["max_abs_probability_difference"] <= TOLERANCE
        assert report[key]["prediction_disagreements"] == 0
    assert audit.predictions.history_epochs.tolist() == HISTORY_LENGTHS
    assert report["history_lengths_match"]
    assert report["history_start_epochs"] == 3
    assert report["full_context_epochs"] == 2
    assert report["reset_counts"] == {"segment_change": 1, "participant_change": 1}
    assert report["weights_unchanged"] and not report["gradients_created"]
    assert not report["training_performed"]
    assert _source_hashes(fold_dir) == before
    failed_fit.assert_not_called()


def test_replay_accepts_equivalent_csv_timestamp_types(fold_dir: Path) -> None:
    """Accept identical times serialized as integers in the saved predictions.

    Args:
        fold_dir (Path): Synthetic fold with floating-point prepared timestamps.
    """
    path = fold_dir / "results/validation_predictions.csv"
    saved = pd.read_csv(path)
    saved["time"] = saved.time.astype(int)
    saved.to_csv(path, index=False)

    assert ChronologicalReplayAudit(fold_dir, "cpu", TOLERANCE).build()["status"] == "passed"


def test_stream_resets_when_epoch_gap_has_no_segment_change(fold_dir: Path) -> None:
    """Match a clean session after a gap that arrives under the same segment key.

    Args:
        fold_dir (Path): Synthetic saved checkpoint directory.
    """
    predictor = StreamingEpochPredictor(fold_dir / "results/model_checkpoint.pt")
    rows = {column: 1.0 for column in predictor.feature_columns}
    predictor.predict_epoch("p1", 0, 0, rows)
    assert predictor.predict_epoch("p1", 0, 30, rows)["history_epochs"] == 2
    after_gap = predictor.predict_epoch("p1", 0, 90, rows)
    predictor.reset()
    fresh = predictor.predict_epoch("p1", 0, 90, rows)

    assert after_gap == fresh | {"history_epochs": 1, "reset_reason": "time_gap"}


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("target", 0, "differ from the replay validation identities"),
        ("prediction", -1, "probabilities or class predictions are invalid"),
        ("probability_0", -0.1, "probabilities or class predictions are invalid"),
        ("probability_0", float("nan"), "probabilities or class predictions are invalid"),
    ],
)
def test_replay_rejects_corrupt_saved_predictions(
    fold_dir: Path, column: str, value: int | float, message: str,
) -> None:
    """Reject invalid target identities and malformed probability evidence.

    Args:
        fold_dir (Path): Synthetic saved checkpoint and predictions.
        column (str): Evidence field to corrupt.
        value (int | float): Replacement field value.
        message (str): Expected failure reason.
    """
    path = fold_dir / "results/validation_predictions.csv"
    predictions = pd.read_csv(path)
    predictions.loc[0, column] = value
    predictions.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        ChronologicalReplayAudit(fold_dir, "cpu", TOLERANCE).build()


def test_replay_fails_when_saved_confidence_changes_without_a_class_change(fold_dir: Path) -> None:
    """Require numerical parity even when all selected classes agree.

    Args:
        fold_dir (Path): Synthetic saved checkpoint and predictions.
    """
    path = fold_dir / "results/validation_predictions.csv"
    predictions = pd.read_csv(path)
    selected = int(predictions.loc[0, "prediction"])
    for stage in range(4):
        predictions.loc[0, f"probability_{stage}"] = 0.91 if stage == selected else 0.03
    predictions.to_csv(path, index=False)

    report = ChronologicalReplayAudit(fold_dir, "cpu", TOLERANCE).build()

    assert report["status"] == "failed"
    assert report["offline_vs_saved"]["prediction_disagreements"] == 0
    assert report["offline_vs_saved"]["max_abs_probability_difference"] > TOLERANCE


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"absolute_tolerance": 0}, "tolerance"),
        ({"absolute_tolerance": True}, "tolerance"),
        ({"absolute_tolerance": 0.001}, "tolerance"),
        ({"absolute_tolerance": float("inf")}, "tolerance"),
        ({"absolute_tolerance": "2e-6"}, "tolerance"),
        ({"device": "auto"}, "explicit backend"),
        ({"unused": 1}, "supported audit settings"),
        ({"output_dir": "reference/overwrite"}, "separate from the saved model"),
    ],
)
def test_replay_study_rejects_invalid_settings(
    replay_study: ChronologicalReplayStudy, patch: dict[str, Any], message: str,
) -> None:
    """Validate the audit policy before creating output.

    Args:
        replay_study (ChronologicalReplayStudy): Isolated synthetic replay study.
        patch (dict[str, Any]): Invalid setting to merge into the replay config.
        message (str): Expected failure reason.
    """
    replay_study.config_path.write_text(yaml.safe_dump(replay_study.settings | patch))

    with pytest.raises(ValueError, match=message):
        ChronologicalReplayStudy(replay_study.config_path)
    assert not replay_study.output_dir.exists()


def test_run_preserves_sources_and_reads_persisted_evidence(
    replay_study: ChronologicalReplayStudy,
) -> None:
    """Write one fresh audit and preserve that snapshot on subsequent run attempts.

    Args:
        replay_study (ChronologicalReplayStudy): Isolated synthetic replay study.
    """
    source = _source_hashes(replay_study.reference.output_dir)
    assert replay_study.read_report()["status"] == "pending"
    destination = replay_study.run()
    report = replay_study.read_report()
    assert destination == replay_study.output_dir / "report.json"
    assert report["status"] == "passed"
    assert not report["test_observations_read"] and not report["training_performed"]
    assert report["plan"]["folds"] == ["fold-01"]
    predictions = pd.read_csv(replay_study.output_dir / "fold-01.csv")
    assert predictions.history_epochs.tolist() == HISTORY_LENGTHS
    assert set(predictions.participant_id) == {"p1", "p2"}
    outputs = _source_hashes(replay_study.output_dir)

    with pytest.raises(FileExistsError, match="exists"):
        replay_study.run()
    assert _source_hashes(replay_study.reference.output_dir) == source
    assert _source_hashes(replay_study.output_dir) == outputs


def test_unavailable_device_fails_before_output(
    replay_study: ChronologicalReplayStudy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block an unavailable reference backend before creating the snapshot.

    Args:
        replay_study (ChronologicalReplayStudy): Isolated synthetic replay study.
        monkeypatch (pytest.MonkeyPatch): Make CUDA unavailable on every test host.
    """
    replay_study.settings["device"] = "cuda"
    replay_study.reference.model_settings["device"] = "cuda"
    replay_study.reference._validate_manifest.return_value["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        replay_study.run()
    assert not replay_study.output_dir.exists()


def test_run_rejects_changed_registered_metrics(replay_study: ChronologicalReplayStudy) -> None:
    """Detect a registered metrics change before creating output.

    Args:
        replay_study (ChronologicalReplayStudy): Isolated synthetic replay study.
    """
    metrics = replay_study.reference.output_dir / "fold-01/results/metrics.json"
    metrics.write_text('{"changed": true}\n')

    with pytest.raises(ValueError, match="metrics hash differs"):
        replay_study.run()
    assert not replay_study.output_dir.exists()


@pytest.mark.parametrize("tamper_source", [True, False])
def test_read_report_rejects_changed_artifacts(
    replay_study: ChronologicalReplayStudy, tamper_source: bool,
) -> None:
    """Reject a cached report when a source file or replay CSV changes.

    Args:
        replay_study (ChronologicalReplayStudy): Isolated synthetic replay study.
        tamper_source (bool): Modify the source CSV instead of the replay CSV.
    """
    replay_study.run()
    if tamper_source:
        path = replay_study.reference.output_dir / "fold-01/results/validation_predictions.csv"
        message = "changed inputs or settings"
    else:
        path = replay_study.output_dir / "fold-01.csv"
        message = "replay predictions changed"
    with path.open("a") as output:
        output.write("\n")

    with pytest.raises(ValueError, match=message):
        replay_study.read_report()


@pytest.mark.parametrize("copies", [0, 2])
def test_read_report_requires_each_planned_fold_once(
    replay_study: ChronologicalReplayStudy, copies: int,
) -> None:
    """Reject cached evidence with missing or duplicated fold entries.

    Args:
        replay_study (ChronologicalReplayStudy): Isolated synthetic replay study.
        copies (int): Number of copies of the planned fold to retain in the report.
    """
    path = replay_study.run()
    report = json.loads(path.read_text())
    report["folds"] *= copies
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="fold"):
        replay_study.read_report()
