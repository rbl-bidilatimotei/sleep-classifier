import json
from pathlib import Path
from typing import Any
from unittest.mock import DEFAULT, Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("jsonschema")

from test_bundle_loader import bundle_dir as bundle_dir  # noqa: E402
from test_bundle_loader import checkpoint as checkpoint  # noqa: E402

from sleep_classifier.bundle_loader import EvaluationBundle  # noqa: E402
from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.motion_feature_input import MOTION_FEATURE, MotionFeatureInput  # noqa: E402
from sleep_classifier.raw_event_adapter import LateEventError  # noqa: E402
from sleep_classifier.raw_session_runner import RawSessionRunner  # noqa: E402
from sleep_classifier.streaming_inference import StreamingEpochPredictor  # noqa: E402


@pytest.fixture
def bundle(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> EvaluationBundle:
    """Load a verified synthetic checkpoint and run its real predictor on CPU.

    Args:
        bundle_dir (Path): Synthetic bundle with runtime verification isolated.
        monkeypatch (pytest.MonkeyPatch): Fixture for a portable test device.
    """
    # The version-one bundle manifest requires MPS even on CPU-only test hosts
    monkeypatch.setattr(
        StreamingEpochPredictor, "_resolve_device", Mock(return_value=torch.device("cpu")),
    )
    return EvaluationBundle(bundle_dir)


def _event(event_type: str, **fields: Any) -> dict[str, Any]:
    """Build an input message for the version-one protocol.

    Args:
        event_type (str): Raw session message type.
        **fields (Any): Fields for the selected message type.
    """
    return {"schema_version": "1.0.0", "type": event_type, **fields}


def _start(session_id: str = "session-a") -> dict[str, Any]:
    """Create a new session message for the synthetic participant.

    Args:
        session_id (str): Identifier for the current session.
    """
    return _event(
        "start", session_id=session_id, participant_id="person-a", session_start_seconds=0,
    )


def _feed_epoch(
    runner: RawSessionRunner, start: int, *, motion_offset: float = 0.0, value: float = 1.0,
) -> None:
    """Add observations that meet the moderate coverage rule.

    Args:
        runner (RawSessionRunner): Active session to receive samples.
        start (int): Start of the epoch in seconds.
        motion_offset (float): Offset inside each one-second motion bin.
        value (float): Constant motion value and offset for heart rate.
    """
    for offset in (0, 10, 20):
        event = _event("heart_rate", time=start + offset, heart_rate=60.0 + value)
        assert runner.handle(event) == []
    for offset in range(30):
        assert runner.handle(_event(
            "motion", time=start + offset + motion_offset, x=value, y=0.0, z=0.0,
        )) == []


def _watermark(sensor: str, time: int) -> dict[str, Any]:
    """Declare a new watermark for one sensor.

    Args:
        sensor (str): Sensor whose watermark will move.
        time (int): Watermark timestamp in seconds.
    """
    return _event("watermark", sensor=sensor, time=time)


def _finish(time: int) -> dict[str, Any]:
    """Declare the end of the raw session.

    Args:
        time (int): Exclusive session end in seconds.
    """
    return _event("finish", session_end_seconds=time)


@pytest.mark.parametrize("invalid", [
    _start() | {"session_start_seconds": True},
    _start() | {"participant_id": "  "},
    _start() | {"schema_version": "2.0.0"},
    _event("heart_rate", time=0, heart_rate=60, target=None),
    _event("heart_rate", time=True, heart_rate=60),
    _event("heart_rate", time=0, heart_rate=float("inf")),
    _event("motion", time=0, x=1.0, y=0, z=float("nan")),
    _event("motion", time="0", x=1.0, y=0, z=0),
    _event("watermark", sensor="temperature", time=30),
    _event("finish", session_end_seconds=30, label=0),
])
def test_invalid_events_do_not_change_session(
    bundle: EvaluationBundle, invalid: dict[str, Any],
) -> None:
    """Reject invalid messages while preserving the previous session state.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
        invalid (dict[str, Any]): Raw input that violates the protocol.
    """
    runner = RawSessionRunner(bundle)
    if invalid["type"] != "start":
        runner.handle(_start())
    before = runner.status()
    with pytest.raises(ValueError):
        runner.handle(invalid)
    assert runner.status() == before


def test_requires_explicit_start_and_finish(bundle: EvaluationBundle) -> None:
    """Keep one participant and one explicit lifecycle per runner.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
    """
    runner = RawSessionRunner(bundle)
    assert runner.status()["phase"] == "new"
    with pytest.raises((RuntimeError, ValueError)):
        runner.handle(_watermark("motion", 30))
    started, = runner.handle(_start())
    assert started["type"] == "session_started"
    assert started["participant_id"] == "person-a"
    assert runner.status()["phase"] == "active"
    with pytest.raises((RuntimeError, ValueError)):
        runner.handle(_start("different-session"))

    before = runner.status()
    with pytest.raises((RuntimeError, ValueError)):
        runner.ensure_finished()
    assert runner.status() == before
    finished, = runner.handle(_finish(5))
    assert finished["type"] == "session_finished"
    assert runner.status()["phase"] == "finished"
    runner.ensure_finished()
    for event in (_start(), _watermark("motion", 5), _finish(5)):
        with pytest.raises((RuntimeError, ValueError)):
            runner.handle(event)


def test_buffer_backpressure_duplicates_and_lateness(bundle: EvaluationBundle) -> None:
    """Preserve the raw adapter's buffer capacity and exclusive watermarks.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
    """
    runner = RawSessionRunner(bundle, max_buffered_events=1)
    runner.handle(_start())
    heart = _event("heart_rate", time=0, heart_rate=60)
    motion = _event("motion", time=30, x=1, y=0, z=0)
    assert runner.handle(heart) == []
    assert runner.handle(heart) == []
    before = runner.status()
    with pytest.raises(ValueError):
        runner.handle(heart | {"heart_rate": 61})
    with pytest.raises(BufferError):
        runner.handle(motion)
    assert runner.status() == before
    assert runner.handle(_watermark("heart_rate", 30)) == []
    with pytest.raises(LateEventError):
        runner.handle(heart)

    skipped, = runner.handle(_watermark("motion", 30))
    assert skipped["type"] == "skipped_epoch"
    assert runner.handle(motion) == []
    finished, = runner.handle(_finish(35))
    assert finished["type"] == "session_finished"
    assert finished["duplicate_events"] == 1
    assert finished["peak_buffered_events"] == finished["max_buffered_events"] == 1
    assert finished["discarded_tail_events"] == 1


def test_watermark_and_finish_limits_are_atomic(bundle: EvaluationBundle) -> None:
    """Reject large drains before committing watermarks or discarding samples.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
    """
    runner = RawSessionRunner(bundle, max_epochs_per_event=1)
    runner.handle(_start())
    _feed_epoch(runner, 0)
    before = runner.status()
    with pytest.raises(ValueError):
        runner.handle(_finish(60))
    assert runner.status() == before

    assert runner.handle(_watermark("heart_rate", 60)) == []
    before = runner.status()
    with pytest.raises(ValueError):
        runner.handle(_watermark("motion", 60))
    assert runner.status() == before
    first, = runner.handle(_watermark("motion", 30))
    assert first["type"] == "prediction"
    final_epoch, finished = runner.handle(_finish(60))
    assert [final_epoch["type"], finished["type"]] == ["skipped_epoch", "session_finished"]


def test_future_observations_cannot_change_prefix_or_new_session(bundle: EvaluationBundle) -> None:
    """Keep a closed prediction independent of buffered future data and prior sessions.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
    """
    prefix = RawSessionRunner(bundle)
    buffered = RawSessionRunner(bundle)
    for runner in (prefix, buffered):
        runner.handle(_start())
        _feed_epoch(runner, 0)
    _feed_epoch(buffered, 30, value=15.0)
    for runner in (prefix, buffered):
        assert runner.handle(_watermark("heart_rate", 30)) == []
    first, = prefix.handle(_watermark("motion", 30))
    assert buffered.handle(_watermark("motion", 30)) == [first]

    second, finished = buffered.handle(_finish(60))
    assert second["history_epochs"] == 2
    assert second["reset_reason"] is None
    assert finished["type"] == "session_finished"
    fresh = RawSessionRunner(bundle)
    fresh.handle(_start("session-b"))
    _feed_epoch(fresh, 0)
    repeated, _ = fresh.handle(_finish(30))
    assert repeated["history_epochs"] == 1
    assert repeated["reset_reason"] is None
    assert repeated["probabilities"] == first["probabilities"]


def test_uses_frozen_bundle_and_segments_across_skips(
    bundle: EvaluationBundle, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply saved imputation and preserve frozen prediction across a coverage gap.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
        monkeypatch (pytest.MonkeyPatch): Fixture for observing prediction requests.
    """
    predictor = bundle.predictor()
    integrity = predictor.integrity()
    predict = Mock(wraps=predictor.predict)
    monkeypatch.setattr(predictor, "predict", predict)
    monkeypatch.setattr(bundle, "predictor", Mock(return_value=predictor))
    source = Mock(side_effect=AssertionError("Unexpected raw sensor source read"))
    monkeypatch.setattr(MotionFeatureInput, "_load_sensor", source)
    monkeypatch.setattr(torch, "load", source)

    runner = RawSessionRunner(bundle)
    started, = runner.handle(_start())
    _feed_epoch(runner, 0, motion_offset=0.5)
    duplicate = _event("motion", time=29.5, x=1.0, y=0.0, z=0.0)
    assert runner.handle(duplicate) == []
    _feed_epoch(runner, 30)
    _feed_epoch(runner, 90)
    assert runner.handle(_event("heart_rate", time=121, heart_rate=60)) == []

    assert runner.handle(_watermark("heart_rate", 90)) == []
    first, second, skipped = runner.handle(_watermark("motion", 90))
    resumed, finished = runner.handle(_finish(125))
    assert [row["type"] for row in (first, second, skipped, resumed, finished)] == [
        "prediction", "prediction", "skipped_epoch", "prediction", "session_finished",
    ]
    assert [row["time"] for row in (first, second, skipped, resumed)] == [0, 30, 60, 90]
    assert [row["available_at"] for row in (first, second, skipped, resumed)] == [30, 60, 90, 120]
    assert [row["watermark_at_emission"] for row in (first, second, skipped, resumed)] == [
        90, 90, 90, 125,
    ]
    assert [row["history_epochs"] for row in (first, second, resumed)] == [1, 2, 1]
    assert resumed["reset_reason"] == "segment_change"
    assert [row["segment_id"] for row in (first, second, resumed)] == [0, 0, 1]
    manifest = bundle.manifest
    assert started["checkpoint_sha256"] == manifest["files"]["checkpoint"]["sha256"]
    assert started["coverage_profile"] == "moderate-v1"
    assert started["class_names"] == manifest["model"]["class_names"]
    for row in (started, first, second, skipped, resumed, finished):
        assert row["session_id"] == "session-a"
        assert row["participant_id"] == "person-a"
        assert row["bundle_id"] == manifest["bundle_id"]
        assert row["artifact_role"] == "evaluation_fold"
        json.dumps(row, allow_nan=False)
    for row in (first, second, resumed):
        assert len(row["probabilities"]) == 4
        assert sum(row["probabilities"]) == pytest.approx(1.0)
        assert row["stage"] == started["class_names"][row["prediction"]]
    assert skipped["reason"] == "insufficient_sensor_coverage"
    assert skipped["coverage"]["heart_sample_count"] == 0
    assert skipped["coverage"]["motion_sample_count"] == 0

    assert finished["emitted_epochs"] == 4
    assert finished["predicted_epochs"] == 3
    assert finished["skipped_epochs"] == 1
    assert finished["peak_buffered_events"] == 100
    assert finished["duplicate_events"] == 1
    assert finished["discarded_tail_events"] == 1
    assert finished["integrity"] == integrity | {"weights_unchanged": True}
    requests = [call.args[0] for call in predict.call_args_list]
    assert len(requests) == 3
    assert all(set(item) == {"participant_id", "segment_id", "time", "features"}
               for item in requests)
    assert all(set(item["features"]) == {*FEATURE_COLUMNS, MOTION_FEATURE} for item in requests)
    assert [item["features"][MOTION_FEATURE] for item in requests] == [0.125, 0.0, 0.0]
    assert predictor.integrity() == integrity
    assert integrity["evaluation_mode"] is True
    assert integrity["gradients_created"] is False
    source.assert_not_called()


def test_failure_during_drain_poison_session(
    bundle: EvaluationBundle, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject continuation after model failure following a successful prediction.

    Args:
        bundle (EvaluationBundle): Verified synthetic model and preprocessing.
        monkeypatch (pytest.MonkeyPatch): Fixture for simulating model failure.
    """
    predictor = bundle.predictor()
    predict = Mock(wraps=predictor.predict, side_effect=[DEFAULT, RuntimeError("model failure")])
    monkeypatch.setattr(predictor, "predict", predict)
    monkeypatch.setattr(bundle, "predictor", Mock(return_value=predictor))

    runner = RawSessionRunner(bundle)
    runner.handle(_start())
    _feed_epoch(runner, 0)
    _feed_epoch(runner, 30)
    with pytest.raises(RuntimeError, match="model failure"):
        runner.handle(_finish(60))
    assert predict.call_count == 2
    assert runner.status()["phase"] == "failed"
    assert runner.status()["predicted_epochs"] == 1
    with pytest.raises((RuntimeError, ValueError)):
        runner.ensure_finished()
    with pytest.raises((RuntimeError, ValueError)):
        runner.handle(_finish(60))
    assert predict.call_count == 2
