from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from sleep_classifier import service_sessions
from sleep_classifier.service_config import ServiceConfig
from sleep_classifier.service_sessions import ServiceError, SessionService


class FakeClock:
    """Provide controllable monotonic time without sleeping."""

    def __init__(self) -> None:
        """Start the synthetic clock at zero."""
        self.now = 0.0


    def __call__(self) -> float:
        """Return the synthetic current time."""
        return self.now


class FakeRunner:
    """Record accepted events without loading model weights or observations."""

    created: list["FakeRunner"] = []
    fail_start = False

    def __init__(self, bundle: Any, **limits: int) -> None:
        """Retain the injected bundle and constructor limits.

        Args:
            bundle (Any): Synthetic verified-bundle substitute.
            **limits (int): Runner resource limits.
        """
        self.bundle = bundle
        self.limits = limits
        self.phase = "new"
        self.events: list[dict[str, Any]] = []
        self.created.append(self)


    def handle(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Accept events or simulate a specific safe or terminal failure.

        Args:
            event (dict[str, Any]): Validated synthetic event.
        """
        if event["type"] == "start" and self.fail_start:
            raise ValueError("private startup failure")
        if event.get("time") == 999:
            raise BufferError("private buffer payload")
        if event.get("time") == 998:
            raise ValueError("private event payload")
        if event.get("time") == 997:
            self.phase = "failed"
            raise ValueError("private failed inference payload")
        if event.get("time") == 996:
            raise RuntimeError("private backend failure")
        self.events.append(event)
        self.phase = "finished" if event["type"] == "finish" else "active"
        return [{"type": "session_finished"}] if self.phase == "finished" else []


    def status(self) -> dict[str, str]:
        """Return the current fake runner phase."""
        return {"phase": self.phase}


@pytest.fixture
def setup_service(monkeypatch: pytest.MonkeyPatch) -> tuple[SessionService, FakeClock, Mock]:
    """Create a loaded service with synthetic dependencies and no device operations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped runner and device substitutions.
    """
    FakeRunner.created = []
    FakeRunner.fail_start = False
    monkeypatch.setattr(service_sessions, "RawSessionRunner", FakeRunner)
    monkeypatch.setattr(service_sessions.StreamingEpochPredictor, "_resolve_device", Mock())
    settings = ServiceConfig(bundle_dir=Path("/synthetic-bundle"), max_buffered_events=8,
                               max_epochs_per_event=2, max_session_events=8,
                               idle_timeout_seconds=10.0)
    factory = Mock(return_value=SimpleNamespace(
        manifest={"runtime": {"validated_device": "mps"}},
    ))
    clock = FakeClock()
    service = SessionService(settings, bundle_factory=factory, clock=clock)
    service.load()
    return service, clock, factory


def start_event() -> dict[str, Any]:
    """Return the first version-one session event."""
    return {"schema_version": "1.0.0", "type": "start", "session_id": "session-a",
            "participant_id": "synthetic-person", "session_start_seconds": 0}


def sensor_event(time: int = 0) -> dict[str, Any]:
    """Return one synthetic heart-rate observation.

    Args:
        time (int): Synthetic sample timestamp.
    """
    return {"schema_version": "1.0.0", "type": "heart_rate", "time": time, "heart_rate": 60}


def finish_event() -> dict[str, Any]:
    """Return an explicit empty-session finish event."""
    return {"schema_version": "1.0.0", "type": "finish", "session_end_seconds": 0}


def test_lifecycle_uses_fresh_runner_and_releases_after_finish(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Prevent cross-session prediction history or owner-token reuse.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, factory = setup_service
    first = service.submit(start_event(), None, 0)
    token = first["session_token"]
    assert len(token) >= 32 and first["next_sequence"] == 1
    assert FakeRunner.created[-1].limits == {"max_buffered_events": 8, "max_epochs_per_event": 2}
    assert service.submit(sensor_event(), token, 1)["next_sequence"] == 2
    done = service.submit(finish_event(), token, 2)
    assert done["records"] == [{"type": "session_finished"}]
    with pytest.raises(ServiceError, match="no_active_session"):
        service.submit(sensor_event(), token, 3)
    second = service.submit(start_event(), None, 0)
    assert second["session_token"] != token
    assert len(FakeRunner.created) == 2
    assert FakeRunner.created[-1].events == [start_event()]
    factory.assert_called_once_with(Path("/synthetic-bundle"))


@pytest.mark.parametrize("token,sequence,code", [
    (None, 1, "invalid_owner"), ("wrong", 1, "invalid_owner"),
    ("owner", 0, "sequence_conflict"), ("owner", 2, "sequence_conflict"),
])
def test_wrong_owner_and_sequence_preserve_state(
    setup_service: tuple[SessionService, FakeClock, Mock], token: str | None,
    sequence: int, code: str,
) -> None:
    """Reject unauthorized or repeated input before it reaches the runner.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
        token (str | None): Claimed owner token or the owner placeholder.
        sequence (int): Claimed event sequence.
        code (str): Expected safe error identifier.
    """
    service, _, _ = setup_service
    owner = service.submit(start_event(), None, 0)["session_token"]
    attempted = owner if token == "owner" else token
    with pytest.raises(ServiceError, match=code):
        service.submit(sensor_event(), attempted, sequence)
    assert len(FakeRunner.created[-1].events) == 1
    assert service.submit(sensor_event(), owner, 1)["next_sequence"] == 2


@pytest.mark.parametrize("token,sequence,code", [
    ("unexpected", 0, "invalid_start"), (None, 1, "invalid_start"),
    (None, True, "invalid_sequence"), (None, -1, "invalid_sequence"),
    ("\u2603", 0, "invalid_token"), ("x" * 129, 0, "invalid_token"),
])
def test_invalid_start_headers_do_not_construct_a_runner(
    setup_service: tuple[SessionService, FakeClock, Mock], token: str | None,
    sequence: int, code: str,
) -> None:
    """Require absent owner credentials and sequence zero for a new session.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
        token (str | None): Unsupported owner value.
        sequence (int): Unsupported start sequence.
        code (str): Expected safe error identifier.
    """
    service, _, _ = setup_service
    with pytest.raises(ServiceError, match=code):
        service.submit(start_event(), token, sequence)
    assert not FakeRunner.created


def test_active_start_and_invalid_event_keep_owner_state(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Reject a competing start and labels without replacing the current owner.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, _ = setup_service
    token = service.submit(start_event(), None, 0)["session_token"]
    with pytest.raises(ServiceError, match="session_active"):
        service.submit(start_event(), None, 0)
    with pytest.raises(ServiceError, match="invalid_event"):
        service.submit(sensor_event() | {"label": 1}, token, 1)
    assert service.submit(sensor_event(), token, 1)["next_sequence"] == 2


def test_event_limit_still_allows_explicit_finish(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Bound session input while retaining an orderly close operation.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, _ = setup_service
    service._settings = replace(service._settings, max_session_events=2)
    token = service.submit(start_event(), None, 0)["session_token"]
    service.submit(sensor_event(), token, 1)
    with pytest.raises(ServiceError, match="session_event_limit") as caught:
        service.submit(sensor_event(1), token, 2)
    assert caught.value.status_code == 429
    assert service.submit(finish_event(), token, 2)["next_sequence"] == 3


@pytest.mark.parametrize("time,status,code", [
    (999, 429, "buffer_limit"), (998, 422, "event_rejected"),
    (997, 500, "session_failed"), (996, 500, "session_failed"),
])
def test_runner_errors_preserve_safe_retries_or_release_terminal_state(
    setup_service: tuple[SessionService, FakeClock, Mock], time: int, status: int, code: str,
) -> None:
    """Distinguish rejected events from failures after irreversible prediction progress.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
        time (int): Fake runner failure selector.
        status (int): Expected transport status.
        code (str): Expected safe error identifier.
    """
    service, _, _ = setup_service
    token = service.submit(start_event(), None, 0)["session_token"]
    with pytest.raises(ServiceError, match=code) as caught:
        service.submit(sensor_event(time), token, 1)
    assert caught.value.status_code == status
    assert "private" not in str(caught.value)
    if status == 500:
        with pytest.raises(ServiceError, match="no_active_session"):
            service.submit(sensor_event(), token, 1)
        service.submit(start_event(), None, 0)
    else:
        assert service.submit(sensor_event(), token, 1)["next_sequence"] == 2


def test_expiry_uses_last_accepted_event_and_readiness_checks(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Do not extend ownership when rejected events arrive before idle expiry.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, clock, _ = setup_service
    token = service.submit(start_event(), None, 0)["session_token"]
    clock.now = 8
    with pytest.raises(ServiceError, match="invalid_owner"):
        service.submit(sensor_event(), "wrong", 1)
    clock.now = 10
    assert service.readiness() == {"ready": True, "reason": None}
    with pytest.raises(ServiceError, match="no_active_session"):
        service.submit(sensor_event(), token, 1)
    service.submit(start_event(), None, 0)


def test_accepted_events_refresh_idle_expiry(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Keep a recently active owner until its new idle deadline.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, clock, _ = setup_service
    token = service.submit(start_event(), None, 0)["session_token"]
    clock.now = 8
    service.submit(sensor_event(), token, 1)
    clock.now = 17
    service.submit(sensor_event(1), token, 2)
    clock.now = 27
    with pytest.raises(ServiceError, match="no_active_session"):
        service.submit(sensor_event(2), token, 3)


def test_busy_requests_do_not_queue_or_disrupt_readiness(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Reject concurrent state mutations without reporting an unavailable bundle.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, _ = setup_service
    service._lock.acquire()
    try:
        with pytest.raises(ServiceError, match="session_busy") as caught:
            service.submit(start_event(), None, 0)
        assert caught.value.status_code == 409
        assert service.readiness()["ready"]
    finally:
        service._lock.release()
    assert not FakeRunner.created


def test_missing_bundle_and_close_are_not_ready(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Do not invent predictions when no compatible bundle is configured.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, factory = setup_service
    service.close()
    assert service.readiness() == {"ready": False, "reason": "no_bundle"}
    with pytest.raises(ServiceError, match="not_ready") as caught:
        service.submit(start_event(), None, 0)
    assert caught.value.status_code == 503
    service._settings = replace(service._settings, bundle_dir=None)
    service.load()
    factory.assert_called_once()


@pytest.mark.parametrize("failure", ["load", "device"])
def test_bundle_or_device_failure_is_closed_and_redacted(
    setup_service: tuple[SessionService, FakeClock, Mock], monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, failure: str,
) -> None:
    """Retain strict loading and device checks without exposing private errors.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
        monkeypatch (pytest.MonkeyPatch): Scoped device substitution.
        caplog (pytest.LogCaptureFixture): Captured operational logs.
        failure (str): Startup stage to fail.
    """
    service, _, factory = setup_service
    if failure == "load":
        factory.side_effect = ValueError("private path and sensor payload")
    else:
        monkeypatch.setattr(service_sessions.StreamingEpochPredictor, "_resolve_device",
                            Mock(side_effect=ValueError("private unavailable device")))
    service.load()
    assert service.readiness() == {"ready": False, "reason": "bundle_unavailable"}
    assert "private" not in caplog.text
    with pytest.raises(ServiceError, match="not_ready"):
        service.submit(start_event(), None, 0)


def test_start_restoration_failure_releases_the_slot(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Allow a fresh session after failed model restoration without retaining its owner.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, _ = setup_service
    FakeRunner.fail_start = True
    with pytest.raises(ServiceError, match="session_failed"):
        service.submit(start_event(), None, 0)
    FakeRunner.fail_start = False
    assert service.submit(start_event(), None, 0)["next_sequence"] == 1


@pytest.mark.parametrize("token,sequence,code", [
    ("wrong", 1, "invalid_owner"), ("owner", 0, "sequence_conflict"),
    ("owner", 2, "sequence_conflict"),
])
def test_cancel_requires_owner_and_next_sequence(
    setup_service: tuple[SessionService, FakeClock, Mock], token: str, sequence: int, code: str,
) -> None:
    """Preserve session state after a rejected cancellation.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
        token (str): Claimed owner token or the owner placeholder.
        sequence (int): Claimed cancellation sequence.
        code (str): Expected safe error identifier.
    """
    service, _, _ = setup_service
    owner = service.submit(start_event(), None, 0)["session_token"]
    attempted = owner if token == "owner" else token
    with pytest.raises(ServiceError, match=code):
        service.cancel(attempted, sequence)
    assert service.submit(sensor_event(), owner, 1)["next_sequence"] == 2
    assert service.cancel(owner, 2) == {"status": "cancelled"}
    with pytest.raises(ServiceError, match="no_active_session"):
        service.submit(sensor_event(), owner, 2)
    assert service.submit(start_event(), None, 0)["session_token"] != owner


def test_cancel_does_not_emit_a_completion_record(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Release the slot without pretending an aborted session passed integrity checks.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, _, _ = setup_service
    token = service.submit(start_event(), None, 0)["session_token"]
    runner = FakeRunner.created[-1]
    assert service.cancel(token, 1) == {"status": "cancelled"}
    assert runner.events == [start_event()]


def test_cancel_checks_idle_expiry_and_busy_mutation(
    setup_service: tuple[SessionService, FakeClock, Mock],
) -> None:
    """Reject queued cancellation and expired owner tokens.

    Args:
        setup_service (tuple[SessionService, FakeClock, Mock]): Loaded synthetic service.
    """
    service, clock, _ = setup_service
    token = service.submit(start_event(), None, 0)["session_token"]
    service._lock.acquire()
    try:
        with pytest.raises(ServiceError, match="session_busy"):
            service.cancel(token, 1)
    finally:
        service._lock.release()
    clock.now = 10
    with pytest.raises(ServiceError, match="no_active_session"):
        service.cancel(token, 1)


def test_missing_bundle_startup_explains_readiness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explain the default not-ready state without trying to load a research model.

    Args:
        caplog (pytest.LogCaptureFixture): Captured startup diagnostics.
    """
    factory = Mock()
    service = SessionService(ServiceConfig(), bundle_factory=factory)
    service.load()
    factory.assert_not_called()
    assert service.readiness() == {"ready": False, "reason": "no_bundle"}
    assert "No evaluation bundle configured" in caplog.text
    assert "SLEEP_SERVICE_CONFIG" in caplog.text
