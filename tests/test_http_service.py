import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from fastapi.testclient import TestClient
from test_bundle_loader import bundle_dir as bundle_dir
from test_bundle_loader import checkpoint as checkpoint
from test_raw_session_runner import bundle as bundle

from sleep_classifier.bundle_export import BundleExportConfig
from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.http_service import create_app
from sleep_classifier.raw_session_runner import RawSessionRunner
from sleep_classifier.service_config import ServiceConfig
from sleep_classifier.service_sessions import ServiceError, SessionService


def config_file(tmp_path: Path, **changes: Any) -> Path:
    """Create a small operator-owned settings file.

    Args:
        tmp_path (Path): Isolated test directory.
        **changes (Any): Settings to replace or extend.
    """
    settings = {"bundle_dir": None, "max_buffered_events": 8192,
                "max_epochs_per_event": 128, "max_line_bytes": 4096,
                "max_session_events": 100, "idle_timeout_seconds": 30}
    settings.update(changes)
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_empty_service_is_live_but_not_ready() -> None:
    """Keep the HTTP service available without claiming a model was loaded."""
    service = SessionService(ServiceConfig())
    with TestClient(create_app(service=service)) as client:
        assert client.get("/healthz").json() == {"status": "alive"}
        readiness = client.get("/readyz")
        assert readiness.status_code == 503
        assert readiness.json() == {"ready": False, "reason": "no_bundle"}
        event = {"schema_version": "1.0.0", "type": "start", "session_id": "demo",
                 "participant_id": "demo", "session_start_seconds": 0}
        response = client.post("/v1/events", json=event, headers={"x-event-sequence": "0"})
        assert response.status_code == 503 and response.json() == {"error": "not_ready"}


@pytest.fixture
def recording_service() -> Mock:
    """Return a transport fake that never restores weights or reads observations."""
    service = Mock(spec=SessionService)
    service.readiness.return_value = {"ready": True, "reason": None}
    service.submit.return_value = {"session_token": "test-token", "next_sequence": 1,
                                   "records": [{"type": "session_started"}]}
    return service


def test_lifespan_and_event_headers(recording_service: Mock) -> None:
    """Pass ownership and sequencing to the session owner and close it on shutdown.

    Args:
        recording_service (Mock): In-memory service implementation.
    """
    app = create_app(service=recording_service)
    with TestClient(app) as client:
        recording_service.load.assert_called_once()
        assert client.get("/readyz").status_code == 200
        event = {"schema_version": "1.0.0", "type": "finish", "session_end_seconds": 30}
        response = client.post("/v1/events", json=event,
                               headers={"x-event-sequence": "7", "x-session-token": "owner"})
        assert response.status_code == 200
        recording_service.submit.assert_called_once_with(event, "owner", 7)
    recording_service.close.assert_called_once()


@pytest.mark.parametrize("sequence", [None, "-1", "+1", "01", "1.0", "1"*17, ""])
def test_bad_sequence_is_rejected_before_session_mutation(
    recording_service: Mock, sequence: str | None,
) -> None:
    """Reject noncanonical sequencing before dispatching an event.

    Args:
        recording_service (Mock): In-memory service implementation.
        sequence (str | None): Invalid or absent event sequence.
    """
    headers = {} if sequence is None else {"x-event-sequence": sequence}
    with TestClient(create_app(service=recording_service)) as client:
        assert client.post("/v1/events", json={}, headers=headers).status_code == 400
        recording_service.submit.assert_not_called()


@pytest.mark.parametrize("payload,content_type,status", [
    ('{"type":"motion","type":"finish"}', "application/json", 422),
    ('{"time":NaN}', "application/json", 422),
    ('[]', "application/json", 422), ('{', "application/json", 422),
    ('{}', "text/plain", 415), ('x'*4097, "application/json", 413),
])
def test_request_bounds_and_strict_json(
    recording_service: Mock, payload: str, content_type: str, status: int,
) -> None:
    """Reject malformed and oversized bodies before session mutation.

    Args:
        recording_service (Mock): In-memory service implementation.
        payload (str): Untrusted HTTP body.
        content_type (str): Declared request content type.
        status (int): Expected rejection status.
    """
    with TestClient(create_app(service=recording_service)) as client:
        response = client.post("/v1/events", content=payload,
                               headers={"x-event-sequence": "0", "content-type": content_type})
        assert response.status_code == status
        recording_service.submit.assert_not_called()


def test_chunked_request_obeys_body_limit(recording_service: Mock) -> None:
    """Reject an oversized chunked body without relying on Content-Length.

    Args:
        recording_service (Mock): In-memory service implementation.
    """
    with TestClient(create_app(service=recording_service)) as client:
        response = client.post("/v1/events", content=iter([b" "*2048, b" "*2049]),
                               headers={"x-event-sequence": "0",
                                        "content-type": "application/json"})
        assert response.status_code == 413
        recording_service.submit.assert_not_called()


@pytest.mark.parametrize("status", [403, 409, 422, 429, 500, 503])
def test_service_errors_have_stable_redacted_responses(
    recording_service: Mock, status: int,
) -> None:
    """Map domain failures without reflecting payloads or ownership credentials.

    Args:
        recording_service (Mock): In-memory service implementation.
        status (int): Domain error's HTTP status.
    """
    recording_service.submit.side_effect = ServiceError(status, "controlled_error")
    with TestClient(create_app(service=recording_service)) as client:
        response = client.post("/v1/events", json={}, headers={"x-event-sequence": "0"})
        assert response.status_code == status
        assert response.json() == {"error": "controlled_error"}


@pytest.mark.parametrize("changes", [
    {"bundle_dir": ""}, {"max_line_bytes": True}, {"max_buffered_events": 0},
    {"idle_timeout_seconds": float("nan")}, {"idle_timeout_seconds": float("inf")},
    {"idle_timeout_seconds": True}, {"extra": 3},
])
def test_service_config_rejects_invalid_limits(tmp_path: Path, changes: dict[str, Any]) -> None:
    """Reject ambiguous settings and invalid resource limits.

    Args:
        tmp_path (Path): Isolated test directory.
        changes (dict[str, Any]): Invalid settings.
    """
    with pytest.raises(ValueError):
        ServiceConfig.load(config_file(tmp_path, **changes))


def test_service_config_resolves_bundle_from_yaml(tmp_path: Path) -> None:
    """Anchor the bundle path to the operator's settings file.

    Args:
        tmp_path (Path): Isolated test directory.
    """
    settings = ServiceConfig.load(config_file(tmp_path, bundle_dir="bundle"))
    assert settings.bundle_dir == tmp_path / "bundle"
    assert settings.max_line_bytes == 4096



def test_http_synthetic_session_matches_direct_runner(bundle: EvaluationBundle) -> None:
    """Preserve direct-runner predictions through the real HTTP/session composition on CPU.

    Args:
        bundle (EvaluationBundle): Random-weight synthetic bundle with test-only CPU resolver.
    """
    events_path = Path(__file__).resolve().parents[1] / "examples/raw-session/session.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    direct = RawSessionRunner(bundle)
    expected = [record for event in events for record in direct.handle(event)]
    service = SessionService(ServiceConfig(bundle_dir=Path("synthetic")),
                             bundle_factory=Mock(return_value=bundle))
    records: list[dict[str, Any]] = []
    token = None
    with TestClient(create_app(service=service)) as client:
        assert client.get("/readyz").status_code == 200
        for sequence, event in enumerate(events):
            headers = {"x-event-sequence": str(sequence)}
            if token is not None:
                headers["x-session-token"] = token
            response = client.post("/v1/events", json=event, headers=headers)
            assert response.status_code == 200, response.text
            reply = response.json()
            token = reply["session_token"]
            assert reply["next_sequence"] == sequence + 1
            records.extend(reply["records"])
    assert records == expected
    assert records[-1]["predicted_epochs"] == 2
    assert records[-1]["skipped_epochs"] == 1
    assert records[-1]["integrity"]["weights_unchanged"]


def test_http_cancellation_preserves_explicit_cleanup(recording_service: Mock) -> None:
    """Forward owner and sequence headers without inventing a finished prediction record.

    Args:
        recording_service (Mock): In-memory service implementation.
    """
    recording_service.cancel.return_value = {"status": "cancelled"}
    with TestClient(create_app(service=recording_service)) as client:
        response = client.delete("/v1/session", headers={"x-event-sequence": "5",
                                                        "x-session-token": "owner"})
        assert response.json() == {"status": "cancelled"}
        recording_service.cancel.assert_called_once_with("owner", 5)



def test_routers_preserve_paths_and_resolve_each_app_service() -> None:
    """Keep the public API stable while preventing state sharing between app instances."""
    first = Mock(spec=SessionService)
    second = Mock(spec=SessionService)
    first.readiness.return_value = {"ready": False, "reason": "no_bundle"}
    second.readiness.return_value = {"ready": True, "reason": None}
    app_a = create_app(service=first)
    app_b = create_app(service=second)
    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        assert client_a.get("/readyz").status_code == 503
        assert client_b.get("/readyz").status_code == 200
        schema = client_a.get("/openapi.json").json()
        assert set(schema["paths"]) == {"/healthz", "/readyz", "/v1/events", "/v1/session"}
        assert schema["paths"]["/healthz"]["get"]["tags"] == ["health"]
        assert schema["paths"]["/v1/events"]["post"]["tags"] == ["sessions"]
    first.close.assert_called_once()
    second.close.assert_called_once()



def test_manual_model_configs_select_one_fresh_bundle_profile() -> None:
    """Keep the operator's export and service paths aligned without loading source artifacts."""
    root = Path(__file__).resolve().parents[1]
    export = BundleExportConfig.load(root / "configs/service-bundle.yaml")
    service = ServiceConfig.load(root / "configs/service-model.yaml")
    default = ServiceConfig.load(root / "configs/service.yaml")
    assert service.bundle_dir.resolve() == export.output_dir / export.bundle_id
    assert export.fold == "fold-01"
    assert export.bundle_id == "motion-seed42-fold01-http-v1"
    assert default.bundle_dir is None
