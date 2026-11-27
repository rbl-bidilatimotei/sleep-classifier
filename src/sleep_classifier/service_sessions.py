import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.raw_session_runner import RawSessionRunner
from sleep_classifier.service_config import ServiceConfig
from sleep_classifier.session_protocol import validate_event
from sleep_classifier.streaming_inference import StreamingEpochPredictor

LOGGER = logging.getLogger(__name__)


class ServiceError(Exception):
    """Describe a safe transport error without echoing sensor inputs or tokens."""

    def __init__(self, status_code: int, code: str) -> None:
        """Retain the public status and error code.

        Args:
            status_code (int): HTTP status for the failed operation.
            code (str): Stable public error identifier.
        """
        super().__init__(code)
        self.status_code = status_code
        self.code = code


@dataclass
class _Session:
    """Hold the single active runner and its ownership and progress fields."""

    runner: RawSessionRunner
    token: str
    next_sequence: int
    accepted_events: int
    last_activity: float


class SessionService:
    """Own one frozen bundle and at most one active raw-event session."""

    def __init__(
        self, settings: ServiceConfig,
        bundle_factory: Callable[[Path], EvaluationBundle] = EvaluationBundle,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set fixed startup settings and dependency seams for synthetic tests.

        Args:
            settings (ServiceConfig): Validated fixed bundle path and resource limits.
            bundle_factory (Callable[[Path], EvaluationBundle]): Verified bundle loader.
            clock (Callable[[], float]): Monotonic clock for idle expiry.
        """
        self._settings = settings
        self._bundle_factory = bundle_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._bundle: EvaluationBundle | None = None
        self._session: _Session | None = None
        self._unready_reason: str | None = "no_bundle"


    def load(self) -> None:
        """Load the configured bundle once and require its validated device without fallback."""
        with self._lock:
            self._session = None
            self._bundle = None
            if self._settings.bundle_dir is None:
                self._unready_reason = "no_bundle"
                LOGGER.warning(
                    "No evaluation bundle configured; inference readiness is disabled. "
                    "Set SLEEP_SERVICE_CONFIG to a config with a compatible bundle_dir. "
                    "Use /healthz for liveness and /readyz for the reason.",
                )
                return
            try:
                bundle = self._bundle_factory(self._settings.bundle_dir)
                device = bundle.manifest["runtime"]["validated_device"]
                StreamingEpochPredictor._resolve_device(device)
            except Exception as error:
                self._unready_reason = "bundle_unavailable"
                LOGGER.error("Evaluation bundle unavailable (%s)", type(error).__name__)
                return
            self._bundle = bundle
            self._unready_reason = None
            LOGGER.info("Evaluation bundle ready")


    def close(self) -> None:
        """Release the active session and bundle after in-flight inference returns."""
        with self._lock:
            self._session = None
            self._bundle = None
            self._unready_reason = "no_bundle"


    def readiness(self) -> dict[str, Any]:
        """Report bundle availability and expire an idle session when the lock is free."""
        if self._lock.acquire(blocking=False):
            try:
                self._expire()
                return {"ready": self._bundle is not None, "reason": self._unready_reason}
            finally:
                self._lock.release()
        return {"ready": self._bundle is not None, "reason": self._unready_reason}


    def submit(
        self, event: dict[str, Any], token: str | None, sequence: int,
    ) -> dict[str, Any]:
        """Accept one ordered event for the current owner without queuing concurrent inference.

        Args:
            event (dict[str, Any]): One decoded version-one raw event.
            token (str | None): Owner token, absent only for a new session.
            sequence (int): Zero-based event sequence for the active session.
        """
        if not self._lock.acquire(blocking=False):
            raise ServiceError(409, "session_busy")
        try:
            self._expire()
            if self._bundle is None:
                raise ServiceError(503, "not_ready")
            self._validate_submission(event, token, sequence)
            if event["type"] == "start":
                if token is not None or sequence != 0:
                    raise ServiceError(409, "invalid_start")
                return self._start(event)
            return self._advance(event, token, sequence)
        finally:
            self._lock.release()


    def cancel(self, token: str | None, sequence: int) -> dict[str, Any]:
        """Release the current owner without claiming successful model completion.

        Args:
            token (str | None): Claimed session owner token.
            sequence (int): Requested next event sequence.
        """
        if not self._lock.acquire(blocking=False):
            raise ServiceError(409, "session_busy")
        try:
            self._expire()
            if self._bundle is None:
                raise ServiceError(503, "not_ready")
            self._validate_headers(token, sequence)
            self._owner(token, sequence)
            self._session = None
            return {"status": "cancelled"}
        finally:
            self._lock.release()


    @staticmethod
    def _validate_headers(token: str | None, sequence: int) -> None:
        """Check strict ownership and progress field types before session lookup.

        Args:
            token (str | None): Claimed session owner token.
            sequence (int): Requested event sequence.
        """
        if type(sequence) is not int or sequence < 0:
            raise ServiceError(422, "invalid_sequence")
        if token is not None and (type(token) is not str or not token or len(token) > 128
                                  or not token.isascii()):
            raise ServiceError(422, "invalid_token")


    @staticmethod
    def _validate_submission(event: dict[str, Any], token: str | None, sequence: int) -> None:
        """Validate public types before any runner state can change.

        Args:
            event (dict[str, Any]): Decoded event record.
            token (str | None): Optional owner token.
            sequence (int): Requested event sequence.
        """
        SessionService._validate_headers(token, sequence)
        try:
            validate_event(event)
        except (ValueError, RecursionError):
            raise ServiceError(422, "invalid_event") from None


    def _expire(self) -> None:
        """Release an idle session when another service operation checks its age."""
        if (self._session is not None
                and self._clock() - self._session.last_activity
                >= self._settings.idle_timeout_seconds):
            self._session = None
            LOGGER.info("Idle session expired")


    def _start(self, event: dict[str, Any]) -> dict[str, Any]:
        """Create fresh prediction state for the first owner event.

        Args:
            event (dict[str, Any]): Validated start record.
        """
        if self._session is not None:
            raise ServiceError(409, "session_active")
        # The caller's start headers are checked before this method is entered
        assert self._bundle is not None
        runner = RawSessionRunner(
            self._bundle, max_buffered_events=self._settings.max_buffered_events,
            max_epochs_per_event=self._settings.max_epochs_per_event,
        )
        try:
            records = runner.handle(event)
        except Exception:
            raise ServiceError(500, "session_failed") from None
        token = secrets.token_urlsafe(32)
        session = _Session(runner, token, 1, 1, self._clock())
        self._session = session
        return {"session_token": token, "next_sequence": 1, "records": records}


    def _advance(
        self, event: dict[str, Any], token: str | None, sequence: int,
    ) -> dict[str, Any]:
        """Apply one owner event and release state after finish or a terminal failure.

        Args:
            event (dict[str, Any]): Validated non-start event.
            token (str | None): Claimed session owner token.
            sequence (int): Requested next event sequence.
        """
        session = self._owner(token, sequence)
        if (session.accepted_events >= self._settings.max_session_events
                and event["type"] != "finish"):
            raise ServiceError(429, "session_event_limit")
        try:
            records = session.runner.handle(event)
        except Exception as error:
            if session.runner.status()["phase"] == "failed":
                self._session = None
                raise ServiceError(500, "session_failed") from None
            if isinstance(error, BufferError):
                raise ServiceError(429, "buffer_limit") from None
            if isinstance(error, ValueError):
                raise ServiceError(422, "event_rejected") from None
            self._session = None
            raise ServiceError(500, "session_failed") from None
        session.accepted_events += 1
        session.next_sequence += 1
        session.last_activity = self._clock()
        response = {"session_token": session.token, "next_sequence": session.next_sequence,
                    "records": records}
        if event["type"] == "finish":
            self._session = None
        return response


    def _owner(self, token: str | None, sequence: int) -> _Session:
        """Find the active owner and require its exact next event sequence.

        Args:
            token (str | None): Claimed session owner token.
            sequence (int): Requested next event sequence.
        """
        session = self._session
        if session is None:
            raise ServiceError(409, "no_active_session")
        if token is None or not secrets.compare_digest(token, session.token):
            raise ServiceError(403, "invalid_owner")
        if sequence != session.next_sequence:
            raise ServiceError(409, "sequence_conflict")
        return session
