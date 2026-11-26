import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from sleep_classifier.bundle_validation import strict_json_loads
from sleep_classifier.service_config import ServiceConfig
from sleep_classifier.service_sessions import ServiceError, SessionService

LOGGER = logging.getLogger(__name__)


class LocalHTTPService:
    """Expose one bounded session with explicit ownership and ordered event requests."""

    def __init__(self, config: ServiceConfig, service: SessionService) -> None:
        """Bind immutable settings and the transport-independent session owner.

        Args:
            config (ServiceConfig): Operator-selected input and resource limits.
            service (SessionService): Single-session manager with strict bundle loading.
        """
        self.config = config
        self.service = service


    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Load the configured bundle once and release state on shutdown.

        Args:
            app (FastAPI): Application whose process owns the session.
        """
        await run_in_threadpool(self.service.load)
        try:
            yield
        finally:
            await run_in_threadpool(self.service.close)


    async def health(self) -> dict[str, str]:
        """Report process liveness without claiming model readiness."""
        return {"status": "alive"}


    async def ready(self) -> JSONResponse:
        """Report bundle readiness and perform request-time idle cleanup."""
        try:
            status = await run_in_threadpool(self.service.readiness)
        except ServiceError:
            return JSONResponse({"ready": False, "reason": "busy"}, status_code=503)
        return JSONResponse(status, status_code=200 if status["ready"] else 503)


    async def events(self, request: Request) -> JSONResponse:
        """Submit one bounded JSON event with its ownership and sequence headers.

        Args:
            request (Request): HTTP request containing one versioned raw-session event.
        """
        try:
            sequence = self._sequence(request)
            token = request.headers.get("x-session-token")
            if token is not None and len(token) > 128:
                raise ServiceError(403, "invalid_session_token")
            event = await self._read_event(request)
            response = await run_in_threadpool(self.service.submit, event, token, sequence)
            return JSONResponse(response)
        except ServiceError as error:
            return JSONResponse({"error": error.code}, status_code=error.status_code)


    async def cancel(self, request: Request) -> JSONResponse:
        """Release the current owner's session without claiming successful completion.

        Args:
            request (Request): Request carrying the owner token and next sequence.
        """
        try:
            sequence = self._sequence(request)
            response = await run_in_threadpool(
                self.service.cancel, request.headers.get("x-session-token"), sequence,
            )
            return JSONResponse(response)
        except ServiceError as error:
            return JSONResponse({"error": error.code}, status_code=error.status_code)


    @staticmethod
    def _sequence(request: Request) -> int:
        """Require a bounded canonical decimal sequence header.

        Args:
            request (Request): Request carrying the event sequence.
        """
        value = request.headers.get("x-event-sequence", "")
        if (not value or len(value) > 16 or not value.isascii() or not value.isdecimal()
                or (len(value) > 1 and value.startswith("0"))):
            raise ServiceError(400, "invalid_event_sequence")
        return int(value)


    async def _read_event(self, request: Request) -> dict[str, Any]:
        """Check content type and byte limits before decoding strict JSON.

        Args:
            request (Request): Streaming request with one event body.
        """
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ServiceError(415, "expected_application_json")
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            raise ServiceError(415, "unsupported_content_encoding")
        length = request.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdecimal() or len(length) > 16:
                raise ServiceError(400, "invalid_content_length")
            if int(length) > self.config.max_line_bytes:
                raise ServiceError(413, "event_body_too_large")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > self.config.max_line_bytes:
                raise ServiceError(413, "event_body_too_large")
            body.extend(chunk)
        try:
            event = strict_json_loads(body.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ServiceError(422, "invalid_json_event") from error
        if type(event) is not dict:
            raise ServiceError(422, "invalid_json_event")
        return event


def create_app(
    config_path: Path | None = None, service: SessionService | None = None,
) -> FastAPI:
    """Create a local single-worker service without restoring a model at import time.

    Args:
        config_path (Path | None): Settings file; otherwise use the environment or defaults.
        service (SessionService | None): Session owner supplied by tests or local composition.
    """
    if config_path is None and os.environ.get("SLEEP_SERVICE_CONFIG"):
        config_path = Path(os.environ["SLEEP_SERVICE_CONFIG"])
    config = ServiceConfig.load(config_path) if config_path is not None else ServiceConfig()
    controller = LocalHTTPService(config, service or SessionService(config))
    app = FastAPI(title="Sleep classifier local evaluation service", version="1.0.0",
                  lifespan=controller.lifespan)
    app.add_api_route("/healthz", controller.health, methods=["GET"])
    app.add_api_route("/readyz", controller.ready, methods=["GET"])
    app.add_api_route("/v1/events", controller.events, methods=["POST"])
    app.add_api_route("/v1/session", controller.cancel, methods=["DELETE"])
    return app
