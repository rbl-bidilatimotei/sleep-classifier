from typing import Any

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from sleep_classifier.bundle_validation import strict_json_loads
from sleep_classifier.service_config import ServiceConfig
from sleep_classifier.service_sessions import ServiceError, SessionService


class SessionHTTPController:
    """Decode bounded HTTP input and delegate mutations to the session owner."""

    def __init__(self, config: ServiceConfig, service: SessionService) -> None:
        """Bind immutable settings and the transport-independent session owner.

        Args:
            config (ServiceConfig): Operator-selected input and resource limits.
            service (SessionService): Single-session manager with strict bundle loading.
        """
        self.config = config
        self.service = service


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

