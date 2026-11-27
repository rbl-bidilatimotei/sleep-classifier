import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from sleep_classifier.api.routers import health, sessions
from sleep_classifier.api.session_controller import SessionHTTPController
from sleep_classifier.service_config import ServiceConfig
from sleep_classifier.service_sessions import SessionService


class ServiceLifecycle:
    """Restore and release the process-owned service at application boundaries."""

    def __init__(self, service: SessionService) -> None:
        """Retain the service used by this application instance.

        Args:
            service (SessionService): Fixed bundle and session owner.
        """
        self.service = service


    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """Load once and release state after in-flight inference finishes.

        Args:
            app (FastAPI): Application that owns this lifecycle.
        """
        await run_in_threadpool(self.service.load)
        try:
            yield
        finally:
            await run_in_threadpool(self.service.close)


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
    session_service = service if service is not None else SessionService(config)
    lifecycle = ServiceLifecycle(session_service)
    app = FastAPI(title="Sleep classifier local evaluation service", version="1.0.0",
                  lifespan=lifecycle.lifespan)
    app.state.session_service = session_service
    app.state.session_controller = SessionHTTPController(config, session_service)
    app.include_router(health.router)
    app.include_router(sessions.router)
    return app
