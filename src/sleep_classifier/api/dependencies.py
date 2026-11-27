from typing import Annotated

from fastapi import Depends, Request

from sleep_classifier.api.session_controller import SessionHTTPController
from sleep_classifier.service_sessions import SessionService


def get_session_service(request: Request) -> SessionService:
    """Resolve the session owner from this application instead of global mutable state.

    Args:
        request (Request): Request bound to the active application.
    """
    return request.app.state.session_service


def get_session_controller(request: Request) -> SessionHTTPController:
    """Resolve the configured HTTP decoder for this application instance.

    Args:
        request (Request): Request bound to the active application.
    """
    return request.app.state.session_controller


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
SessionControllerDep = Annotated[SessionHTTPController, Depends(get_session_controller)]
