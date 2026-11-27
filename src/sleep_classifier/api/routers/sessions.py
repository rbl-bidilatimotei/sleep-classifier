from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from sleep_classifier.api.dependencies import SessionControllerDep

router = APIRouter(prefix="/v1", tags=["sessions"])


@router.post("/events")
async def submit_event(request: Request, controller: SessionControllerDep) -> JSONResponse:
    """Submit one owned, sequenced event through the bounded decoder.

    Args:
        request (Request): Request containing a versioned raw-sensor event.
        controller (SessionControllerDep): This application's HTTP input controller.
    """
    return await controller.events(request)


@router.delete("/session")
async def cancel_session(request: Request, controller: SessionControllerDep) -> JSONResponse:
    """Release the active owner's session without claiming successful completion.

    Args:
        request (Request): Request carrying the owner and next sequence headers.
        controller (SessionControllerDep): This application's HTTP input controller.
    """
    return await controller.cancel(request)
