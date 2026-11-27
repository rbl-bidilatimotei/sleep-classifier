from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from sleep_classifier.api.dependencies import SessionServiceDep
from sleep_classifier.service_sessions import ServiceError

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def health() -> dict[str, str]:
    """Report process liveness without claiming that a model is ready."""
    return {"status": "alive"}


@router.get("/readyz")
async def ready(service: SessionServiceDep) -> JSONResponse:
    """Report bundle readiness and perform request-time idle cleanup.

    Args:
        service (SessionServiceDep): Application-owned session manager.
    """
    try:
        status = await run_in_threadpool(service.readiness)
    except ServiceError:
        return JSONResponse({"ready": False, "reason": "busy"}, status_code=503)
    return JSONResponse(status, status_code=200 if status["ready"] else 503)
