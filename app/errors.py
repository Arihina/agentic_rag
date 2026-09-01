from __future__ import annotations

"""Единый формат ошибок — совместимый с OpenAI (`{"error": {...}}`).

Внутри сервиса кидаем обычный HTTPException; сюда ошибки попадают уже на
выходе, и здесь же превращаются в OpenAI-body, чтобы клиенту через мастер
приходил один и тот же формат независимо от того, какой из агентов упал.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_STATUS_TO_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_denied_error",
    404: "not_found_error",
    409: "invalid_request_error",
    413: "invalid_request_error",
    415: "invalid_request_error",
    422: "invalid_request_error",
    502: "bad_gateway_error",
    503: "service_unavailable_error",
    504: "timeout_error",
}


def error_body(status_code: int, message: str, param: str | None = None) -> dict:
    return {
        "error": {
            "message": message,
            "type": _STATUS_TO_TYPE.get(status_code, "server_error"),
            "param": param,
            "code": None,
        }
    }


def install(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.status_code, str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(p) for p in first.get("loc", ())
               if p not in ("body", "query", "path", "header")]
        return JSONResponse(
            status_code=400,
            content=error_body(
                400, first.get("msg", "Некорректный запрос"),
                ".".join(loc) or None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Необработанная ошибка на %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(500, f"Внутренняя ошибка: {type(exc).__name__}"))
