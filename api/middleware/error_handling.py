"""
Error handling middleware for the API.
"""

import logging
import traceback
from typing import Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

from api.models import ErrorResponse
from config.logging_config import get_logger

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle and log errors consistently."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            return await self.handle_exception(request, exc)

    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle different types of exceptions."""

        # Log the error
        logger.error(f"Error processing request {request.url}: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Determine error type and response
        if isinstance(exc, HTTPException):
            error_response = ErrorResponse(
                error=exc.detail,
                detail=f"HTTP {exc.status_code}"
            )
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response.dict()
            )

        elif isinstance(exc, ValueError):
            error_response = ErrorResponse(
                error="Invalid input data",
                detail=str(exc)
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )

        elif isinstance(exc, FileNotFoundError):
            error_response = ErrorResponse(
                error="Required resource not found",
                detail="Model or data files not found. Please ensure the system is properly initialized."
            )
            return JSONResponse(
                status_code=503,
                content=error_response.dict()
            )

        else:
            # Generic server error
            error_response = ErrorResponse(
                error="Internal server error",
                detail="An unexpected error occurred. Please try again later."
            )
            return JSONResponse(
                status_code=500,
                content=error_response.dict()
            )


def create_error_response(status_code: int, message: str, detail: str = None) -> JSONResponse:
    """Create standardized error response."""

    error_response = ErrorResponse(
        error=message,
        detail=detail
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


async def validation_exception_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation exceptions."""

    logger.warning(f"Validation error for {request.url}: {str(exc)}")

    return create_error_response(
        status_code=422,
        message="Validation error",
        detail=str(exc)
    )


async def not_found_exception_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Handle file not found exceptions."""

    logger.error(f"Resource not found for {request.url}: {str(exc)}")

    return create_error_response(
        status_code=503,
        message="Service unavailable",
        detail="Required model or data files are not available"
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions."""

    logger.error(f"Unexpected error for {request.url}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    return create_error_response(
        status_code=500,
        message="Internal server error",
        detail="An unexpected error occurred"
    )