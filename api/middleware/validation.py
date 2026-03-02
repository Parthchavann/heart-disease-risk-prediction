"""
Input validation middleware and utilities.
"""

import time
import json
from typing import Dict, Any
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests and responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log incoming request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Incoming {request.method} request to {request.url.path} from {client_ip}")

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(f"Request to {request.url.path} completed with status {response.status_code} "
                   f"in {process_time:.3f}s")

        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, max_requests_per_minute: int = 60):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.request_counts: Dict[str, Dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries (older than 1 minute)
        self._cleanup_old_entries(current_time)

        # Check rate limit
        if client_ip in self.request_counts:
            request_data = self.request_counts[client_ip]
            if request_data["count"] >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Maximum {self.max_requests} requests per minute."
                )
            request_data["count"] += 1
        else:
            self.request_counts[client_ip] = {
                "count": 1,
                "start_time": current_time
            }

        response = await call_next(request)
        return response

    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than 1 minute."""
        expired_ips = []
        for ip, data in self.request_counts.items():
            if current_time - data["start_time"] > 60:  # 1 minute
                expired_ips.append(ip)

        for ip in expired_ips:
            del self.request_counts[ip]


def validate_patient_data_ranges(patient_data: Dict[str, Any]) -> None:
    """Validate that patient data is within expected ranges."""

    from utils.constants import FEATURE_RANGES

    errors = []

    for feature, value in patient_data.items():
        if feature in FEATURE_RANGES:
            min_val = FEATURE_RANGES[feature]["min"]
            max_val = FEATURE_RANGES[feature]["max"]

            if not (min_val <= value <= max_val):
                errors.append(f"{feature} value {value} is outside valid range [{min_val}, {max_val}]")

    if errors:
        raise ValueError(f"Validation errors: {', '.join(errors)}")


def validate_request_size(request_data: Dict[str, Any], max_size_mb: float = 1.0) -> None:
    """Validate that request size is within limits."""

    request_size = len(json.dumps(request_data).encode('utf-8'))
    max_size_bytes = max_size_mb * 1024 * 1024

    if request_size > max_size_bytes:
        raise ValueError(f"Request size {request_size} bytes exceeds maximum {max_size_bytes} bytes")


def sanitize_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data to prevent injection attacks."""

    sanitized = {}

    for key, value in data.items():
        # Convert to appropriate types
        if isinstance(value, str):
            # Remove any potentially dangerous characters
            value = value.strip()
            # Try to convert to number if it looks like one
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if conversion fails
                pass

        # Ensure numeric values are within reasonable bounds
        if isinstance(value, (int, float)):
            value = max(-1000, min(1000, value))

        sanitized[key] = value

    return sanitized