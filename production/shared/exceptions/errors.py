"""
production/shared/exceptions/errors.py

Domain-level exception hierarchy. Services and repositories raise these
instead of `fastapi.HTTPException` directly, which keeps HTTP-status
knowledge out of the service layer (a service shouldn't need to know it's
being called over HTTP) and gives consistent, structured error responses
via the handlers registered in production/backend/app/main.py.

Usage:
    from production.shared.exceptions.errors import NotFoundError
    raise NotFoundError("Movie", movie_id)
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for all domain errors raised anywhere in production/."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": self.error_code, "message": self.message, "details": self.details}


class NotFoundError(AppError):
    """Raised when a requested resource (user, movie, etc.) doesn't exist."""

    status_code = 404
    error_code = "not_found"

    def __init__(self, resource: str, identifier):
        super().__init__(f"{resource} not found: {identifier}", details={"resource": resource, "id": str(identifier)})


class ValidationError(AppError):
    """Raised for domain-level validation failures that aren't caught by
    Pydantic schema validation (e.g. cross-field business rules)."""

    status_code = 422
    error_code = "validation_error"


class AuthenticationError(AppError):
    """Raised for invalid credentials or expired/invalid tokens."""

    status_code = 401
    error_code = "authentication_error"


class AuthorizationError(AppError):
    """Raised when an authenticated user lacks permission for the action."""

    status_code = 403
    error_code = "authorization_error"


class ModelServerUnavailableError(AppError):
    """Raised when the hybrid recommendation engine failed to load or the
    inference call otherwise cannot be served."""

    status_code = 503
    error_code = "model_unavailable"

    def __init__(self, message: str = "Recommendation engine is not currently available"):
        super().__init__(message)


class ConflictError(AppError):
    """Raised for uniqueness violations (e.g. registering a duplicate username)."""

    status_code = 409
    error_code = "conflict"
