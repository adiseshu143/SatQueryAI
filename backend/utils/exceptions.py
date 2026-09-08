class SatQueryException(Exception):
    """Base exception for SatQuery AI errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)

class InvalidImageError(SatQueryException):
    def __init__(self, message: str):
        super().__init__(message, code="INVALID_IMAGE")

class RegistrationFailedError(SatQueryException):
    def __init__(self, message: str = "Image registration failed to find sufficient matching features."):
        super().__init__(message, code="REGISTRATION_FAILED")

class ModelInferenceError(SatQueryException):
    def __init__(self, message: str):
        super().__init__(message, code="MODEL_ERROR")

class QueryValidationError(SatQueryException):
    def __init__(self, message: str):
        super().__init__(message, code="QUERY_VALIDATION_ERROR")
