"""
Custom exception classes for the Vanna AI service
"""

from typing import Optional, Dict, Any


class VannaServiceError(Exception):
    """Base exception for Vanna service errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(VannaServiceError):
    """Database connection or query errors"""
    pass


class GroqAPIError(VannaServiceError):
    """Groq API related errors"""
    pass


class RateLimitError(GroqAPIError):
    """Rate limit exceeded errors"""
    pass


class SQLGenerationError(VannaServiceError):
    """SQL generation related errors"""
    pass


class SQLValidationError(VannaServiceError):
    """SQL validation and safety check errors"""
    pass


class QueryExecutionError(VannaServiceError):
    """Query execution errors"""
    pass


class SchemaError(VannaServiceError):
    """Database schema related errors"""
    pass


class AuthenticationError(VannaServiceError):
    """Authentication and authorization errors"""
    pass


class ValidationError(VannaServiceError):
    """Input validation errors"""
    pass


class TimeoutError(VannaServiceError):
    """Operation timeout errors"""
    pass


class CacheError(VannaServiceError):
    """Cache related errors"""
    pass