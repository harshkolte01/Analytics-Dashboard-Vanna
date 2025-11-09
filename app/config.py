"""
Configuration module for Vanna AI service
Loads environment variables and provides configuration settings
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", env="VANNA_HOST")
    port: int = Field(default=8000, env="VANNA_PORT")
    debug: bool = Field(default=False, env="VANNA_DEBUG")
    
    # Database configuration
    database_url: str = Field(..., env="DATABASE_URL")
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    # AI API configuration (supports any OpenAI-compatible API)
    ai_api_key: str = Field(..., env="AI_API_KEY")
    ai_base_url: str = Field(default="https://api.groq.com/openai/v1", env="AI_BASE_URL")
    ai_model: str = Field(default="llama-3.1-70b-versatile", env="AI_MODEL")
    ai_max_tokens: int = Field(default=2048, env="AI_MAX_TOKENS")
    ai_temperature: float = Field(default=0.1, env="AI_TEMPERATURE")
    
    # Legacy Groq API configuration (for backward compatibility)
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-70b-versatile", env="GROQ_MODEL")
    groq_max_tokens: int = Field(default=2048, env="GROQ_MAX_TOKENS")
    groq_temperature: float = Field(default=0.1, env="GROQ_TEMPERATURE")
    
    # Vanna configuration
    vanna_model_name: str = Field(default="analytics_model", env="VANNA_MODEL_NAME")
    vanna_api_key: Optional[str] = Field(default=None, env="VANNA_API_KEY")
    
    # Security settings
    query_timeout: int = Field(default=30, env="QUERY_TIMEOUT")
    max_query_complexity: int = Field(default=100, env="MAX_QUERY_COMPLEXITY")
    allowed_operations: str = Field(default="SELECT", env="ALLOWED_OPERATIONS")
    
    @property
    def allowed_operations_list(self) -> List[str]:
        """Convert allowed_operations string to list"""
        if isinstance(self.allowed_operations, str):
            return [op.strip().upper() for op in self.allowed_operations.split(',')]
        return self.allowed_operations
    
    # Cache configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()


def get_database_config() -> dict:
    """Get database configuration dictionary"""
    return {
        "url": settings.database_url,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "echo": settings.debug
    }


def get_groq_config() -> dict:
    """Get AI API configuration dictionary (backward compatible)"""
    return {
        "api_key": settings.ai_api_key,
        "base_url": settings.ai_base_url,
        "model": settings.ai_model,
        "max_tokens": settings.ai_max_tokens,
        "temperature": settings.ai_temperature
    }

def get_ai_config() -> dict:
    """Get AI API configuration dictionary"""
    return {
        "api_key": settings.ai_api_key,
        "base_url": settings.ai_base_url,
        "model": settings.ai_model,
        "max_tokens": settings.ai_max_tokens,
        "temperature": settings.ai_temperature
    }


def get_security_config() -> dict:
    """Get security configuration dictionary"""
    return {
        "query_timeout": settings.query_timeout,
        "max_complexity": settings.max_query_complexity,
        "allowed_operations": settings.allowed_operations_list
    }