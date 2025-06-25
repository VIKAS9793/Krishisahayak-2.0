"""API-specific configuration management using Pydantic-Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class ApiSettings(BaseSettings):
    """
    Defines API configuration, loaded from a .env file or environment variables.
    """
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='ignore', case_sensitive=False
    )
    
    # API settings
    ENVIRONMENT: str = "development"
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    
    # Model settings for the API
    DEFAULT_MODEL_PATH: str = "models/checkpoints/best.ckpt"

# A single, importable instance for the application to use
settings = ApiSettings()
