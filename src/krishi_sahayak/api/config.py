"""API-specific configuration management using Pydantic-Settings."""
from typing import List, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """
    Defines API configuration, loaded from a .env file or environment variables.
    Variable names here must be uppercase to be loaded from the environment.
    """
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore', 
        case_sensitive=False
    )
    
    # --- Application Metadata ---
    PROJECT_NAME: str = "KrishiSahayak API"
    VERSION: str = "2.1.0"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # --- API Settings ---
    ENVIRONMENT: Literal["development", "production"] = "development"
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"] # Use a specific list of domains in production
    
    # --- Model Loading Settings ---
    DEFAULT_MODEL_PATH: str = "output/checkpoints/best_model.ckpt"

# A single, importable instance for the application to use
settings = ApiSettings()