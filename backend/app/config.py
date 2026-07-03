from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "SafeVision AI"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = Field(
        default="postgresql+psycopg2://safevision:safevision@localhost:5432/safevision"
    )
    secret_key: str = "change-this-secret-before-production"
    access_token_expire_minutes: int = 60
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:8541"]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

