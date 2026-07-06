from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SafeVision AI"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./safevision.db"
    secret_key: str = Field(alias="JWT_SECRET_KEY")
    access_token_expire_minutes: int = 60
    cors_origins: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        if not value or len(value.strip()) < 32:
            raise ValueError("JWT_SECRET_KEY must be set and at least 32 characters long")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
