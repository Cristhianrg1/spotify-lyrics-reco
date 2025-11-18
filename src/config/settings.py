from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # obligatorios
    spotify_client_id: str = Field(..., env="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str = Field(..., env="SPOTIFY_CLIENT_SECRET")

    # opcionales
    gcp_project_id: str | None = Field(default=None, env="GCP_PROJECT_ID")
    bigquery_dataset: str | None = Field(default=None, env="BIGQUERY_DATASET")

    # ruta al JSON de la SA
    google_application_credentials: str | None = Field(
        default=None,
        env="GOOGLE_APPLICATION_CREDENTIALS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
