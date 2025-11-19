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

    mongo_uri: str | None = Field(default=None, env="MONGO_URI")
    mongo_db_name: str | None = Field(default=None, env="MONGO_DB_NAME")

    # ruta al JSON de la SA
    google_application_credentials: str | None = Field(
        default=None,
        env="GOOGLE_APPLICATION_CREDENTIALS",
    )

    # Embeddings
    # "openai" o "huggingface_api"
    embedding_provider: str = Field(
        "openai",
        env="EMBEDDING_PROVIDER",
    )

    # OpenAI
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        "text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL",
    )

    # Hugging Face Inference API
    hf_api_key: str | None = Field(None, env="HF_API_KEY")
    hf_embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        env="HF_EMBEDDING_MODEL",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
