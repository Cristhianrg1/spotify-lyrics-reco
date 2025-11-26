from __future__ import annotations

from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config.settings import get_settings


class MongoClientWrapper:
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.mongo_uri:
            raise ValueError("MONGO_URI no está configurado en .env")
        if not settings.mongo_db_name:
            raise ValueError("MONGO_DB_NAME no está configurado en .env")

        self._client = AsyncIOMotorClient(settings.mongo_uri)
        self._db: AsyncIOMotorDatabase = self._client[settings.mongo_db_name]

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    @property
    def lyrics(self):
        return self._db["lyrics"]

    @property
    def lyrics_chunks(self):
        return self._db["lyrics_chunks"]

    @property
    def album_analysis(self):
        return self._db["album_analysis"]


@lru_cache
def get_mongo() -> MongoClientWrapper:
    return MongoClientWrapper()
