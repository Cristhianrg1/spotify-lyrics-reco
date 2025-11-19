from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List

from src.clients.mongo_client import get_mongo
from src.clients.embedding_client import EmbeddingClient


class LyricsEmbeddingService:
    """
    Servicio que:
    - recibe un documento de `lyrics` (con lyrics_text),
    - genera 1 "chunk" por canción (letra completa),
    - calcula embedding,
    - guarda en la colección `lyrics_chunks`.
    """

    def __init__(self) -> None:
        self.mongo = get_mongo()
        self.embedding_client = EmbeddingClient()

    async def _split_into_chunks_full(self, text: str) -> List[str]:
        """
        Versión simplificada: 1 chunk = texto completo.
        Si quieres, puedes truncar si el texto es extremadamente largo.
        """
        text = text.strip()
        if not text:
            return []
        # Ejemplo: truncar a 4000 caracteres para no pasarse de tokens
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars]
        return [text]

    async def process_lyrics_doc(self, lyrics_doc: Dict[str, Any]) -> int:
        """
        Dado un doc de `lyrics`, crea (si no existen) embeddings en `lyrics_chunks`.
        Devuelve el número de chunks insertados.
        """
        track_id = lyrics_doc.get("track_id")
        album_id = lyrics_doc.get("album_id")
        track_name = lyrics_doc.get("track_name")
        artists = lyrics_doc.get("artists")
        language = lyrics_doc.get("language")
        text = lyrics_doc.get("lyrics_text")

        if not track_id or not text:
            return 0

        # ¿Ya tenemos embeddings para este track?
        existing = await self.mongo.lyrics_chunks.find_one({"track_id": track_id})
        if existing:
            # Ya hay algo, no hacemos nada (idempotencia simple)
            return 0

        chunks = await self._split_into_chunks_full(text)
        if not chunks:
            return 0

        docs = []
        for idx, chunk_text in enumerate(chunks):
            embedding = await self.embedding_client.embed(chunk_text)

            docs.append(
                {
                    "track_id": track_id,
                    "album_id": album_id,
                    "track_name": track_name,
                    "artists": artists,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "language": language,
                    "embedding": embedding,
                    "created_at": dt.datetime.utcnow(),
                }
            )

        if docs:
            await self.mongo.lyrics_chunks.insert_many(docs)

        return len(docs)
