from __future__ import annotations

from typing import Any, Dict, List

from src.clients.mongo_client import get_mongo
from src.clients.embedding_client import EmbeddingClient


class LyricsSearchService:
    """
    Servicio de búsqueda semántica sobre la colección lyrics_chunks.

    Requiere que Mongo Atlas tenga un índice vectorial en lyrics_chunks,
    con el campo "embedding" y el nombre de índice definido abajo.
    """

    VECTOR_INDEX_NAME = "lyrics_chunks_vector_index"  # cámbialo si usaste otro nombre

    def __init__(self) -> None:
        self.mongo = get_mongo()
        self.embedding_client = EmbeddingClient()

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        1. Genera embedding del query.
        2. Lanza $vectorSearch sobre lyrics_chunks.
        3. Devuelve lista de hits (track + chunk + score).
        """
        # 1) Embedding del query
        query_vec = await self.embedding_client.embed(query)

        # 2) Pipeline de vector search en Mongo Atlas
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.VECTOR_INDEX_NAME,
                    "path": "embedding",         # campo donde guardaste el vector
                    "queryVector": query_vec,
                    "numCandidates": max(50, top_k * 5),
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "track_id": 1,
                    "track_name": 1,
                    "artists": 1,
                    "album_id": 1,
                    "chunk_text": 1,
                    "mode": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        cursor = self.mongo.lyrics_chunks.aggregate(pipeline)
        docs = await cursor.to_list(length=top_k)

        # Normalizamos un poco por si hay campos faltantes
        results: List[Dict[str, Any]] = []
        for d in docs:
            results.append(
                {
                    "track_id": d.get("track_id"),
                    "track_name": d.get("track_name"),
                    "artists": d.get("artists"),
                    "album_id": d.get("album_id"),
                    "chunk_text": d.get("chunk_text"),
                    "mode": d.get("mode"),
                    "score": float(d.get("score", 0.0)),
                }
            )

        return results
