# src/pipelines/ingest_album_pipeline.py

from __future__ import annotations

from typing import Any, Dict

from src.services.album_service import AlbumService
from src.services.track_service import TrackService


async def ingest_album_pipeline(
    album_ref: str,
) -> Dict[str, Any]:
    """
    Orquesta el flujo completo para un álbum:

      1. Normaliza album_ref → album_id.
      2. Álbum → BigQuery (MERGE en tabla `albums`).
      3. Tracks → BigQuery (MERGE en tabla `tracks`).
      4. Letras → Mongo (y, si está configurado, embeddings → lyrics_chunks).

    Devuelve un resumen para el API.
    """
    album_service = AlbumService()
    track_service = TrackService(with_lyrics=True)

    # 1) Ingesta de álbum
    album_info = await album_service.ingest_album(
        album_ref=album_ref
    )
    # Se asume que album_info es dict con al menos 'album_id' y 'album_name'
    album_id = album_info["album_id"]
    album_name = album_info.get("album_name")

    # 2) Ingesta de tracks (incluye letras y embeddings)
    track_rows_count, has_lyrics_count = await track_service.ingest_tracks_for_album(
        album_ref=album_ref,
    )

    return {
        "album_id": album_id,
        "album_name": album_name,
        "tracks_ingested": track_rows_count,
        "has_lyrics_count": has_lyrics_count,
        "message": f"Álbum {album_id} procesado con {track_rows_count} tracks (con letras: {has_lyrics_count}).",
    }
