from __future__ import annotations

from typing import List, Dict, Any
import datetime as dt

from src.clients.spotify_client import SpotifyClient
from src.clients.bigquery_client import BigQueryClient
from src.clients.lyrics_client import LyricsClient
from src.services.lyrics_embedding_service import LyricsEmbeddingService
from src.clients.mongo_client import get_mongo


class TrackService:
    def __init__(
        self,
        spotify_client: SpotifyClient | None = None,
        bq_client: BigQueryClient | None = None,
        tracks_table: str = "tracks",
        with_lyrics: bool = True,
    ) -> None:
        self.spotify = spotify_client or SpotifyClient()
        self.bq = bq_client or BigQueryClient()
        self.tracks_table = tracks_table

        self.with_lyrics = with_lyrics
        self.lyrics_client = LyricsClient() if with_lyrics else None
        self.mongo = get_mongo() if with_lyrics else None
        self.lyrics_embedding_service = LyricsEmbeddingService() if with_lyrics else None

    async def get_track_ids_for_album(
        self,
        album_ref: str,
    ) -> List[str]:
        """
        Devuelve los track_ids de un álbum usando iter_album_tracks.
        """
        track_ids: List[str] = []
        async for t in self.spotify.iter_album_tracks(album_ref):
            if t and t.get("id"):
                track_ids.append(t["id"])
        return track_ids

    async def _fetch_and_store_lyrics(
        self,
        row: Dict[str, Any],
    ) -> bool:
        """
        Dado un row de track (sin has_lyrics), intenta:
        - Ver si ya hay letras en Mongo.
        - Si no, llama a LyricsClient y guarda el doc en Mongo.
        - Si hay letras, genera embeddings en lyrics_chunks (si no existen).
        Devuelve True/False según si hay letra.
        """
        if not (self.with_lyrics and self.lyrics_client and self.mongo):
            return False

        track_id = row["track_id"]
        track_name = row["track_name"]
        artists = row["artists"]
        album_id = row["album_id"]
        mode = row.get("mode", "unknown")

        # ¿ya hay letras para este track?
        existing = await self.mongo.lyrics.find_one({"track_id": track_id})
        if existing:
            has_lyrics = bool(existing.get("has_lyrics", False))

            # Si ya hay letras pero aún no hay embeddings, los creamos
            if (
                has_lyrics
                and self.lyrics_embedding_service is not None
            ):
                existing_chunk = await self.mongo.lyrics_chunks.find_one(
                    {"track_id": track_id}
                )
                if not existing_chunk:
                    await self.lyrics_embedding_service.process_lyrics_doc(existing)

            return has_lyrics

        # Si no existe, buscamos letras en el proveedor
        main_artist = (artists or "").split(",")[0].strip()

        texts = await self.lyrics_client.get_lyrics_texts(
            track_name=track_name,
            artist_name=main_artist,
        )

        has_lyrics = bool(texts.get("lyrics_text"))

        doc = {
            "track_id": track_id,
            "album_id": album_id,
            "track_name": track_name,
            "artists": artists,
            "mode": mode,
            "provider": "lrclib",
            "language": texts.get("language"),
            "has_lyrics": has_lyrics,
            "lyrics_text": texts.get("lyrics_text"),
            "synced_lrc": texts.get("synced_lrc"),
            "created_at": dt.datetime.utcnow(),
        }

        await self.mongo.lyrics.insert_one(doc)

        # Si hay letra y tenemos servicio de embeddings, generamos vector y lo guardamos
        if has_lyrics and self.lyrics_embedding_service is not None:
            await self.lyrics_embedding_service.process_lyrics_doc(doc)

        return has_lyrics

    async def _process_single_track(
        self,
        track: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """
        Procesa un único track: extrae metadatos, busca letras y genera embeddings.
        Devuelve el row listo para BigQuery o None si el track es inválido.
        """
        if not track:
            return None

        t_id = track.get("id")
        t_name = track.get("name")
        t_artists = ", ".join(a["name"] for a in track.get("artists", []))

        t_album = track.get("album") or {}
        t_album_id = t_album.get("id")

        t_spotify_url = (track.get("external_urls") or {}).get("spotify")
        t_preview_url = track.get("preview_url")
        t_uri = track.get("uri")

        t_duration_ms = track.get("duration_ms")
        t_track_number = track.get("track_number")
        t_disc_number = track.get("disc_number")
        t_explicit = track.get("explicit")
        t_popularity = track.get("popularity")

        t_external_ids = track.get("external_ids") or {}
        t_isrc = t_external_ids.get("isrc")

        t_available_markets = track.get("available_markets") or []
        t_markets_count = len(t_available_markets)

        row: Dict[str, Any] = {
            # Álbum
            "album_id": t_album_id,

            # Track
            "track_id": t_id,
            "track_name": t_name,
            "artists": t_artists,
            "spotify_url": t_spotify_url,
            "preview_url": t_preview_url,
            "uri": t_uri,
            "duration_ms": t_duration_ms,
            "track_number": t_track_number,
            "disc_number": t_disc_number,
            "explicit": t_explicit,
            "popularity": t_popularity,
            "isrc": t_isrc,
            "available_markets_count": t_markets_count,
        }

        # --- Letras → Mongo + flag has_lyrics + embeddings ---
        has_lyrics = await self._fetch_and_store_lyrics(row)
        row["has_lyrics"] = has_lyrics

        return row

    async def build_track_rows_for_album(
        self,
        album_ref: str,
        mode: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        1) Obtiene los track_ids del álbum.
        2) Llama a get_tracks_by_ids para traer los tracks completos.
        3) Procesa cada track en PARALELO (asyncio.gather).
        """
        track_ids = await self.get_track_ids_for_album(album_ref)
        if not track_ids:
            return []

        tracks = await self.spotify.get_tracks_by_ids(track_ids)
        
        # Procesamiento concurrente
        import asyncio
        tasks = [self._process_single_track(t) for t in tracks]
        results = await asyncio.gather(*tasks)

        # Filtrar Nones
        rows = [r for r in results if r is not None]
        return rows

    async def ingest_tracks_for_album(
        self,
        album_ref: str,
    ) -> tuple[int, int]:
        """
        Construye las filas de tracks para un álbum (en paralelo),
        y hace un único MERGE batch en BigQuery usando JSON parameter.
        """
        rows = await self.build_track_rows_for_album(album_ref)
        if not rows:
            print(f"[tracks] No se encontraron tracks para álbum {album_ref}")
            return 0, 0

        table_fq = self.bq.table(self.tracks_table)
        
        # Serializamos a JSON string para evitar problemas de tipos complejos en parámetros
        import json
        # Usamos default=str para manejar fechas u otros tipos no serializables si los hubiera
        tracks_json = json.dumps(rows, default=str)

        # Query optimizada usando JSON parsing
        # JSON_QUERY_ARRAY extrae el array de objetos del string JSON
        # Luego UNNEST itera sobre ellos
        # Luego JSON_VALUE extrae cada campo (siempre devuelve STRING, hay que castear)
        sql = f"""
        MERGE `{table_fq}` AS T
        USING (
          SELECT
            JSON_VALUE(item, '$.album_id') AS album_id,
            JSON_VALUE(item, '$.track_id') AS track_id,
            JSON_VALUE(item, '$.track_name') AS track_name,
            JSON_VALUE(item, '$.artists') AS artists,
            JSON_VALUE(item, '$.spotify_url') AS spotify_url,
            JSON_VALUE(item, '$.preview_url') AS preview_url,
            JSON_VALUE(item, '$.uri') AS uri,
            CAST(JSON_VALUE(item, '$.duration_ms') AS INT64) AS duration_ms,
            CAST(JSON_VALUE(item, '$.track_number') AS INT64) AS track_number,
            CAST(JSON_VALUE(item, '$.disc_number') AS INT64) AS disc_number,
            CAST(JSON_VALUE(item, '$.explicit') AS BOOL) AS explicit,
            CAST(JSON_VALUE(item, '$.popularity') AS INT64) AS popularity,
            JSON_VALUE(item, '$.isrc') AS isrc,
            CAST(JSON_VALUE(item, '$.available_markets_count') AS INT64) AS available_markets_count,
            CAST(JSON_VALUE(item, '$.has_lyrics') AS BOOL) AS has_lyrics
          FROM UNNEST(JSON_QUERY_ARRAY(@tracks_json)) AS item
        ) AS S
        ON T.track_id = S.track_id
        WHEN MATCHED THEN
          UPDATE SET
            album_id                = S.album_id,
            track_name              = S.track_name,
            artists                 = S.artists,
            spotify_url             = S.spotify_url,
            preview_url             = S.preview_url,
            uri                     = S.uri,
            duration_ms             = S.duration_ms,
            track_number            = S.track_number,
            disc_number             = S.disc_number,
            explicit                = S.explicit,
            popularity              = S.popularity,
            isrc                    = S.isrc,
            available_markets_count = S.available_markets_count,
            has_lyrics              = S.has_lyrics
        WHEN NOT MATCHED THEN
          INSERT (
            album_id,
            track_id,
            track_name,
            artists,
            spotify_url,
            preview_url,
            uri,
            duration_ms,
            track_number,
            disc_number,
            explicit,
            popularity,
            isrc,
            available_markets_count,
            has_lyrics
          )
          VALUES (
            S.album_id,
            S.track_id,
            S.track_name,
            S.artists,
            S.spotify_url,
            S.preview_url,
            S.uri,
            S.duration_ms,
            S.track_number,
            S.disc_number,
            S.explicit,
            S.popularity,
            S.isrc,
            S.available_markets_count,
            S.has_lyrics
          );
        """

        # Pasamos un único parámetro STRING. BigQueryClient.execute maneja strings sin problemas.
        params = {"tracks_json": tracks_json}
        
        try:
            self.bq.execute(sql, params=params)
        except Exception as e:
            print(f"[ERROR] BigQuery Batch Merge failed: {e}")
            raise e

        total = len(rows)
        with_lyrics = sum(1 for r in rows if r.get("has_lyrics"))
        print(
            f"[tracks] BATCH MERGE OK para {total} tracks del álbum {album_ref} "
            f"(con letras: {with_lyrics})"
        )
        return total, with_lyrics
