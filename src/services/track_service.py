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

    async def build_track_rows_for_album(
        self,
        album_ref: str,
        mode: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        1) Obtiene los track_ids del álbum.
        2) Llama a get_tracks_by_ids para traer los tracks completos.
        3) Construye las filas para la tabla `tracks`,
           incluyendo la flag has_lyrics (y guardando letras en Mongo + embeddings).
        """
        track_ids = await self.get_track_ids_for_album(album_ref)
        if not track_ids:
            return []

        tracks = await self.spotify.get_tracks_by_ids(track_ids)
        rows: List[Dict[str, Any]] = []

        for track in tracks:
            if not track:
                continue

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

            rows.append(row)

        return rows

    async def ingest_tracks_for_album(
        self,
        album_ref: str,
    ) -> tuple[int, int]:
        """
        Construye las filas de tracks para un álbum,
        las enriquece con has_lyrics,
        guarda letras en Mongo (y embeddings en lyrics_chunks)
        y hace MERGE por track_id en BigQuery.
        """
        rows = await self.build_track_rows_for_album(album_ref)
        if not rows:
            print(f"[tracks] No se encontraron tracks para álbum {album_ref}")
            return 0, 0

        table_fq = self.bq.table(self.tracks_table)

        sql = f"""
        MERGE `{table_fq}` AS T
        USING (
          SELECT
            @album_id                AS album_id,

            @track_id                AS track_id,
            @track_name              AS track_name,
            @artists                 AS artists,
            @spotify_url             AS spotify_url,
            @preview_url             AS preview_url,
            @uri                     AS uri,
            @duration_ms             AS duration_ms,
            @track_number            AS track_number,
            @disc_number             AS disc_number,
            @explicit                AS explicit,
            @popularity              AS popularity,
            @isrc                    AS isrc,
            @available_markets_count AS available_markets_count,

            @has_lyrics              AS has_lyrics
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

        for row in rows:
            # row ya incluye has_lyrics porque lo llenaste en build_track_rows_for_album
            self.bq.execute(sql, params=row)

        total = len(rows)
        with_lyrics = sum(1 for r in rows if r.get("has_lyrics"))
        print(
            f"[tracks] MERGE OK para {total} tracks del álbum {album_ref} "
            f"(con letras: {with_lyrics})"
        )
        return total, with_lyrics
