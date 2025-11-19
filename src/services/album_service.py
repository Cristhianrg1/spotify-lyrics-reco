from __future__ import annotations

from typing import Dict, Any

from src.clients.spotify_client import SpotifyClient
from src.clients.bigquery_client import BigQueryClient


class AlbumService:
    def __init__(
        self,
        spotify_client: SpotifyClient | None = None,
        bq_client: BigQueryClient | None = None,
        albums_table: str = "albums",
    ) -> None:
        self.spotify = spotify_client or SpotifyClient()
        self.bq = bq_client or BigQueryClient()
        self.albums_table = albums_table

    async def build_album_row(
        self,
        album_ref: str,
    ) -> Dict[str, Any]:
        album = await self.spotify.get_album(album_ref)

        album_id = album.get("id")
        album_name = album.get("name")
        album_type = album.get("album_type")
        release_date = album.get("release_date")
        release_date_precision = album.get("release_date_precision")
        total_tracks = album.get("total_tracks")
        label = album.get("label")
        album_spotify_url = (album.get("external_urls") or {}).get("spotify")

        genres_list = album.get("genres") or []
        album_genres = ", ".join(genres_list) if genres_list else None

        album_image_url = None
        images = album.get("images") or []
        if images:
            album_image_url = images[0].get("url")

        album_available_markets = album.get("available_markets") or []
        album_markets_count = len(album_available_markets)

        return {
            "album_id": album_id,
            "album_name": album_name,
            "album_type": album_type,
            "album_release_date": release_date,
            "album_release_precision": release_date_precision,
            "album_total_tracks": total_tracks,
            "album_label": label,
            "album_spotify_url": album_spotify_url,
            "album_genres": album_genres,
            "album_image_url": album_image_url,
            "album_available_markets_count": album_markets_count,
        }

    async def ingest_album(
        self,
        album_ref: str,
    ) -> None:
        """
        Inserta o actualiza el álbum usando MERGE por album_id.
        """
        row = await self.build_album_row(album_ref)

        table_fq = self.bq.table(self.albums_table)

        sql = f"""
        MERGE `{table_fq}` AS T
        USING (
          SELECT
            @album_id                      AS album_id,
            @album_name                    AS album_name,
            @album_type                    AS album_type,
            @album_release_date            AS album_release_date,
            @album_release_precision       AS album_release_precision,
            @album_total_tracks            AS album_total_tracks,
            @album_label                   AS album_label,
            @album_spotify_url             AS album_spotify_url,
            @album_genres                  AS album_genres,
            @album_image_url               AS album_image_url,
            @album_available_markets_count AS album_available_markets_count
        ) AS S
        ON T.album_id = S.album_id
        WHEN MATCHED THEN
          UPDATE SET
            album_name                    = S.album_name,
            album_type                    = S.album_type,
            album_release_date            = S.album_release_date,
            album_release_precision       = S.album_release_precision,
            album_total_tracks            = S.album_total_tracks,
            album_label                   = S.album_label,
            album_spotify_url             = S.album_spotify_url,
            album_genres                  = S.album_genres,
            album_image_url               = S.album_image_url,
            album_available_markets_count = S.album_available_markets_count
        WHEN NOT MATCHED THEN
          INSERT (
            album_id,
            album_name,
            album_type,
            album_release_date,
            album_release_precision,
            album_total_tracks,
            album_label,
            album_spotify_url,
            album_genres,
            album_image_url,
            album_available_markets_count
          )
          VALUES (
            S.album_id,
            S.album_name,
            S.album_type,
            S.album_release_date,
            S.album_release_precision,
            S.album_total_tracks,
            S.album_label,
            S.album_spotify_url,
            S.album_genres,
            S.album_image_url,
            S.album_available_markets_count
          );
        """

        self.bq.execute(sql, params=row)
        print(f"[albums] MERGE OK para álbum {row['album_id']} ({row['album_name']})")
