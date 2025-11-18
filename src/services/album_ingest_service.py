from __future__ import annotations

import asyncio
from typing import Iterable, List, Dict, Any

import pandas as pd

from src.config.spotify_client import SpotifyClient
from src.config.bigquery_client import BigQueryClient


def normalize_album_id(raw: str) -> str:
    """
    Acepta tanto un ID '4aawyAB9vmqN3uQ7FjRGTy' como una URL
    'https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy?si=...'
    y devuelve solo el ID.
    """
    if "open.spotify.com/album/" in raw:
        return raw.split("album/")[1].split("?")[0]
    return raw


def chunked(iterable: Iterable[str], n: int) -> List[List[str]]:
    """
    Divide una lista en chunks de tamaño n.
    """
    items = list(iterable)
    return [items[i : i + n] for i in range(0, len(items), n)]


async def fetch_album_row(
    spotify: SpotifyClient,
    album_id: str,
    mode: str = "unknown",
) -> Dict[str, Any]:
    """
    Extrae la metadata del álbum y construye un row para la tabla `albums`.
    """
    album = await spotify._get(f"/albums/{album_id}")

    album_name = album.get("name")
    album_type = album.get("album_type")
    release_date = album.get("release_date")
    release_date_precision = album.get("release_date_precision")
    total_tracks = album.get("total_tracks")
    label = album.get("label")
    album_spotify_url = (album.get("external_urls") or {}).get("spotify")

    album_genres = album.get("genres") or []

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
        "mode": mode,
    }


async def fetch_album_track_ids(
    spotify: SpotifyClient,
    album_id: str,
) -> List[str]:
    """
    Devuelve los track_ids de un álbum usando iter_album_tracks.
    """
    track_ids: List[str] = []
    async for t in spotify.iter_album_tracks(album_id):
        if t and t.get("id"):
            track_ids.append(t["id"])
    return track_ids


async def fetch_track_rows_via_tracks_endpoint(
    spotify: SpotifyClient,
    track_ids: List[str],
    mode: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Dado un conjunto de track_ids, llama /tracks en batches y construye rows
    detallados para la tabla `tracks`.
    """
    all_track_rows: List[Dict[str, Any]] = []

    for batch in chunked(track_ids, 50):
        ids_param = ",".join(batch)
        data = await spotify._get("/tracks", params={"ids": ids_param})
        tracks = data.get("tracks") or []

        for track in tracks:
            if not track:
                continue

            t_id = track.get("id")
            t_name = track.get("name")
            t_artists = ", ".join(a["name"] for a in track.get("artists", []))

            t_album = track.get("album") or {}
            t_album_id = t_album.get("id")
            t_album_name = t_album.get("name")
            t_album_release_date = t_album.get("release_date")
            t_album_release_precision = t_album.get("release_date_precision")

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

            row = {
                # Link al álbum
                "album_id": t_album_id,
                "album_name": t_album_name,
                "album_release_date": t_album_release_date,
                "album_release_precision": t_album_release_precision,

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

                # Tag manual
                "mode": mode,
            }
            all_track_rows.append(row)

    return all_track_rows


async def ingest_album(
    album_ref: str,
    mode: str = "unknown",
    albums_table: str = "albums",
    tracks_table: str = "tracks",
) -> None:
    """
    Punto de entrada principal:

    - Recibe ID o URL de álbum.
    - Guarda metadata del álbum en `albums`.
    - Guarda tracks enriquecidos en `tracks`.
    """
    spotify = SpotifyClient()
    bq = BigQueryClient()

    album_id = normalize_album_id(album_ref)

    # 1) Álbum
    album_row = await fetch_album_row(spotify, album_id, mode=mode)
    df_albums = pd.DataFrame([album_row])
    bq.load_dataframe(albums_table, df_albums, write_disposition="WRITE_APPEND")
    print(f"[albums] Insertado álbum {album_id} ({album_row['album_name']})")

    # 2) Track IDs
    track_ids = await fetch_album_track_ids(spotify, album_id)
    if not track_ids:
        print(f"[tracks] El álbum {album_id} no tiene tracks o no se pudieron leer.")
        return

    print(f"[tracks] Encontrados {len(track_ids)} track_ids para el álbum {album_id}")

    # 3) Detalle de tracks vía /tracks
    track_rows = await fetch_track_rows_via_tracks_endpoint(
        spotify, track_ids, mode=mode
    )
    if not track_rows:
        print(f"[tracks] No se construyeron rows para tracks de {album_id}")
        return

    df_tracks = pd.DataFrame(track_rows)
    bq.load_dataframe(tracks_table, df_tracks, write_disposition="WRITE_APPEND")
    print(f"[tracks] Insertados {len(track_rows)} tracks para el álbum {album_id}")
