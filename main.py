import asyncio
from typing import List, Dict, Any

import pandas as pd

from src.config.spotify_client import SpotifyClient


async def fetch_tracks_for_album(
    album_id: str,
    mode: str = "unknown",
) -> List[Dict[str, Any]]:
    client = SpotifyClient()
    rows: List[Dict[str, Any]] = []

    # Info del álbum (una sola llamada)
    album = await client._get(f"/albums/{album_id}")
    album_name = album.get("name")
    album_type = album.get("album_type")
    release_date = album.get("release_date")
    total_tracks = album.get("total_tracks")
    label = album.get("label")
    album_spotify_url = (album.get("external_urls") or {}).get("spotify")

    # géneros del álbum (puede venir vacío)
    album_genres = album.get("genres") or []

    # portada del álbum (tomamos la primera imagen si existe)
    album_image_url = None
    images = album.get("images") or []
    if images:
        album_image_url = images[0].get("url")

    async for track in client.iter_album_tracks(album_id):
        if not track:
            continue

        track_id = track.get("id")
        name = track.get("name")
        artists = ", ".join(a["name"] for a in track.get("artists", []))
        spotify_url = (track.get("external_urls") or {}).get("spotify")
        preview_url = track.get("preview_url")

        duration_ms = track.get("duration_ms")
        track_number = track.get("track_number")
        disc_number = track.get("disc_number")
        explicit = track.get("explicit")

        # identificadores externos
        external_ids = track.get("external_ids") or {}
        isrc = external_ids.get("isrc")

        # extras de track
        popularity = track.get("popularity")
        uri = track.get("uri")
        available_markets = track.get("available_markets") or []
        available_markets_count = len(available_markets)

        rows.append(
            {
                # Álbum
                "album_id": album_id,
                "album_name": album_name,
                "album_type": album_type,
                "album_release_date": release_date,
                "album_total_tracks": total_tracks,
                "album_label": label,
                "album_spotify_url": album_spotify_url,
                "album_genres": album_genres,
                "album_image_url": album_image_url,

                # Track
                "track_id": track_id,
                "name": name,
                "artists": artists,
                "spotify_url": spotify_url,
                "preview_url": preview_url,
                "duration_ms": duration_ms,
                "track_number": track_number,
                "disc_number": disc_number,
                "explicit": explicit,
                "isrc": isrc,
                "popularity": popularity,
                "available_markets_count": available_markets_count,
                "uri": uri,

                # Tag manual
                "mode": mode,
            }
        )

    return rows




async def _demo():
    # ⚠️ Pon aquí el album_id que ya probaste que funciona
    TEST_ALBUM_ID = "4aawyAB9vmqN3uQ7FjRGTy"

    rows = await fetch_tracks_for_album(TEST_ALBUM_ID, mode="focus")
    df = pd.DataFrame(rows)
    print(df.head())

    print(f"Tracks obtenidos: {len(rows)}\n")

    # Mostrar los primeros 5 para revisar
    for i, row in enumerate(rows[:5], start=1):
        print(
            f"{i:02d}. {row['name']} - {row['artists']}"
            f" | track_id={row['track_id']}"
        )


if __name__ == "__main__":
    asyncio.run(_demo())

