import asyncio
from typing import AsyncIterator, Dict, Any, List

import httpx
from .settings import get_settings

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def normalize_album_id(raw: str) -> str:
    """
    Acepta tanto un ID '4aawyAB9vmqN3uQ7FjRGTy' como una URL
    'https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy?si=...'
    y devuelve solo el ID.
    """
    if "open.spotify.com/album/" in raw:
        return raw.split("album/")[1].split("?")[0]
    return raw


class SpotifyClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.client_id = settings.spotify_client_id
        self.client_secret = settings.spotify_client_secret
        self._token: str | None = None

    async def _get_token(self) -> str:
        if self._token is not None:
            return self._token

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                SPOTIFY_TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.client_secret),
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            return self._token

    async def _get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        token = await self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SPOTIFY_API_BASE}{path}",
                headers=headers,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def iter_playlist_tracks(
        self,
        playlist_id: str,
        limit: int = 100,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Itera sobre todos los tracks de una playlist.
        Devuelve cada track como dict (el objeto 'track' de Spotify).
        """
        offset = 0
        while True:
            data = await self._get(
                f"/playlists/{playlist_id}/tracks",
                params={"limit": limit, "offset": offset},
            )
            items: List[Dict[str, Any]] = data.get("items", [])
            if not items:
                break

            for item in items:
                track = item.get("track")
                if track:
                    yield track

            if data.get("next") is None:
                break

            offset += limit

    async def iter_album_tracks(
            self,
            album_id: str,
            limit: int = 50,
    ):
        """
        Itera sobre los tracks de un álbum.
        Devuelve cada track como dict (objeto 'track' de Spotify).
        """
        album_id = normalize_album_id(album_id)

        offset = 0
        while True:
            data = await self._get(
                f"/albums/{album_id}/tracks",
                params={"limit": limit, "offset": offset},
            )
            items = data.get("items", [])
            if not items:
                break

            for track in items:
                yield track

            if data.get("next") is None:
                break

            offset += limit


# Pequeño test manual
async def _demo():
    client = SpotifyClient()
    playlist_id = "37i9dQZF1DXcBWIGoYBM5M"  # Today's Top Hits
    count = 0
    async for track in client.iter_playlist_tracks(playlist_id):
        print(track["name"], " - ", ", ".join(a["name"] for a in track["artists"]))
        count += 1
        if count >= 5:
            break

if __name__ == "__main__":
    asyncio.run(_demo())
