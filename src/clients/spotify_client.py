from __future__ import annotations

from typing import AsyncIterator, Dict, Any, List

import httpx

from src.config.settings import get_settings

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


class SpotifyClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.client_id = settings.spotify_client_id
        self.client_secret = settings.spotify_client_secret
        self._token: str | None = None

    # ----------------- helpers internos -----------------

    @staticmethod
    def normalize_album_id(raw: str) -> str:
        """
        Acepta tanto un ID '4aawyAB9vmqN3uQ7FjRGTy' como una URL
        'https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy?si=...'
        y devuelve solo el ID.
        """
        if "open.spotify.com/album/" in raw:
            return raw.split("album/")[1].split("?")[0]
        return raw

    @staticmethod
    def _chunked(items: List[str], n: int) -> List[List[str]]:
        return [items[i : i + n] for i in range(0, len(items), n)]

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

    async def _get(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
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

    # ----------------- API pública (solo álbum / tracks) -----------------

    async def get_album(self, album_ref: str) -> Dict[str, Any]:
        """
        Devuelve el JSON completo del álbum a partir de un ID o URL.
        """
        album_id = self.normalize_album_id(album_ref)
        return await self._get(f"/albums/{album_id}")

    async def iter_album_tracks(
        self,
        album_ref: str,
        limit: int = 50,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Itera sobre los tracks de un álbum.
        Devuelve cada track como dict (objeto 'track' parcial de Spotify).
        """
        album_id = self.normalize_album_id(album_ref)

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

    async def get_tracks_by_ids(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Dado un listado de track_ids, devuelve la lista de objetos track
        completos (vía /tracks), manejando el chunking (50 por request).
        """
        all_tracks: List[Dict[str, Any]] = []

        for batch in self._chunked(track_ids, 50):
            ids_param = ",".join(batch)
            data = await self._get("/tracks", params={"ids": ids_param})
            tracks = data.get("tracks") or []
            all_tracks.extend(tracks)

        return all_tracks
