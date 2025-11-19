from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class LyricsClient:
    """
    Cliente simple para un proveedor tipo LRCLIB.
    Ajusta BASE_URL y nombres de campos según el proveedor real.
    """

    BASE_URL = "https://lrclib.net/api"

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def search_lyrics(
        self,
        track_name: str,
        artist_name: str,
    ) -> Dict[str, Any]:
        params = {
            "track_name": track_name,
            "artist_name": artist_name,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self.BASE_URL}/search", params=params)
            if resp.status_code != 200:
                return {}

            data = resp.json()
            if not data:
                return {}

            # LRCLIB devuelve una lista; nos quedamos con el primer match
            best = data[0]
        return best

    async def get_lyrics_texts(
        self,
        track_name: str,
        artist_name: str,
    ) -> Dict[str, Optional[str]]:
        hit = await self.search_lyrics(track_name, artist_name)
        if not hit:
            return {
                "lyrics_text": None,
                "synced_lrc": None,
                "language": None,
            }

        lyrics_text = hit.get("plainLyrics")
        synced_lrc = hit.get("syncedLyrics")
        language = hit.get("language")

        return {
            "lyrics_text": lyrics_text,
            "synced_lrc": synced_lrc,
            "language": language,
        }
