import asyncio

from src.services.album_service import AlbumService
from src.services.track_service import TrackService


async def pipeline(album_ref="4aawyAB9vmqN3uQ7FjRGTy"):

    album_service = AlbumService()
    track_service = TrackService(with_lyrics=True)

    await album_service.ingest_album(album_ref)
    await track_service.ingest_tracks_for_album(album_ref)


if __name__ == "__main__":
    asyncio.run(pipeline())
