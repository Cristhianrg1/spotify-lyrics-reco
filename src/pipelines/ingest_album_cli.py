import asyncio
import os

from src.services.album_ingest_service import ingest_album


async def main() -> None:
    # Puedes leer de env, de args, etc. Por ahora hardcodeado o desde env:
    album_ref = os.getenv("ALBUM_REF") or "4aawyAB9vmqN3uQ7FjRGTy"
    mode = os.getenv("ALBUM_MODE") or "focus"

    await ingest_album(album_ref, mode=mode)


if __name__ == "__main__":
    asyncio.run(main())
