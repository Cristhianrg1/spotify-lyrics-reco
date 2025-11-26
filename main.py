from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.ingest_album_pipeline import ingest_album_pipeline
from src.services.lyrics_search_service import LyricsSearchService


import logging
from src.config.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spotify Lyrics Ingestion API",
    version="0.1.0",
    description="API para disparar el pipeline de ingesta de álbum + tracks + letras",
)


class IngestAlbumRequest(BaseModel):
    album_ref: str  # puede ser ID o URL de Spotify


class IngestAlbumResponse(BaseModel):
    album_id: str
    album_name: str | None = None
    tracks_ingested: int
    has_lyrics_count: int | None = None  # cuántos tracks con letra
    message: str


class SearchLyricsRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchLyricsHit(BaseModel):
    track_id: str | None = None
    track_name: str | None = None
    artists: str | None = None
    album_id: str | None = None
    chunk_text: str
    mode: str | None = None
    score: float


class SearchLyricsResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchLyricsHit]


@app.get("/ping")
async def ping():
    """
    Endpoint simple de healthcheck.
    Útil para Cloud Run, load balancers, etc.
    """
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "service": "spotify-lyrics-reco",
        "version": "0.1.0",
        "endpoints": ["/ping", "/ingest_album"],
    }


@app.post("/ingest_album", response_model=IngestAlbumResponse)
async def ingest_album_endpoint(payload: IngestAlbumRequest):
    """
    Dispara el pipeline de ingesta:
      - Álbum → BigQuery (tabla albums)
      - Tracks → BigQuery (tabla tracks)
      - Letras → Mongo
      - (opcional) Embeddings → lyrics_chunks
    """
    try:
        result = await ingest_album_pipeline(
            album_ref=payload.album_ref,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"ingest_album_endpoint failed: {repr(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return IngestAlbumResponse(**result)


@app.post("/search_lyrics", response_model=SearchLyricsResponse)
async def search_lyrics_endpoint(payload: SearchLyricsRequest):
    """
    Búsqueda semántica sobre letras (collection lyrics_chunks en Mongo).

    - Genera embedding del query.
    - Hace vector search en Mongo Atlas.
    - Devuelve los mejores chunks con score.
    """
    try:
        service = LyricsSearchService()
        hits = await service.search(
            query=payload.query,
            top_k=payload.top_k,
        )
    except Exception as e:
        logger.error(f"search_lyrics_endpoint failed: {repr(e)}")
        raise HTTPException(status_code=500, detail="Vector search failed")

    return SearchLyricsResponse(
        query=payload.query,
        top_k=payload.top_k,
        results=[SearchLyricsHit(**h) for h in hits],
    )


# --- Album Analysis Endpoints ---

from src.services.album_analysis_service import AlbumAnalysisService, AnalysisNotFound
from src.models.album_analysis import AlbumAnalysis

@app.get("/albums/{album_id}/analysis", response_model=AlbumAnalysis)
async def get_album_analysis(
    album_id: str,
    create_if_missing: bool = False,
    force_recompute: bool = False
):
    """
    Obtiene el análisis de un álbum (topics, sentiment, similarity, etc.).
    Si no existe y create_if_missing=True, lo genera on-demand.
    """
    service = AlbumAnalysisService()
    try:
        analysis = await service.get_or_create_analysis(
            album_id=album_id,
            create_if_missing=create_if_missing,
            force_recompute=force_recompute
        )
        return analysis
    except AnalysisNotFound:
        raise HTTPException(status_code=404, detail="Analysis not found. Use create_if_missing=true to generate.")
    except Exception as e:
        logger.error(f"get_album_analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RebuildAnalysisRequest(BaseModel):
    recompute_topics: bool = True
    recompute_sentiment: bool = True
    recompute_similarity: bool = True


@app.post("/albums/{album_id}/analysis/rebuild", response_model=AlbumAnalysis)
async def rebuild_album_analysis(
    album_id: str,
    payload: RebuildAnalysisRequest
):
    """
    Fuerza el re-cálculo del análisis de un álbum.
    """
    service = AlbumAnalysisService()
    try:
        # Por ahora ignoramos los flags finos y recomputamos todo
        analysis = await service.get_or_create_analysis(
            album_id=album_id,
            create_if_missing=True,
            force_recompute=True
        )
        return analysis
    except Exception as e:
        logger.error(f"rebuild_album_analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
