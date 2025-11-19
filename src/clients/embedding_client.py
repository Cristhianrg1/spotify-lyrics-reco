# src/clients/embedding_client.py

from __future__ import annotations

from typing import List

import httpx
import asyncio

from src.config.settings import get_settings


class EmbeddingClient:
    """
    Wrapper para embeddings con dos proveedores:
    - OpenAI (API)
    - Hugging Face Inference API
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = (self.settings.embedding_provider or "openai").lower()

        if self.provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY no está configurado en .env")
        elif self.provider == "huggingface_api":
            if not self.settings.hf_api_key:
                raise ValueError("HF_API_KEY no está configurado en .env")
        else:
            raise ValueError(
                f"Proveedor de embeddings no soportado: {self.provider} "
                "(usa 'openai' o 'huggingface_api')"
            )

    async def embed(self, text: str) -> List[float]:
        """
        Devuelve un solo vector (lista de floats) para el texto dado.
        """
        if self.provider == "openai":
            return await self._embed_openai(text)
        elif self.provider == "huggingface_api":
            return await self._embed_hf_api(text)
        else:
            raise RuntimeError("Proveedor de embeddings no inicializado correctamente")

    # ---------- OpenAI ----------

    async def _embed_openai(self, text: str) -> List[float]:
        """
        Llamada directa a la API de OpenAI.
        """
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": text,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        vec = data["data"][0]["embedding"]
        return vec

    # ---------- Hugging Face Inference API ----------

    async def _embed_hf_api(self, text: str) -> List[float]:
        """
        Usa la Hugging Face Inference API (router) con el pipeline
        'feature-extraction' para obtener embeddings.

        Modelo típico: 'sentence-transformers/all-MiniLM-L6-v2'
        """
        model = self.settings.hf_embedding_model
        url = (
            f"https://router.huggingface.co/hf-inference/"
            f"models/{model}/pipeline/feature-extraction"
        )
        headers = {
            "Authorization": f"Bearer {self.settings.hf_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": text}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)

        if resp.status_code != 200:
            print("==== HF API ERROR ====")
            print("Status code:", resp.status_code)
            try:
                print("Response JSON:", resp.json())
            except Exception:
                print("Response text:", resp.text)
            print("==== FIN HF API ERROR ====")
            resp.raise_for_status()

        data = resp.json()

        # Normalizamos recursivamente hasta quedarnos con un vector
        def extract_embedding(obj) -> List[float]:
            # Caso 1: lista de floats -> vector
            if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                return [float(x) for x in obj]

            # Caso 2: lista de listas -> extraer embeddings de cada sublista y promediar
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                # Extraemos un vector por cada sublista
                sub_vecs = [extract_embedding(sub) for sub in obj]
                if not sub_vecs:
                    raise ValueError("HF devolvió lista vacía al generar embeddings")

                dim = len(sub_vecs[0])
                # Promedio simple de todos los vectores
                pooled: List[float] = []
                for j in range(dim):
                    s = 0.0
                    for v in sub_vecs:
                        s += v[j]
                    pooled.append(s / len(sub_vecs))
                return pooled

            raise ValueError(f"Forma no reconocida en respuesta de HF API: {obj}")

        embedding = extract_embedding(data)
        return embedding

