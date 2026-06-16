"""
Embedding client for memory semantic retrieval.
Calls an OpenAI-compatible /embeddings endpoint (e.g. SiliconFlow's BAAI/bge-m3).
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import os
from typing import Optional

import numpy as np

from finagent.config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM,
    MAX_RETRIES, RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)

# Max simultaneous embedding API calls — prevents hammering the endpoint when
# many training windows are in flight concurrently.
_EMBED_CONCURRENCY = int(os.getenv("EMBEDDING_CONCURRENCY", "3"))
# Per-request timeout (seconds) for embedding API calls.
_EMBED_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "30"))

# Module-level semaphore — shared across all Embedder instances so the limit
# is global, not per-instance. Created lazily inside the running event loop.
_global_sem: Optional[asyncio.Semaphore] = None


def _get_global_sem() -> asyncio.Semaphore:
    global _global_sem
    if _global_sem is None:
        _global_sem = asyncio.Semaphore(_EMBED_CONCURRENCY)
    return _global_sem


class Embedder:
    """
    Thin wrapper around the OpenAI-compat embeddings endpoint.

    Sync-friendly: `embed(text)` and `embed_batch(texts)` can be called from
    both sync and async contexts. Internally uses `asyncio.run` if called
    from sync code, or the caller's event loop if already async.
    """

    def __init__(
        self,
        api_key: str = EMBEDDING_API_KEY,
        base_url: str = EMBEDDING_BASE_URL,
        model: str = EMBEDDING_MODEL,
        dim: int = EMBEDDING_DIM,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}
        self._client = None  # lazy

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=_EMBED_TIMEOUT,
            )
        return self._client

    async def aembed(self, text: str) -> np.ndarray:
        k = self._key(text)
        if k in self._cache:
            return self._cache[k]
        vec = (await self.aembed_batch([text]))[0]
        return vec

    async def aembed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        if not self.configured:
            raise RuntimeError(
                "Embedder not configured: set EMBEDDING_API_KEY and EMBEDDING_BASE_URL "
                "(or OPENAI_COMPAT_API_KEY / OPENAI_COMPAT_BASE_URL as fallback)."
            )

        keys = [self._key(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            vectors = await self._call_with_retry(missing_texts)
            for i, v in zip(missing_idx, vectors):
                self._cache[keys[i]] = v

        return [self._cache[k] for k in keys]

    async def _call_with_retry(self, texts: list[str]) -> list[np.ndarray]:
        client = self._get_client()
        last_err: Optional[BaseException] = None
        async with _get_global_sem():  # limit concurrent embedding calls globally
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.embeddings.create(model=self.model, input=texts)
                    out = []
                    for item in resp.data:
                        arr = np.asarray(item.embedding, dtype=np.float32)
                        out.append(arr)
                    if out and out[0].shape[0] != self.dim:
                        logger.warning(
                            f"Embedding dim drift: configured EMBEDDING_DIM={self.dim} but "
                            f"provider returned {out[0].shape[0]}. Update EMBEDDING_DIM env var."
                        )
                        self.dim = out[0].shape[0]
                    return out
                except Exception as e:
                    last_err = e
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Embedding call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
        raise RuntimeError(f"Embedding call failed after {MAX_RETRIES} attempts: {last_err}")

    def embed(self, text: str) -> np.ndarray:
        """Sync wrapper. Uses asyncio.run if no loop is running; else raises."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aembed(text))
        raise RuntimeError("Embedder.embed() called from async context — use aembed() instead.")

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aembed_batch(texts))
        raise RuntimeError("Embedder.embed_batch() called from async context — use aembed_batch() instead.")


_default: Optional[Embedder] = None


def get_default_embedder() -> Embedder:
    """Module-level singleton, safe to share across MemoryManager instances."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default
