"""
Qdrant vector database manager.

Handles two collections:
  - past_proposals: Seed proposals + generated proposals (for RAG)
  - transcripts: Uploaded transcripts (for follow-up Q&A)

Includes dedup check (Smart Save Gate) and revision update logic.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from app.rag.chunking import Chunk
from config import get_settings

logger = logging.getLogger(__name__)

# Collection names
PROPOSALS_COLLECTION = "past_proposals"
TRANSCRIPTS_COLLECTION = "transcripts"


class QdrantManager:
    """
    Wrapper around the Qdrant client.

    Manages embedding, upserting, searching, dedup checking,
    and revision updates across two collections.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()
        self._ensure_collections()

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    def _ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        for name in [PROPOSALS_COLLECTION, TRANSCRIPTS_COLLECTION]:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", name)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using sentence-transformers."""
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def _embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embedder.encode(text, show_progress_bar=False).tolist()

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        collection: str,
        chunks: list[Chunk],
    ) -> int:
        """
        Embed and upsert a list of chunks into a Qdrant collection.

        Returns the number of points upserted.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self._embed(texts)

        points = [
            PointStruct(
                id=c.id,
                vector=vec,
                payload={
                    "text": c.text,
                    "source_file": c.source_file,
                    "section_title": c.section_title,
                    "chunk_index": c.chunk_index,
                    "industry": c.industry,
                    "source": c.source,
                    "thread_ts": c.thread_ts,
                },
            )
            for c, vec in zip(chunks, vectors)
        ]

        self.client.upsert(
            collection_name=collection,
            points=points,
        )

        logger.info(
            "Upserted %d chunks into %s", len(points), collection,
        )
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        collection: str,
        query: str,
        top_k: int = 3,
        filters: Optional[Filter] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search in a Qdrant collection.

        Returns a list of dicts with: text, score, and all metadata.
        """
        query_vector = self._embed_single(query)

        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filters,
        )

        return [
            {
                "id": str(r.id),
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source_file": r.payload.get("source_file", ""),
                "section_title": r.payload.get("section_title", ""),
                "industry": r.payload.get("industry", ""),
                "source": r.payload.get("source", ""),
                "thread_ts": r.payload.get("thread_ts", ""),
            }
            for r in results
        ]

    def search_with_metadata_filter(
        self,
        collection: str,
        query: str,
        industry: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search with metadata filtering — filter by industry and/or source.

        Falls back to unfiltered search if filtered results are empty.
        """
        conditions = []
        if industry:
            conditions.append(
                FieldCondition(
                    key="industry",
                    match=MatchValue(value=industry.lower()),
                )
            )
        if source:
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = self.search_similar(
            collection=collection,
            query=query,
            top_k=top_k,
            filters=query_filter,
        )

        # Fallback: if filtered search returns nothing, try without filter
        if not results and query_filter is not None:
            logger.info(
                "No results with filter (industry=%s, source=%s). "
                "Falling back to unfiltered search.",
                industry,
                source,
            )
            results = self.search_similar(
                collection=collection,
                query=query,
                top_k=top_k,
            )

        return results

    # ------------------------------------------------------------------
    # Smart Save Gate: Dedup Check
    # ------------------------------------------------------------------

    def check_duplicate(
        self,
        proposal_text: str,
        industry: str,
        threshold: float = 0.85,
    ) -> tuple[bool, float, str]:
        """
        Check if a similar proposal already exists for this industry.

        Args:
            proposal_text: The generated proposal text.
            industry: The industry to check within.
            threshold: Similarity score above which we consider it a duplicate.

        Returns:
            Tuple of (is_duplicate, highest_score, most_similar_source).
        """
        # Use the first ~500 chars as the query (representative excerpt)
        query = proposal_text[:500]

        results = self.search_with_metadata_filter(
            collection=PROPOSALS_COLLECTION,
            query=query,
            industry=industry,
            top_k=1,
        )

        if not results:
            return False, 0.0, ""

        top = results[0]
        is_dup = top["score"] > threshold

        if is_dup:
            logger.info(
                "Dedup: Similar proposal found (score=%.3f, source=%s). Skipping save.",
                top["score"],
                top["source_file"],
            )
        else:
            logger.info(
                "Dedup: No duplicate found (best score=%.3f). Will save.",
                top["score"],
            )

        return is_dup, top["score"], top.get("source_file", "")

    # ------------------------------------------------------------------
    # Smart Save Gate: Save Generated Proposal
    # ------------------------------------------------------------------

    def save_generated_proposal(
        self,
        chunks: list[Chunk],
    ) -> str:
        """
        Save a generated proposal's chunks to the past_proposals collection.

        Returns the thread_ts used as the logical proposal ID.
        """
        if not chunks:
            return ""

        self.upsert_chunks(PROPOSALS_COLLECTION, chunks)
        thread_ts = chunks[0].thread_ts
        logger.info(
            "Saved generated proposal (thread_ts=%s, %d chunks)",
            thread_ts,
            len(chunks),
        )
        return thread_ts

    # ------------------------------------------------------------------
    # Smart Save Gate: Update After Revision
    # ------------------------------------------------------------------

    def update_generated_proposal(
        self,
        thread_ts: str,
        new_chunks: list[Chunk],
    ) -> int:
        """
        Update an existing generated proposal in Qdrant.

        Deletes old chunks by thread_ts, then inserts new chunks.
        This prevents duplicates when a proposal is revised.

        Returns the number of new chunks inserted.
        """
        # Delete old chunks for this thread
        self.client.delete(
            collection_name=PROPOSALS_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="thread_ts",
                        match=MatchValue(value=thread_ts),
                    ),
                    FieldCondition(
                        key="source",
                        match=MatchValue(value="generated"),
                    ),
                ],
            ),
        )
        logger.info("Deleted old chunks for thread_ts=%s", thread_ts)

        # Insert updated chunks
        count = self.upsert_chunks(PROPOSALS_COLLECTION, new_chunks)
        logger.info(
            "Updated proposal (thread_ts=%s, %d new chunks)",
            thread_ts,
            count,
        )
        return count

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def collection_count(self, collection: str) -> int:
        """Get the number of points in a collection."""
        info = self.client.get_collection(collection)
        return info.points_count or 0
