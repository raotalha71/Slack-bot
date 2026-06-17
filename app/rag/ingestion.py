"""
Data ingestion for the RAG system.

Three ingestion paths:
  1. Seed proposals — loaded from seed_data/ at startup
  2. Transcripts — uploaded by users via Slack
  3. Generated proposals — saved back with dedup (Smart Save Gate)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.rag.chunking import Chunk, chunk_document
from app.rag.vector_db import (
    PROPOSALS_COLLECTION,
    TRANSCRIPTS_COLLECTION,
    QdrantManager,
)
from config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Seed Proposal Ingestion
# ---------------------------------------------------------------------------

def ingest_seed_proposals(
    qdrant: QdrantManager,
    seed_dir: str | None = None,
) -> int:
    """
    Load all .txt proposal files from seed_data/ into Qdrant.

    Each file is chunked with section-aware splitting and tagged
    with source="seed" and industry extracted from the filename.

    Returns total number of chunks ingested.
    """
    if seed_dir is None:
        seed_dir = get_settings().SEED_DATA_DIR

    if not os.path.isdir(seed_dir):
        logger.warning("Seed data directory not found: %s", seed_dir)
        return 0

    total_chunks = 0

    for filename in sorted(os.listdir(seed_dir)):
        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(seed_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            logger.warning("Empty file: %s", filename)
            continue

        chunks = chunk_document(
            text=text,
            source_file=filename,
            source="seed",
        )

        qdrant.upsert_chunks(PROPOSALS_COLLECTION, chunks)
        total_chunks += len(chunks)
        logger.info(
            "Ingested %s → %d chunks (industry: %s)",
            filename,
            len(chunks),
            chunks[0].industry if chunks else "unknown",
        )

    logger.info("Seed ingestion complete: %d total chunks", total_chunks)
    return total_chunks


# ---------------------------------------------------------------------------
# 2. Transcript Ingestion
# ---------------------------------------------------------------------------

def ingest_transcript(
    qdrant: QdrantManager,
    text: str,
    thread_ts: str,
    source_file: str = "uploaded_transcript",
) -> int:
    """
    Chunk and store an uploaded transcript for follow-up Q&A.

    Stored in the 'transcripts' collection so the Revision agent
    can search it when the user asks "what did the client say about X?"

    Returns number of chunks ingested.
    """
    chunks = chunk_document(
        text=text,
        source_file=source_file,
        source="transcript",
        thread_ts=thread_ts,
    )

    count = qdrant.upsert_chunks(TRANSCRIPTS_COLLECTION, chunks)
    logger.info(
        "Transcript ingested (thread_ts=%s, %d chunks)",
        thread_ts,
        count,
    )
    return count


# ---------------------------------------------------------------------------
# 3. Smart Save — Generated Proposal (with Dedup)
# ---------------------------------------------------------------------------

def smart_save_proposal(
    qdrant: QdrantManager,
    proposal_text: str,
    client_info: dict[str, Any],
    thread_ts: str,
) -> dict[str, Any]:
    """
    Save a generated proposal back into Qdrant — but only if it's novel.

    Smart Save Gate logic:
      1. Check for duplicates (similarity > threshold → skip)
      2. If novel → chunk and save with source="generated"
      3. Return {saved: bool, reason: str} for transparency

    Args:
        qdrant: QdrantManager instance.
        proposal_text: The full generated proposal markdown.
        client_info: Dict with at least 'industry' and 'company_name'.
        thread_ts: Slack thread timestamp (used as logical proposal ID).

    Returns:
        Dict with 'saved', 'reason', and optionally 'score'.
    """
    settings = get_settings()
    industry = client_info.get("industry", "general")
    company = client_info.get("company_name", "unknown")

    # Step 1: Dedup check
    is_dup, score, similar_source = qdrant.check_duplicate(
        proposal_text=proposal_text,
        industry=industry,
        threshold=settings.DEDUP_THRESHOLD,
    )

    if is_dup:
        reason = (
            f"Skipped save — similar to '{similar_source}' "
            f"(score: {score:.3f}, threshold: {settings.DEDUP_THRESHOLD})"
        )
        logger.info("Smart Save Gate: %s", reason)
        return {"saved": False, "reason": reason, "score": score}

    # Step 2: Chunk and save
    chunks = chunk_document(
        text=proposal_text,
        source_file=f"generated_{company}_{industry}",
        source="generated",
        thread_ts=thread_ts,
        industry=industry,
    )

    qdrant.save_generated_proposal(chunks)

    reason = f"Saved — novel proposal for {industry} (best match score: {score:.3f})"
    logger.info("Smart Save Gate: %s", reason)
    return {"saved": True, "reason": reason, "score": score}


# ---------------------------------------------------------------------------
# 4. Update Saved Proposal (After Revision)
# ---------------------------------------------------------------------------

def update_saved_proposal(
    qdrant: QdrantManager,
    thread_ts: str,
    updated_text: str,
    client_info: dict[str, Any],
) -> int:
    """
    Update an existing generated proposal in Qdrant after revision.

    Deletes old chunks for this thread_ts, re-chunks the updated text,
    and inserts the new chunks. No duplicate created.

    Returns number of new chunks inserted.
    """
    industry = client_info.get("industry", "general")
    company = client_info.get("company_name", "unknown")

    new_chunks = chunk_document(
        text=updated_text,
        source_file=f"generated_{company}_{industry}",
        source="generated",
        thread_ts=thread_ts,
        industry=industry,
    )

    count = qdrant.update_generated_proposal(thread_ts, new_chunks)
    logger.info(
        "Updated saved proposal (thread_ts=%s, %d chunks)",
        thread_ts,
        count,
    )
    return count
