#!/usr/bin/env python3
"""
One-time script to ingest seed proposals into Qdrant.

Usage:
    python scripts/ingest_proposals.py

Reads all .txt files from seed_data/ and loads them into
the 'past_proposals' Qdrant collection.
"""

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.vector_db import QdrantManager, PROPOSALS_COLLECTION
from app.rag.ingestion import ingest_seed_proposals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load seed proposals into Qdrant."""
    logger.info("Connecting to Qdrant...")
    qdrant = QdrantManager()

    # Check if already ingested
    count = qdrant.collection_count(PROPOSALS_COLLECTION)
    if count > 0:
        logger.info(
            "Collection '%s' already has %d points. Skipping ingestion.",
            PROPOSALS_COLLECTION,
            count,
        )
        logger.info("To re-ingest, delete the collection first.")
        return

    logger.info("Ingesting seed proposals...")
    total = ingest_seed_proposals(qdrant)

    if total == 0:
        logger.warning(
            "No proposals ingested. Make sure seed_data/ contains .txt files."
        )
    else:
        logger.info("Done! Ingested %d chunks into Qdrant.", total)


if __name__ == "__main__":
    main()
