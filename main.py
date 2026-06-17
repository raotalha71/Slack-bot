"""
Main FastAPI application entry point.

Startup:
  - Initialize SQLite database
  - Connect to Qdrant and create collections
  - Load embedding model
  - Ingest seed proposals (if not already ingested)
  - Mount Slack bolt handler

Routes:
  - /health    → health check
  - /slack/events → Slack events endpoint
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from app.rag.ingestion import ingest_seed_proposals
from app.rag.vector_db import PROPOSALS_COLLECTION, QdrantManager
from app.slack.bot import get_slack_app
from app.slack.handlers import register_handlers
from app.state.session_store import init_db
from config import get_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Slack AI Proposal Generator",
    description="Multi-agent AI system that generates business proposals from transcripts",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@api.on_event("startup")
async def startup() -> None:
    """Initialize all services on startup."""
    settings = get_settings()

    # 1. Initialize SQLite database
    logger.info("Initializing database...")
    init_db()
    logger.info("Database ready (SQLite at %s)", settings.DATABASE_URL)

    # 2. Connect to Qdrant and create collections
    logger.info("Connecting to Qdrant at %s:%d...", settings.QDRANT_HOST, settings.QDRANT_PORT)
    qdrant = QdrantManager()  # Also loads embedding model
    logger.info("Qdrant connected. Embedding model loaded: %s", settings.EMBEDDING_MODEL)

    # 3. Ingest seed proposals if collection is empty
    count = qdrant.collection_count(PROPOSALS_COLLECTION)
    if count == 0:
        logger.info("Ingesting seed proposals from %s...", settings.SEED_DATA_DIR)
        total = ingest_seed_proposals(qdrant, settings.SEED_DATA_DIR)
        logger.info("Seed ingestion complete: %d chunks", total)
    else:
        logger.info("Qdrant already has %d proposal chunks — skipping seed ingestion", count)

    # 4. Register Slack handlers
    slack_app = get_slack_app()
    register_handlers(slack_app)
    logger.info("Slack handlers registered")

    logger.info("🚀 Slack AI Proposal Generator is ready!")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@api.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "ok",
            "service": "Slack AI Proposal Generator",
        }
    )


@api.post("/slack/events")
async def slack_events(request: Request) -> JSONResponse:
    """Slack events endpoint — receives and processes all Slack events."""
    slack_app = get_slack_app()
    handler = AsyncSlackRequestHandler(slack_app)
    return await handler.handle(request)
