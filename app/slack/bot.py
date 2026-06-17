"""
Slack bot initialization.

Supports both Socket Mode (development) and HTTP mode (production).
"""

from __future__ import annotations

import logging

from slack_bolt.async_app import AsyncApp

from config import get_settings

logger = logging.getLogger(__name__)

_app: AsyncApp | None = None


def get_slack_app() -> AsyncApp:
    """Get or initialize the Slack AsyncApp (singleton)."""
    global _app
    if _app is None:
        settings = get_settings()
        _app = AsyncApp(
            token=settings.SLACK_BOT_TOKEN,
            signing_secret=settings.SLACK_SIGNING_SECRET,
        )
        logger.info("Slack AsyncApp initialized")
    return _app
