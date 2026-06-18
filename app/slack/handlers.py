"""
Slack event handlers.

Handles:
  1. file_shared — new transcript uploaded → start pipeline
  2. message    — thread reply → route to correct pipeline stage
"""

from __future__ import annotations

import asyncio
import logging

from slack_bolt.async_app import AsyncApp

from app.services.proposal_service import ProposalService
from app.slack.bot import get_slack_app
from app.state.session_store import get_session, get_sessions_by_user, session_exists
from config import get_settings

logger = logging.getLogger(__name__)

# Accepted transcript file types
ACCEPTED_TYPES = {"text/plain", "text/markdown", "text/csv", "application/octet-stream"}
ACCEPTED_EXTENSIONS = {".txt", ".md", ".csv"}


def register_handlers(app: AsyncApp) -> None:
    """Register all Slack event handlers on the app."""

    # ------------------------------------------------------------------
    # Handler 1: File uploaded → start pipeline
    # ------------------------------------------------------------------
    @app.event("file_shared")
    async def handle_file_shared(event: dict, client, say) -> None:
        """
        Triggered when a user shares a file in a channel.

        Flow:
          1. Get file metadata
          2. Check it's a text file
          3. Download content
          4. Check for existing session (cross-session persistence)
          5. Start the pipeline
        """
        file_id = event.get("file_id")
        channel_id = event.get("channel_id")
        user_id = event.get("user_id")

        if not file_id or not channel_id:
            return

        try:
            # Get file metadata
            file_info = await client.files_info(file=file_id)
            file_obj = file_info.get("file", {})

            filename = file_obj.get("name", "")
            mimetype = file_obj.get("mimetype", "")

            # Check file type
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if mimetype not in ACCEPTED_TYPES and ext not in ACCEPTED_EXTENSIONS:
                logger.info(
                    "Ignoring non-text file: %s (%s)", filename, mimetype
                )
                return

            # Download file content
            url = file_obj.get("url_private_download") or file_obj.get("url_private")
            if not url:
                logger.error("No download URL for file: %s", file_id)
                return

            settings = get_settings()
            import httpx
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    url,
                    headers={"Authorization": f"Bearer {settings.SLACK_BOT_TOKEN}"},
                )
                transcript_text = resp.text

            if not transcript_text.strip():
                await say(
                    channel=channel_id,
                    text="⚠️ The uploaded file appears to be empty. Please try again.",
                )
                return

            # Get thread_ts (use file_id as thread key if no explicit thread)
            thread_ts = event.get("event_ts", file_id)

            # Check for existing session for this user
            existing_sessions = []
            from app.state.session_store import get_sessions_by_user
            if user_id:
                existing_sessions = get_sessions_by_user(user_id)

            if existing_sessions and not session_exists(thread_ts):
                await say(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=(
                        f"📂 Welcome back! I found {len(existing_sessions)} previous session(s).\n"
                        "Starting a new session with this transcript."
                    ),
                )

            # Acknowledge immediately
            await say(
                channel=channel_id,
                thread_ts=thread_ts,
                text=(
                    f"📄 Got it! Processing *{filename}*...\n"
                    "🔍 Extracting client information from the transcript."
                ),
            )

            # Start pipeline in background (non-blocking)
            service = ProposalService(client, channel_id, thread_ts, user_id)
            asyncio.create_task(
                service.run_generation_pipeline(transcript_text)
            )

        except Exception as e:
            logger.error("file_shared handler error: %s", e, exc_info=True)
            await say(
                channel=channel_id,
                text=f"❌ Error processing file: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Handler 2: Message in thread → route to correct stage
    # ------------------------------------------------------------------
    @app.event("message")
    async def handle_message(event: dict, client, say) -> None:
        """
        Triggered on any message in channels.

        Routes based on session status:
          - awaiting_confirmation → resume pipeline (intake confirmed)
          - awaiting_search_decision → resume pipeline (web search yes/no)
          - awaiting_tone → resume pipeline (tone selected)
          - complete → route to revision agent
          - (no session) → ignore
        """
        # Ignore bot messages and non-thread messages
        if event.get("bot_id") or event.get("subtype"):
            logger.info("Message ignored: bot_id=%s, subtype=%s", event.get("bot_id"), event.get("subtype"))
            return

        thread_ts = event.get("thread_ts")
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "").strip()

        logger.info(
            "Message received: thread_ts=%s, channel=%s, user=%s, text='%s'",
            thread_ts, channel_id, user_id, text,
        )

        if not text or not channel_id:
            return

        # Look up session — if no thread_ts (user replied in channel, not thread),
        # fall back to the user's most recent active session
        session = None
        if thread_ts:
            session = get_session(thread_ts)

        if session is None and user_id:
            # Fallback: find the most recent non-complete session for this user
            user_sessions = get_sessions_by_user(user_id)
            active = [
                s for s in user_sessions
                if s.status not in ("complete",)
            ]
            if active:
                session = active[0]
                thread_ts = session.thread_ts
                logger.info(
                    "No thread session found — using user's latest active session: thread_ts=%s, status=%s",
                    thread_ts, session.status,
                )

        if session is None:
            logger.info("No session found for thread_ts=%s / user=%s", thread_ts, user_id)
            return

        status = session.status
        service = ProposalService(client, channel_id, thread_ts, user_id)

        try:
            if status == "awaiting_confirmation":
                # User confirmed intake data (or corrected something)
                asyncio.create_task(
                    service.resume_pipeline(user_input=text, resume_point="intake")
                )

            elif status == "awaiting_search_decision":
                # User answered yes/no to web search
                asyncio.create_task(
                    service.resume_pipeline(user_input=text, resume_point="search")
                )

            elif status == "awaiting_tone":
                # User picked a tone
                asyncio.create_task(
                    service.resume_pipeline(user_input=text, resume_point="tone")
                )

            elif status == "complete":
                # Post-generation: question or change request
                asyncio.create_task(
                    service.handle_revision(revision_request=text)
                )

            elif status == "revising":
                # Still in revision loop
                asyncio.create_task(
                    service.handle_revision(revision_request=text)
                )

        except Exception as e:
            logger.error("Message handler error: %s", e, exc_info=True)
            await say(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"❌ Something went wrong: {str(e)}",
            )
