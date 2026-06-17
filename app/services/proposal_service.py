"""
Proposal Service — orchestrates the LangGraph pipeline.

Manages:
  - Starting the generation pipeline
  - Resuming after human-in-loop interrupts
  - Routing to the revision agent
  - Posting status updates and DOCX files to Slack
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.agents.graph import get_generation_graph, get_revision_graph
from app.rag.ingestion import ingest_transcript
from app.rag.vector_db import QdrantManager
from app.state.session_store import (
    create_session,
    get_session,
    session_exists,
    update_session,
)

logger = logging.getLogger(__name__)


class ProposalService:
    """
    Orchestrates the full proposal generation and revision pipeline.

    One instance per Slack thread interaction.
    """

    def __init__(
        self,
        slack_client,
        channel_id: str,
        thread_ts: str,
        user_id: str,
    ) -> None:
        self.client = slack_client
        self.channel_id = channel_id
        self.thread_ts = thread_ts
        self.user_id = user_id
        self.graph_config = {"configurable": {"thread_id": thread_ts}}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, text: str) -> None:
        """Post a message in the Slack thread."""
        await self.client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            text=text,
        )

    async def _upload_docx(
        self,
        docx_bytes: bytes,
        client_info: dict,
    ) -> None:
        """Upload the DOCX proposal to Slack."""
        company = client_info.get("company_name") or "Client"
        today = date.today().strftime("%Y-%m-%d")
        filename = f"Proposal_{company}_{today}.docx".replace(" ", "_")

        await self.client.files_upload_v2(
            channel=self.channel_id,
            thread_ts=self.thread_ts,
            content=docx_bytes,
            filename=filename,
            title=f"Proposal for {company}",
            initial_comment="📄 Here's your proposal! Let me know if you'd like any changes.",
        )

    def _build_initial_state(self, transcript: str) -> dict[str, Any]:
        """Build the initial ProposalState for a new pipeline run."""
        return {
            "thread_ts": self.thread_ts,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "raw_transcript": transcript,
            "client_info": {},
            "missing_fields": [],
            "user_confirmed": False,
            "similar_proposals": [],
            "web_results": [],
            "rag_status": "",
            "search_internet": None,
            "tone": None,
            "draft_proposal": None,
            "docx_bytes": None,
            "save_status": None,
            "qdrant_proposal_id": None,
            "revision_request": None,
            "revised_section": None,
            "status": "intake",
            "error_message": None,
        }

    # ------------------------------------------------------------------
    # 1. Start the generation pipeline
    # ------------------------------------------------------------------

    async def run_generation_pipeline(self, transcript_text: str) -> None:
        """
        Start the full proposal generation pipeline.

        Runs until the first human-in-loop interrupt, then waits.
        The message handler will call resume_pipeline() to continue.
        """
        try:
            # Create or update session
            if not session_exists(self.thread_ts):
                create_session(
                    thread_ts=self.thread_ts,
                    channel_id=self.channel_id,
                    user_id=self.user_id,
                    state=self._build_initial_state(transcript_text),
                    transcript_text=transcript_text,
                )
            else:
                # Existing session — add new transcript
                session = get_session(self.thread_ts)
                if session:
                    old_state = session.get_state()
                    old_state["raw_transcript"] = transcript_text
                    update_session(
                        self.thread_ts,
                        state=old_state,
                        status="intake",
                    )

            # Ingest transcript into Qdrant for follow-up Q&A
            qdrant = QdrantManager()
            ingest_transcript(qdrant, transcript_text, self.thread_ts)

            # Run the graph — will stop at first interrupt (intake confirmation)
            graph = get_generation_graph()
            initial_state = self._build_initial_state(transcript_text)

            async for event in graph.astream(
                initial_state,
                config=self.graph_config,
            ):
                await self._handle_graph_event(event)

        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
            await self._post(f"❌ Pipeline error: {str(e)}")
            update_session(self.thread_ts, status="error")

    # ------------------------------------------------------------------
    # 2. Resume after human-in-loop
    # ------------------------------------------------------------------

    async def resume_pipeline(
        self,
        user_input: str,
        resume_point: str,
    ) -> None:
        """
        Resume the LangGraph pipeline after a human-in-loop interrupt.

        Args:
            user_input: The user's reply in Slack.
            resume_point: Which interrupt we're resuming ("intake", "search", "tone").
        """
        try:
            graph = get_generation_graph()

            # Update session status
            status_map = {
                "intake": "researching",
                "search": "researching",
                "tone": "writing",
            }
            update_session(
                self.thread_ts,
                status=status_map.get(resume_point, "researching"),
            )

            await self._post(
                {
                    "intake": "✅ Got it! Searching for similar past proposals...",
                    "search": "🔍 Understood! Proceeding...",
                    "tone": "✍️ Great! Generating your proposal now...",
                }.get(resume_point, "✅ Continuing...")
            )

            # Resume the graph with the user's input
            async for event in graph.astream(
                {"__interrupt__": user_input},  # Resume with user input
                config=self.graph_config,
            ):
                await self._handle_graph_event(event)

        except Exception as e:
            logger.error("Resume error: %s", e, exc_info=True)
            await self._post(f"❌ Error resuming pipeline: {str(e)}")

    # ------------------------------------------------------------------
    # 3. Handle revision requests
    # ------------------------------------------------------------------

    async def handle_revision(self, revision_request: str) -> None:
        """
        Route a user message to the revision agent.

        Handles both follow-up questions and section edit requests.
        """
        try:
            session = get_session(self.thread_ts)
            if not session:
                await self._post("⚠️ No active session found for this thread.")
                return

            state = session.get_state()
            state["revision_request"] = revision_request
            state["thread_ts"] = self.thread_ts
            state["channel_id"] = self.channel_id
            state["user_id"] = self.user_id

            update_session(self.thread_ts, status="revising")

            # Run revision graph
            graph = get_revision_graph()
            final_state = None

            async for event in graph.astream(
                state,
                config={"configurable": {"thread_id": f"{self.thread_ts}_revision"}},
            ):
                if isinstance(event, dict):
                    for node_output in event.values():
                        if isinstance(node_output, dict):
                            final_state = node_output

            if not final_state:
                return

            # Post the answer or updated DOCX
            answer = final_state.get("revision_answer", "")
            if answer:
                await self._post(answer)

            # If a section was edited, upload new DOCX
            if final_state.get("revised_section") and final_state.get("docx_bytes"):
                client_info = final_state.get("client_info", {})
                await self._upload_docx(final_state["docx_bytes"], client_info)

            # Update session with new draft
            if final_state.get("draft_proposal"):
                update_session(
                    self.thread_ts,
                    state=final_state,
                    status="complete",
                    draft=final_state.get("draft_proposal"),
                )

        except Exception as e:
            logger.error("Revision error: %s", e, exc_info=True)
            await self._post(f"❌ Error handling revision: {str(e)}")

    # ------------------------------------------------------------------
    # Graph event handler
    # ------------------------------------------------------------------

    async def _handle_graph_event(self, event: Any) -> None:
        """
        Process events emitted by the LangGraph graph.

        Handles:
          - Interrupt events → post question to Slack, update session status
          - Node completion → post progress updates, upload DOCX on completion
        """
        if not isinstance(event, dict):
            return

        # Check for interrupt (human-in-loop pause)
        for node_name, node_output in event.items():
            if node_name == "__interrupt__":
                await self._handle_interrupt(node_output)
                return

            if not isinstance(node_output, dict):
                continue

            # Node completed — check for key outputs
            status = node_output.get("status", "")

            if node_name == "intake_node" and node_output.get("client_info"):
                # Intake done — interrupt message was already sent
                pass

            elif node_name == "writer_node" and node_output.get("docx_bytes"):
                # Writer done — upload DOCX
                client_info = node_output.get("client_info", {})
                docx_bytes = node_output.get("docx_bytes")
                draft = node_output.get("draft_proposal", "")
                save_status = node_output.get("save_status", "")

                await self._upload_docx(docx_bytes, client_info)

                # Notify about save status
                if save_status == "saved":
                    await self._post(
                        "💾 _This proposal has been saved to my knowledge base "
                        "for future reference._"
                    )
                elif save_status == "skipped_duplicate":
                    await self._post(
                        "ℹ️ _A similar proposal already exists in my knowledge base. "
                        "Skipping save to avoid duplicates._"
                    )

                # Update session
                update_session(
                    self.thread_ts,
                    state=node_output,
                    status="complete",
                    draft=draft,
                )

                await self._post(
                    "✅ *Proposal complete!* Reply in this thread to:\n"
                    "• Ask questions: _'What did the client say about their timeline?'_\n"
                    "• Request changes: _'Make the budget section more detailed'_"
                )

            elif status == "error":
                error_msg = node_output.get("error_message", "Unknown error")
                await self._post(f"❌ Error in {node_name}: {error_msg}")
                update_session(self.thread_ts, status="error")

    async def _handle_interrupt(self, interrupt_data: Any) -> None:
        """Post the interrupt message to Slack and update session status."""
        if isinstance(interrupt_data, (list, tuple)):
            # LangGraph wraps interrupt value in a list
            for item in interrupt_data:
                if hasattr(item, "value"):
                    interrupt_data = item.value
                    break

        if not isinstance(interrupt_data, dict):
            return

        action = interrupt_data.get("action", "")
        message = interrupt_data.get("message", "")

        # Map action to session status
        status_map = {
            "confirm_intake": "awaiting_confirmation",
            "web_search_approval": "awaiting_search_decision",
            "tone_selection": "awaiting_tone",
        }
        new_status = status_map.get(action, "awaiting_confirmation")

        if message:
            await self._post(message)

        update_session(self.thread_ts, status=new_status)
        logger.info("Pipeline paused at: %s (status: %s)", action, new_status)
