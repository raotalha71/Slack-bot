"""
LangGraph pipeline wiring for the Slack Proposal Generator.

Two graph modes:
  Mode A — Initial generation pipeline (intake → research → writer)
  Mode B — Revision handler (question/edit → revision agent)

Human-in-loop points (LangGraph interrupts):
  1. After intake: user confirms extracted info
  2. After research (no RAG match): user approves web search
  3. Before writer: user picks tone
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.intake import intake_agent
from app.agents.research import needs_web_search_approval, research_agent
from app.agents.revision import revision_agent
from app.agents.writer import writer_agent
from app.models.schemas import ProposalState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node wrappers (add human-in-loop interrupts)
# ---------------------------------------------------------------------------


async def intake_node(state: ProposalState) -> dict[str, Any]:
    """Run intake agent, then pause for user confirmation."""
    result = await intake_agent(state)

    # Human-in-loop #1: Show extracted info, ask user to confirm
    client_info = result.get("client_info", {})
    missing = result.get("missing_fields", [])

    confirmation_msg = _build_intake_confirmation(client_info, missing)

    # Interrupt — pauses graph until user confirms in Slack
    user_response = interrupt(
        {
            "action": "confirm_intake",
            "message": confirmation_msg,
            "client_info": client_info,
            "missing_fields": missing,
        }
    )

    return {**result, "user_confirmed": True, "user_response": user_response}


async def research_node(state: ProposalState) -> dict[str, Any]:
    """Run research agent with optional web search interrupt."""
    # First pass: run research without web search decision
    result = await research_agent(state)

    # Human-in-loop #2: If no RAG match, ask about web search
    if needs_web_search_approval(result):
        user_response = interrupt(
            {
                "action": "web_search_approval",
                "message": (
                    "🔍 I didn't find similar past proposals for this industry in my knowledge base.\n"
                    "Would you like me to search the internet for context?\n"
                    "Reply *yes* to search or *no* to proceed with general knowledge."
                ),
            }
        )
        # Parse user response
        approved = _parse_yes_no(str(user_response))
        result["search_internet"] = approved

        # Re-run research with the decision
        result = await research_agent(result)

    return result


async def tone_node(state: ProposalState) -> dict[str, Any]:
    """Human-in-loop #3: Ask user to pick a tone."""
    user_response = interrupt(
        {
            "action": "tone_selection",
            "message": (
                "🎨 What tone should I use for this proposal?\n\n"
                "Reply with a number:\n"
                "1️⃣ *Professional & Formal* — corporate, authoritative\n"
                "2️⃣ *Technical & Data-Driven* — metrics, specifications\n"
                "3️⃣ *Consultative & Friendly* — warm, advisory, partner-focused\n"
                "4️⃣ *Industry Default* — auto-adapt to the client's industry"
            ),
        }
    )

    tone_map = {
        "1": "formal",
        "2": "technical",
        "3": "consultative",
        "4": "industry_default",
        "formal": "formal",
        "technical": "technical",
        "consultative": "consultative",
        "default": "industry_default",
        "industry": "industry_default",
    }

    raw = str(user_response).strip().lower()
    tone = tone_map.get(raw, "industry_default")

    logger.info("Tone selected: %s (raw input: %s)", tone, raw)
    return {**state, "tone": tone}


async def writer_node(state: ProposalState) -> dict[str, Any]:
    """Run writer agent to generate proposal + DOCX + Smart Save Gate."""
    return await writer_agent(state)


async def revision_node(state: ProposalState) -> dict[str, Any]:
    """Run revision agent for questions and section edits."""
    return await revision_agent(state)


# ---------------------------------------------------------------------------
# Conditional Routing Functions
# ---------------------------------------------------------------------------


def route_after_research(
    state: ProposalState,
) -> Literal["tone_node", "research_node"]:
    """After research: always go to tone selection."""
    return "tone_node"


# ---------------------------------------------------------------------------
# Graph A: Initial Generation Pipeline
# ---------------------------------------------------------------------------


def build_generation_graph(checkpointer) -> Any:
    """
    Build and compile the initial proposal generation graph.

    Flow: intake → research → tone → writer → END
    Human-in-loop at: intake (confirm), research (web search), tone (pick)
    """
    graph = StateGraph(ProposalState)

    # Add nodes
    graph.add_node("intake_node", intake_node)
    graph.add_node("research_node", research_node)
    graph.add_node("tone_node", tone_node)
    graph.add_node("writer_node", writer_node)

    # Add edges
    graph.add_edge(START, "intake_node")
    graph.add_edge("intake_node", "research_node")
    graph.add_edge("research_node", "tone_node")
    graph.add_edge("tone_node", "writer_node")
    graph.add_edge("writer_node", END)

    # Compile with persistent SQLite checkpointer (survives server reloads)
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Generation graph compiled with persistent checkpointer")
    return compiled


# ---------------------------------------------------------------------------
# Graph B: Revision Pipeline
# ---------------------------------------------------------------------------


def build_revision_graph(checkpointer) -> Any:
    """
    Build and compile the revision graph.

    Simple linear flow: revision_node → END
    No interrupts — triggered directly by Slack message handler.
    """
    graph = StateGraph(ProposalState)
    graph.add_node("revision_node", revision_node)
    graph.add_edge(START, "revision_node")
    graph.add_edge("revision_node", END)

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Revision graph compiled with persistent checkpointer")
    return compiled


# ---------------------------------------------------------------------------
# Persistent checkpointer + Singletons
# ---------------------------------------------------------------------------

import os

_CHECKPOINT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "langgraph_checkpoints.db",
)

# Shared persistent checkpointer — survives uvicorn --reload
_checkpointer = None
_generation_graph = None
_revision_graph = None


async def _get_checkpointer():
    """Get or create the persistent SQLite checkpointer."""
    global _checkpointer
    if _checkpointer is None:
        os.makedirs(os.path.dirname(_CHECKPOINT_DB_PATH), exist_ok=True)
        conn = await aiosqlite.connect(_CHECKPOINT_DB_PATH)
        _checkpointer = AsyncSqliteSaver(conn)
        logger.info("SQLite async checkpointer created at %s", _CHECKPOINT_DB_PATH)
    return _checkpointer


async def get_generation_graph() -> Any:
    """Get or build the generation graph (singleton)."""
    global _generation_graph
    if _generation_graph is None:
        checkpointer = await _get_checkpointer()
        _generation_graph = build_generation_graph(checkpointer)
    return _generation_graph


async def get_revision_graph() -> Any:
    """Get or build the revision graph (singleton)."""
    global _revision_graph
    if _revision_graph is None:
        checkpointer = await _get_checkpointer()
        _revision_graph = build_revision_graph(checkpointer)
    return _revision_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_intake_confirmation(
    client_info: dict[str, Any],
    missing: list[str],
) -> str:
    """Build a human-readable confirmation message for the user."""
    lines = ["📋 *Here's what I extracted from the transcript:*\n"]

    field_labels = {
        "company_name": "🏢 Company",
        "industry": "🏭 Industry",
        "problem_statement": "🔍 Problem",
        "goals": "🎯 Goals",
        "budget": "💰 Budget",
        "timeline": "📅 Timeline",
        "stakeholders": "👥 Stakeholders",
    }

    for key, label in field_labels.items():
        value = client_info.get(key)
        if isinstance(value, list):
            display = ", ".join(str(v) for v in value) if value else "_Not found_"
        else:
            display = str(value) if value else "_Not found_"
        lines.append(f"• {label}: {display}")

    if missing:
        lines.append(
            f"\n⚠️ *Missing fields:* {', '.join(missing)}\n"
            "_These will be noted in the proposal as 'to be discussed'._"
        )

    lines.append("\n✅ Reply *ok* to proceed, or tell me what to correct.")
    return "\n".join(lines)


def _parse_yes_no(text: str) -> bool:
    """Parse a yes/no user response."""
    text = text.lower().strip()
    return text in ("yes", "y", "yeah", "yep", "sure", "ok", "okay", "1")
