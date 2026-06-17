"""
Agent 2: RESEARCH

Searches past proposals in Qdrant (RAG) for relevant context.
3-layer fallback: Qdrant → Web Search (with user approval) → Zero-shot.
"""

from __future__ import annotations

import logging
from typing import Any

from app.rag.vector_db import PROPOSALS_COLLECTION, QdrantManager
from app.tools.web_search import search_industry_context
from config import get_settings

logger = logging.getLogger(__name__)


async def research_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Search for relevant past proposals and industry context.

    3-layer fallback:
      Layer 1: Qdrant RAG (past proposals) — filter by industry first
      Layer 2: Web search (Tavily) — only if user approved
      Layer 3: Zero-shot — Writer uses LLM general knowledge

    Input state keys:
        - client_info: dict (ClientInfo fields)
        - search_internet: bool | None (user's decision from human-in-loop)

    Output state keys:
        - similar_proposals: list[dict]
        - web_results: list[dict]
        - rag_status: "matched" | "web_search" | "zero_shot"
        - status: "researching"
    """
    settings = get_settings()
    client_info = state.get("client_info", {})
    search_internet = state.get("search_internet")

    industry = client_info.get("industry", "")
    problem = client_info.get("problem_statement", "")
    goals = client_info.get("goals", [])

    # Build a rich search query
    query_parts = []
    if industry:
        query_parts.append(industry)
    if problem:
        query_parts.append(problem)
    if goals:
        query_parts.append(" ".join(goals[:3]))  # First 3 goals
    query = " ".join(query_parts) if query_parts else "business proposal"

    logger.info("Research agent searching with query: %s", query[:100])

    # Initialize Qdrant
    qdrant = QdrantManager()

    # ------------------------------------------------------------------
    # Layer 1: Qdrant RAG search (with industry filter)
    # ------------------------------------------------------------------
    results = qdrant.search_with_metadata_filter(
        collection=PROPOSALS_COLLECTION,
        query=query,
        industry=industry.lower() if industry else None,
        top_k=3,
    )

    # Filter by similarity threshold
    threshold = settings.SIMILARITY_THRESHOLD
    relevant = [r for r in results if r["score"] >= threshold]

    if relevant:
        logger.info(
            "RAG matched: %d results above threshold %.2f (top score: %.3f)",
            len(relevant),
            threshold,
            relevant[0]["score"],
        )
        return {
            **state,
            "similar_proposals": relevant,
            "web_results": [],
            "rag_status": "matched",
            "status": "researching",
        }

    logger.info(
        "No RAG match above threshold %.2f (best: %.3f)",
        threshold,
        results[0]["score"] if results else 0.0,
    )

    # ------------------------------------------------------------------
    # Layer 2: Web search (if user approved)
    # ------------------------------------------------------------------
    if search_internet is True:
        logger.info("User approved web search. Searching for %s...", industry)
        web_results = await search_industry_context(
            industry=industry or "general",
            problem=problem or "business challenge",
        )

        if web_results:
            logger.info("Web search returned %d results", len(web_results))
            return {
                **state,
                "similar_proposals": [],
                "web_results": web_results,
                "rag_status": "web_search",
                "status": "researching",
            }

    # ------------------------------------------------------------------
    # Layer 3: Zero-shot (no RAG, no web)
    # ------------------------------------------------------------------
    if search_internet is False:
        logger.info("User declined web search. Proceeding zero-shot.")
    else:
        logger.info("No web search decision. Proceeding zero-shot.")

    return {
        **state,
        "similar_proposals": [],
        "web_results": [],
        "rag_status": "zero_shot",
        "status": "researching",
    }


def needs_web_search_approval(state: dict[str, Any]) -> bool:
    """
    Check if we need to ask the user about web search.

    Returns True if RAG found no match AND user hasn't decided yet.
    Used by the LangGraph conditional edge.
    """
    rag_status = state.get("rag_status", "")
    search_internet = state.get("search_internet")
    return rag_status != "matched" and search_internet is None
