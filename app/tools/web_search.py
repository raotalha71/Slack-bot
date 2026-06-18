"""
Web search fallback using Tavily API.

Called when the Research agent finds no relevant past proposals in Qdrant
and the user approves searching the internet (human-in-loop #2).
"""

from __future__ import annotations

import logging
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


async def search_industry_context(
    industry: str,
    problem: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Search the internet for industry context and proposal examples.

    Uses Tavily API (built for AI agents — returns clean extracted text).

    Args:
        industry: The client's industry.
        problem: The client's problem statement.
        max_results: Max results per query.

    Returns:
        List of dicts with 'title', 'url', 'content' for each result.
    """
    settings = get_settings()

    if not settings.TAVILY_API_KEY:
        logger.warning("Tavily API key not set — skipping web search")
        return []

    try:
        import asyncio

        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        # Search for proposal examples and industry context
        queries = [
            f"{industry} business proposal example best practices",
            f"{industry} {problem} market trends challenges",
        ]

        all_results: list[dict[str, Any]] = []

        for query in queries:
            logger.info("Web search query: %s", query)
            # TavilyClient.search() is synchronous — run in thread pool
            # to avoid blocking the async event loop
            response = await asyncio.to_thread(
                client.search,
                query=query,
                max_results=max_results,
                search_depth="basic",
            )

            for result in response.get("results", []):
                all_results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                    }
                )

        logger.info(
            "Web search returned %d results for industry=%s",
            len(all_results),
            industry,
        )
        return all_results

    except ImportError:
        logger.error("tavily-python not installed. Run: pip install tavily-python")
        return []
    except Exception as e:
        logger.error("Web search failed: %s", str(e), exc_info=True)
        return []

