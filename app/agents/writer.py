"""
Agent 3: WRITER

Synthesizes client info + research results into a structured proposal.
Adapts tone based on user selection. Generates DOCX.
Runs Smart Save Gate to save novel proposals back to Qdrant.

Key rule: Does NOT hardcode proposal structure — the agent reasons
about structure based on client info (assessment requirement).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.rag.ingestion import smart_save_proposal
from app.rag.vector_db import QdrantManager
from app.tools.docx_generator import generate_docx
from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tone Definitions
# ---------------------------------------------------------------------------

TONE_PROMPTS = {
    "formal": (
        "Use professional, corporate language. Maintain a formal and authoritative "
        "tone throughout. Use precise terminology and structured argumentation."
    ),
    "technical": (
        "Use data-driven, precise technical language. Include metrics, "
        "benchmarks, and technical specifications where relevant. "
        "Emphasize methodology and measurable outcomes."
    ),
    "consultative": (
        "Use warm, advisory language. Position yourself as a trusted partner. "
        "Focus on the client's pain points and show empathy. "
        "Use collaborative phrasing like 'together we can' and 'our approach'."
    ),
    "industry_default": (
        "Adapt your tone to be appropriate for the client's industry. "
        "Healthcare → formal and compliant. Fintech → data-driven and innovative. "
        "Retail → consumer-focused and dynamic. Construction → practical and safety-conscious."
    ),
}

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """\
You are an expert proposal writer. Your job is to create a professional business 
proposal based on client information and reference material.

IMPORTANT RULES:
1. DO NOT use a hardcoded template. Reason about the best structure based on the 
   client's industry, problem, and goals.
2. Every proposal MUST include these sections (but adapt their content to the client):
   - Cover Section (client company name, date, "Prepared by [Your Company]")
   - Executive Summary
   - Proposed Solution
   - Implementation Timeline
   - Budget Range
   - Next Steps
3. You may add additional sections if they make sense for this specific client 
   (e.g., "Compliance Considerations" for healthcare, "ROI Projections" for fintech).
4. Use the reference proposals for inspiration on structure and language, but 
   customize everything for THIS client.
5. If budget or timeline info is missing, acknowledge it gracefully: 
   "To be discussed during the next consultation."
6. Output in clean Markdown format.

{tone_instruction}
"""


async def writer_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a proposal document from client info + research results.

    Input state keys:
        - client_info: dict
        - similar_proposals: list[dict] (from RAG)
        - web_results: list[dict] (from web search)
        - rag_status: str
        - tone: str

    Output state keys:
        - draft_proposal: str (Markdown)
        - docx_bytes: bytes
        - save_status: str
        - qdrant_proposal_id: str | None
        - status: "writing"
    """
    settings = get_settings()
    client_info = state.get("client_info", {})
    similar_proposals = state.get("similar_proposals", [])
    web_results = state.get("web_results", [])
    rag_status = state.get("rag_status", "zero_shot")
    tone = state.get("tone", "industry_default")
    thread_ts = state.get("thread_ts", "")

    # Build tone instruction
    tone_instruction = TONE_PROMPTS.get(tone, TONE_PROMPTS["industry_default"])

    # Build the system prompt with tone
    system_prompt = WRITER_SYSTEM_PROMPT.format(tone_instruction=tone_instruction)

    # Build context from research results
    context_parts = []

    if rag_status == "matched" and similar_proposals:
        context_parts.append("## Reference Proposals (from past work)")
        for i, prop in enumerate(similar_proposals[:3], 1):
            context_parts.append(
                f"\n### Reference {i} (Industry: {prop.get('industry', 'N/A')}, "
                f"Source: {prop.get('source_file', 'N/A')})\n"
                f"{prop.get('text', '')}"
            )
    elif rag_status == "web_search" and web_results:
        context_parts.append("## Industry Research (from web search)")
        for i, result in enumerate(web_results[:5], 1):
            context_parts.append(
                f"\n### Source {i}: {result.get('title', 'N/A')}\n"
                f"{result.get('content', '')}"
            )
    else:
        context_parts.append(
            "## Note\nNo past proposals or web research available. "
            "Generate this proposal using your general knowledge about "
            f"the {client_info.get('industry', 'business')} industry."
        )

    context = "\n".join(context_parts)

    # Build client info summary
    client_summary = _format_client_info(client_info)

    # Construct the user message
    user_message = (
        f"Write a professional proposal for this client.\n\n"
        f"## Client Information\n{client_summary}\n\n"
        f"{context}"
    )

    logger.info(
        "Writer agent generating proposal (tone=%s, rag_status=%s, context_len=%d)",
        tone,
        rag_status,
        len(context),
    )

    # Call Groq LLM
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.7,  # Slightly creative for writing
        max_tokens=4000,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    try:
        response = await llm.ainvoke(messages)
        draft = response.content.strip()

        logger.info("Writer agent generated proposal (%d chars)", len(draft))

        # Generate DOCX
        docx_bytes = generate_docx(draft, client_info)
        logger.info("DOCX generated (%d bytes)", len(docx_bytes))

        # Smart Save Gate: check for duplicate and save if novel
        qdrant = QdrantManager()
        save_result = smart_save_proposal(
            qdrant=qdrant,
            proposal_text=draft,
            client_info=client_info,
            thread_ts=thread_ts,
        )

        save_status = "saved" if save_result["saved"] else "skipped_duplicate"
        logger.info("Smart Save Gate: %s — %s", save_status, save_result["reason"])

        return {
            **state,
            "draft_proposal": draft,
            "docx_bytes": docx_bytes,
            "save_status": save_status,
            "qdrant_proposal_id": thread_ts if save_result["saved"] else None,
            "status": "complete",
        }

    except Exception as e:
        logger.error("Writer agent failed: %s", e)
        return {
            **state,
            "draft_proposal": None,
            "docx_bytes": None,
            "save_status": "not_saved",
            "status": "error",
            "error_message": f"Writer error: {e}",
        }


def _format_client_info(info: dict) -> str:
    """Format client info dict into readable text for the LLM prompt."""
    lines = []
    field_labels = {
        "company_name": "Company",
        "industry": "Industry",
        "problem_statement": "Problem",
        "goals": "Goals",
        "budget": "Budget",
        "timeline": "Timeline",
        "stakeholders": "Stakeholders",
        "additional_context": "Additional Context",
    }

    for key, label in field_labels.items():
        value = info.get(key)
        if value is None:
            lines.append(f"- **{label}**: _Not provided_")
        elif isinstance(value, list):
            if value:
                items = ", ".join(str(v) for v in value)
                lines.append(f"- **{label}**: {items}")
            else:
                lines.append(f"- **{label}**: _Not provided_")
        else:
            lines.append(f"- **{label}**: {value}")

    return "\n".join(lines)
