"""
Agent 4: REVISION

Handles two types of post-generation requests:
  1. Follow-up questions: "What did the client say about their deadline?"
     → Searches transcript collection in Qdrant (second RAG source)
  2. Change requests: "Make the timeline section more detailed"
     → Rewrites ONLY that section, merges back, regenerates DOCX
     → Updates Qdrant entry if proposal was saved (no duplicate)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.rag.ingestion import update_saved_proposal
from app.rag.vector_db import TRANSCRIPTS_COLLECTION, QdrantManager
from app.tools.docx_generator import generate_docx
from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about a client discovery call.
You have access to excerpts from the transcript.

Answer the user's question using ONLY the information provided in the transcript excerpts.
If the answer is not in the excerpts, say so clearly — do not guess.
Be concise and direct. Quote the transcript where helpful.
"""

EDIT_SYSTEM_PROMPT = """\
You are an expert proposal editor. You will be given:
1. A specific section of a proposal
2. A change request from the user

Your job is to rewrite ONLY that section based on the request.
Do NOT change other sections. Do NOT add new sections unless asked.
Return ONLY the rewritten section content in Markdown format.
Preserve the section header (e.g. ## Implementation Timeline).
"""

SECTION_CLASSIFIER_PROMPT = """\
Classify the following user request.

Return a JSON object with:
{
    "type": "question" or "edit",
    "section": "name of section to edit (if edit type, else null)",
    "summary": "one-line summary of what the user wants"
}

Known proposal sections:
- Executive Summary
- Proposed Solution
- Implementation Timeline
- Budget Range
- Next Steps
- Cover Section

Return ONLY valid JSON.
"""


async def revision_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Handle follow-up questions and section-level change requests.

    Input state keys:
        - revision_request: str (user's message)
        - draft_proposal: str (current proposal markdown)
        - client_info: dict
        - thread_ts: str
        - qdrant_proposal_id: str | None

    Output state keys:
        - draft_proposal: str (updated if edit was made)
        - docx_bytes: bytes (updated DOCX if edit was made)
        - revised_section: str | None
        - status: "revising"
    """
    settings = get_settings()
    request = state.get("revision_request", "").strip()
    current_draft = state.get("draft_proposal", "")
    client_info = state.get("client_info", {})
    thread_ts = state.get("thread_ts", "")
    qdrant_proposal_id = state.get("qdrant_proposal_id")

    if not request:
        return {**state, "status": "revising"}

    logger.info("Revision agent handling: %s", request[:100])

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0,
        max_tokens=1000,
    )

    # Step 1: Classify the request
    classification = await _classify_request(llm, request)
    request_type = classification.get("type", "question")
    section_name = classification.get("section")

    logger.info(
        "Request classified as: type=%s, section=%s",
        request_type,
        section_name,
    )

    # ------------------------------------------------------------------
    # Branch 1: Follow-up Question → Search transcript in Qdrant
    # ------------------------------------------------------------------
    if request_type == "question":
        answer = await _answer_from_transcript(llm, request, thread_ts)
        return {
            **state,
            "revision_answer": answer,
            "revised_section": None,
            "status": "revising",
        }

    # ------------------------------------------------------------------
    # Branch 2: Section Edit → Rewrite only that section
    # ------------------------------------------------------------------
    if request_type == "edit" and section_name:
        updated_draft, updated_section = await _edit_section(
            llm=llm,
            draft=current_draft,
            section_name=section_name,
            change_request=request,
        )

        if updated_draft == current_draft:
            logger.warning(
                "Section '%s' not found in draft — returning unchanged", section_name
            )
            return {
                **state,
                "revision_answer": (
                    f"I couldn't find the '{section_name}' section in the proposal. "
                    f"Available sections: {_list_sections(current_draft)}"
                ),
                "revised_section": None,
                "status": "revising",
            }

        # Regenerate DOCX with updated draft
        docx_bytes = generate_docx(updated_draft, client_info)

        # Update Qdrant if this proposal was saved
        if qdrant_proposal_id:
            qdrant = QdrantManager()
            update_saved_proposal(
                qdrant=qdrant,
                thread_ts=thread_ts,
                updated_text=updated_draft,
                client_info=client_info,
            )
            logger.info(
                "Updated Qdrant entry for thread_ts=%s after revision", thread_ts
            )

        return {
            **state,
            "draft_proposal": updated_draft,
            "docx_bytes": docx_bytes,
            "revised_section": section_name,
            "revision_answer": f"✅ I've updated the **{section_name}** section.",
            "status": "revising",
        }

    # Fallback
    return {
        **state,
        "revision_answer": "I'm not sure how to handle that request. Please try rephrasing.",
        "revised_section": None,
        "status": "revising",
    }


# ---------------------------------------------------------------------------
# Helper: Classify request type
# ---------------------------------------------------------------------------

async def _classify_request(llm: ChatGroq, request: str) -> dict:
    """Use LLM to classify the user request as 'question' or 'edit'."""
    import json

    messages = [
        SystemMessage(content=SECTION_CLASSIFIER_PROMPT),
        HumanMessage(content=f"User request: {request}"),
    ]
    try:
        response = await llm.ainvoke(messages)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        return json.loads(text)
    except Exception as e:
        logger.warning("Request classification failed: %s", e)
        # Default: treat as question if classification fails
        return {"type": "question", "section": None, "summary": request}


# ---------------------------------------------------------------------------
# Helper: Answer question from transcript via Qdrant
# ---------------------------------------------------------------------------

async def _answer_from_transcript(
    llm: ChatGroq,
    question: str,
    thread_ts: str,
) -> str:
    """Search the transcript collection and answer the question."""
    qdrant = QdrantManager()

    # Filter by thread_ts to get this session's transcript
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    results = qdrant.search_similar(
        collection=TRANSCRIPTS_COLLECTION,
        query=question,
        top_k=5,
        filters=Filter(
            must=[
                FieldCondition(
                    key="thread_ts",
                    match=MatchValue(value=thread_ts),
                )
            ]
        ),
    )

    if not results:
        return (
            "I couldn't find relevant information in the transcript. "
            "The transcript may not have been indexed yet."
        )

    # Build context from transcript excerpts
    excerpts = "\n\n---\n\n".join(
        f"[Section: {r['section_title']}]\n{r['text']}" for r in results
    )

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Question: {question}\n\nTranscript excerpts:\n{excerpts}"
        ),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Helper: Edit a specific section
# ---------------------------------------------------------------------------

async def _edit_section(
    llm: ChatGroq,
    draft: str,
    section_name: str,
    change_request: str,
) -> tuple[str, str]:
    """
    Find a section in the draft, rewrite it, and merge it back.

    Returns (updated_draft, rewritten_section_text).
    """
    # Find the section using regex
    # Matches ## Section Name until the next ## or end of string
    pattern = re.compile(
        r"(#{1,3}\s*" + re.escape(section_name) + r".*?(?=\n#{1,3}\s|\Z))",
        re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(draft)
    if not match:
        return draft, ""

    original_section = match.group(1).strip()

    messages = [
        SystemMessage(content=EDIT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Change request: {change_request}\n\n"
                f"Current section:\n{original_section}"
            )
        ),
    ]

    response = await llm.ainvoke(messages)
    rewritten = response.content.strip()

    # Replace the old section with the rewritten one
    updated_draft = draft.replace(original_section, rewritten)

    return updated_draft, rewritten


# ---------------------------------------------------------------------------
# Helper: List sections in draft
# ---------------------------------------------------------------------------

def _list_sections(draft: str) -> str:
    """Extract and list all section headers from a markdown draft."""
    headers = re.findall(r"^#{1,3}\s+(.+)$", draft, re.MULTILINE)
    return ", ".join(headers) if headers else "no sections found"
