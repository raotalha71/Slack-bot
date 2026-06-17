"""
Agent 1: INTAKE

Reads the raw sales transcript and extracts structured client information.
Flags missing or vague fields — does NOT guess (assessment requirement).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.models.schemas import ClientInfo
from config import get_settings

logger = logging.getLogger(__name__)

# System prompt for structured extraction
INTAKE_SYSTEM_PROMPT = """\
You are an expert business analyst. Your job is to extract structured client 
information from a sales discovery call transcript.

RULES:
1. Extract ONLY information that is explicitly stated in the transcript.
2. If a field is not mentioned or is vague, set it to null. Do NOT guess or infer.
3. List all fields that are null or unclear in the "missing_fields" array.
4. Be precise with company names, industry terms, and stakeholder names.

Return a JSON object with this exact structure:
{
    "client_info": {
        "company_name": "string or null",
        "industry": "string or null",
        "problem_statement": "string or null",
        "goals": ["list of strings"],
        "budget": "string or null (e.g. '$50k-$100k')",
        "timeline": "string or null (e.g. '3 months')",
        "stakeholders": ["list of names/roles"],
        "additional_context": "string or null"
    },
    "missing_fields": ["list of field names that are missing or vague"]
}

Return ONLY valid JSON. No markdown, no explanation, no code blocks.
"""


async def intake_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Extract structured client information from a raw transcript.

    Input state keys:
        - raw_transcript: str

    Output state keys:
        - client_info: dict (serialized ClientInfo)
        - missing_fields: list[str]
        - status: "intake"
    """
    settings = get_settings()
    transcript = state.get("raw_transcript", "")

    if not transcript:
        logger.error("Intake agent received empty transcript")
        return {
            **state,
            "client_info": {},
            "missing_fields": ["all fields — empty transcript"],
            "status": "error",
            "error_message": "Empty transcript provided",
        }

    logger.info("Intake agent processing transcript (%d chars)", len(transcript))

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0,  # Deterministic for extraction
        max_tokens=2000,
    )

    # Call the LLM
    messages = [
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Extract client information from this transcript:\n\n{transcript}"
        ),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw_output = response.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        parsed = json.loads(raw_output)

        # Validate with Pydantic
        client_data = parsed.get("client_info", parsed)
        client_info = ClientInfo(**client_data)
        missing = parsed.get("missing_fields", [])

        # Auto-detect missing fields if LLM didn't flag them
        if not missing:
            missing = _detect_missing_fields(client_info)

        logger.info(
            "Intake complete: company=%s, industry=%s, missing=%s",
            client_info.company_name,
            client_info.industry,
            missing,
        )

        return {
            **state,
            "client_info": client_info.model_dump(),
            "missing_fields": missing,
            "status": "intake",
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        return {
            **state,
            "client_info": {},
            "missing_fields": ["all fields — LLM response was not valid JSON"],
            "status": "error",
            "error_message": f"JSON parse error: {e}",
        }
    except Exception as e:
        logger.error("Intake agent failed: %s", e)
        return {
            **state,
            "client_info": {},
            "missing_fields": ["all fields — intake agent error"],
            "status": "error",
            "error_message": str(e),
        }


def _detect_missing_fields(info: ClientInfo) -> list[str]:
    """Auto-detect which fields are missing or empty."""
    missing = []
    if not info.company_name:
        missing.append("company_name")
    if not info.industry:
        missing.append("industry")
    if not info.problem_statement:
        missing.append("problem_statement")
    if not info.goals:
        missing.append("goals")
    if not info.budget:
        missing.append("budget")
    if not info.timeline:
        missing.append("timeline")
    if not info.stakeholders:
        missing.append("stakeholders")
    return missing
