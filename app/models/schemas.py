"""
Core data models for the proposal generation pipeline.

- ClientInfo: Structured client data extracted from transcripts.
- ProposalState: LangGraph state that flows through the agent pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# ClientInfo — extracted from the sales transcript by the Intake agent
# ---------------------------------------------------------------------------

class ClientInfo(BaseModel):
    """Structured client information extracted from a discovery call transcript."""

    company_name: Optional[str] = Field(
        default=None,
        description="Client company name",
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry or sector the client operates in",
    )
    problem_statement: Optional[str] = Field(
        default=None,
        description="Core problem or challenge described by the client",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Client's goals and desired outcomes",
    )
    budget: Optional[str] = Field(
        default=None,
        description="Budget range if mentioned (e.g. '$50k-$100k')",
    )
    timeline: Optional[str] = Field(
        default=None,
        description="Project timeline if mentioned (e.g. '3 months')",
    )
    stakeholders: list[str] = Field(
        default_factory=list,
        description="Key contacts, decision makers, or stakeholders mentioned",
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Any other relevant information from the transcript",
    )


# ---------------------------------------------------------------------------
# ProposalState — LangGraph state (TypedDict for LangGraph compatibility)
# ---------------------------------------------------------------------------

class ProposalState(TypedDict, total=False):
    """
    State object that flows through the LangGraph pipeline.

    Each agent reads from and writes to this shared state.
    Using TypedDict (not Pydantic) for LangGraph compatibility.
    """

    # Slack context
    thread_ts: str
    channel_id: str
    user_id: str

    # Raw input
    raw_transcript: str

    # Agent 1: Intake output
    client_info: dict[str, Any]  # Serialized ClientInfo
    missing_fields: list[str]

    # Human-in-loop: intake confirmation
    user_confirmed: bool

    # Agent 2: Research output
    similar_proposals: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    rag_status: str  # "matched" | "web_search" | "zero_shot"

    # Human-in-loop: internet search decision
    search_internet: Optional[bool]

    # Human-in-loop: tone selection
    tone: Optional[str]  # "formal" | "technical" | "consultative" | "industry_default"

    # Agent 3: Writer output
    draft_proposal: Optional[str]  # Markdown text
    docx_bytes: Optional[bytes]

    # Smart Save Gate
    save_status: Optional[str]  # "saved" | "skipped_duplicate" | "not_saved"
    qdrant_proposal_id: Optional[str]  # ID if saved, for revision updates

    # Agent 4: Revision
    revision_request: Optional[str]
    revised_section: Optional[str]

    # Pipeline tracking
    status: str  # "intake" | "researching" | "writing" | "complete" | "revising" | "error"
    error_message: Optional[str]
