"""
Section-aware text chunking for proposals and transcripts.

Splits documents by headers/sections first, then by size with overlap.
Each chunk carries metadata for filtering in Qdrant.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with metadata for vector storage."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    source_file: str = ""
    section_title: str = ""
    chunk_index: int = 0
    industry: str = ""
    source: str = "seed"  # "seed" | "generated" | "transcript"
    thread_ts: str = ""  # Links generated proposals to their session


def _extract_industry_from_filename(filename: str) -> str:
    """
    Extract industry from filename like 'proposal_healthcare.txt'.

    Examples:
        proposal_healthcare.txt -> healthcare
        proposal_retail2.txt -> retail
        proposal_fleet.txt -> fleet
    """
    name = filename.replace(".txt", "").replace(".md", "")
    # Remove 'proposal_' prefix
    if name.startswith("proposal_"):
        name = name[len("proposal_"):]
    # Remove trailing numbers (e.g. retail2 -> retail)
    name = re.sub(r"\d+$", "", name)
    return name.strip().lower()


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text by markdown-style headers or separator lines.

    Returns list of (section_title, section_text) tuples.
    """
    # Split on markdown headers (##, ###) or separator lines (---)
    section_pattern = re.compile(
        r"(?:^|\n)(#{1,3}\s+.+|[-=]{3,})\s*\n",
        re.MULTILINE,
    )

    parts = section_pattern.split(text)
    sections: list[tuple[str, str]] = []

    current_title = "Introduction"
    current_text = ""

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        # Check if this part is a header
        if re.match(r"^#{1,3}\s+", stripped):
            # Save previous section
            if current_text.strip():
                sections.append((current_title, current_text.strip()))
            current_title = stripped.lstrip("#").strip()
            current_text = ""
        elif re.match(r"^[-=]{3,}$", stripped):
            # Separator line — start new unnamed section
            if current_text.strip():
                sections.append((current_title, current_text.strip()))
            current_title = "Section"
            current_text = ""
        else:
            current_text += part

    # Don't forget the last section
    if current_text.strip():
        sections.append((current_title, current_text.strip()))

    # If no sections found, treat the whole text as one section
    if not sections:
        sections = [("Full Document", text.strip())]

    return sections


def _split_by_size(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into chunks of roughly `chunk_size` chars with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence end near the chunk boundary
            boundary = text.rfind(". ", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary + 1  # Include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def chunk_document(
    text: str,
    source_file: str = "",
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "seed",
    thread_ts: str = "",
    industry: str = "",
) -> list[Chunk]:
    """
    Split a document into chunks with metadata.

    1. Split by sections (headers/separators)
    2. Split large sections by size with overlap
    3. Attach metadata to each chunk

    Args:
        text: The full document text.
        source_file: Original filename (used for industry extraction).
        chunk_size: Target chunk size in characters.
        overlap: Overlap between consecutive chunks.
        source: "seed" | "generated" | "transcript"
        thread_ts: Slack thread timestamp (for generated/transcript chunks).
        industry: Override industry (if not extractable from filename).

    Returns:
        List of Chunk objects ready for vector storage.
    """
    if not industry and source_file:
        industry = _extract_industry_from_filename(source_file)

    sections = _split_into_sections(text)
    chunks: list[Chunk] = []
    global_index = 0

    for section_title, section_text in sections:
        sub_chunks = _split_by_size(section_text, chunk_size, overlap)

        for sub_text in sub_chunks:
            chunks.append(
                Chunk(
                    text=sub_text,
                    source_file=source_file,
                    section_title=section_title,
                    chunk_index=global_index,
                    industry=industry,
                    source=source,
                    thread_ts=thread_ts,
                )
            )
            global_index += 1

    return chunks
