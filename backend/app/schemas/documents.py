"""
Pydantic schemas for the documents API.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentRead(BaseModel):
    """Document representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source_type: str = Field(description="upload | url | seed")
    file_name: str | None = None
    document_type: str = Field(description="policy | claim | appeal | compliance | ...")
    created_at: datetime


class DocumentReadWithChunkCount(DocumentRead):
    """`DocumentRead` plus a server-computed chunk count."""

    chunk_count: int = Field(ge=0, description="Number of chunks produced for this document.")


class DocumentList(BaseModel):
    items: list[DocumentReadWithChunkCount]
    total: int


class DocumentUploadResponse(BaseModel):
    """Returned by POST /documents/upload after a successful ingest."""

    document: DocumentReadWithChunkCount
    chunks_created: int = Field(ge=0)
    bytes_received: int = Field(ge=0)
    extraction_metadata: dict = Field(
        default_factory=dict,
        description="Loader metadata (page_count, loader name, ...). Useful for debugging.",
    )
