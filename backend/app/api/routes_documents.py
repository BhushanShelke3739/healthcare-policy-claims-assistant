"""
Document endpoints — Phase 2 implementation.

Routes:
    POST   /documents/upload
    GET    /documents
    GET    /documents/{document_id}
    DELETE /documents/{document_id}

Embedding generation does *not* happen here — Phase 3 adds a follow-on
step (synchronous for now, async / batched later) that backfills the
`embedding` column on every chunk.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk
from app.db.session import get_db
from app.schemas.documents import (
    DocumentList,
    DocumentReadWithChunkCount,
    DocumentUploadResponse,
)
from app.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    UnsupportedFileTypeError,
)
from app.services.ingestion import ingest_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Upload
# =============================================================================
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document and ingest it (load → chunk → persist).",
)
async def upload_document(
    file: Annotated[UploadFile, File(..., description="The file to ingest.")],
    title: Annotated[
        str | None,
        Query(
            description="Human-readable title. Defaults to the file's stem.",
            max_length=512,
        ),
    ] = None,
    document_type: Annotated[
        str,
        Query(
            description="Domain tag (policy | claim | appeal | compliance | ...).",
            max_length=64,
        ),
    ] = "policy",
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing filename")

    # Read fully into memory. For Phase 2 + the synthetic corpus this is
    # fine; if we ever accept multi-hundred-MB uploads we'll stream to disk
    # and parse incrementally instead.
    raw = await file.read()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"file is {len(raw)} bytes; max allowed is {max_bytes} "
            f"({settings.max_upload_size_mb} MB)",
        )
    if len(raw) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    try:
        result = ingest_bytes(
            db,
            file_bytes=raw,
            file_name=file.filename,
            title=title,
            document_type=document_type,
            source_type="upload",
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{exc}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}",
        ) from exc
    except DocumentLoadError as exc:
        # Extraction itself failed (corrupt file, image-only PDF, ...).
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    db.commit()
    db.refresh(result.document)

    return DocumentUploadResponse(
        document=DocumentReadWithChunkCount(
            id=result.document.id,
            title=result.document.title,
            source_type=result.document.source_type,
            file_name=result.document.file_name,
            document_type=result.document.document_type,
            created_at=result.document.created_at,
            chunk_count=result.chunks_created,
        ),
        chunks_created=result.chunks_created,
        bytes_received=len(raw),
        extraction_metadata=result.extraction_metadata,
    )


# =============================================================================
# List
# =============================================================================
@router.get("", response_model=DocumentList, summary="List ingested documents.")
def list_documents(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    document_type: Annotated[str | None, Query(max_length=64)] = None,
    db: Session = Depends(get_db),
) -> DocumentList:
    """Paged list of documents with their chunk counts."""

    # Aggregate chunk_count in one query to avoid the N+1 you'd get from
    # `len(doc.chunks)` in a loop.
    chunk_count_subq = (
        select(
            DocumentChunk.document_id.label("document_id"),
            func.count(DocumentChunk.id).label("chunk_count"),
        )
        .group_by(DocumentChunk.document_id)
        .subquery()
    )

    stmt = (
        select(Document, func.coalesce(chunk_count_subq.c.chunk_count, 0))
        .outerjoin(chunk_count_subq, Document.id == chunk_count_subq.c.document_id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)

    rows = db.execute(stmt).all()

    total = db.execute(
        select(func.count(Document.id)).where(Document.document_type == document_type)
        if document_type
        else select(func.count(Document.id))
    ).scalar_one()

    return DocumentList(
        items=[
            DocumentReadWithChunkCount(
                id=doc.id,
                title=doc.title,
                source_type=doc.source_type,
                file_name=doc.file_name,
                document_type=doc.document_type,
                created_at=doc.created_at,
                chunk_count=int(count),
            )
            for doc, count in rows
        ],
        total=int(total),
    )


# =============================================================================
# Get one
# =============================================================================
@router.get(
    "/{document_id}",
    response_model=DocumentReadWithChunkCount,
    summary="Get a single document by id.",
)
def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentReadWithChunkCount:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")

    chunk_count = db.execute(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
    ).scalar_one()

    return DocumentReadWithChunkCount(
        id=doc.id,
        title=doc.title,
        source_type=doc.source_type,
        file_name=doc.file_name,
        document_type=doc.document_type,
        created_at=doc.created_at,
        chunk_count=int(chunk_count),
    )


# =============================================================================
# Delete
# =============================================================================
@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    # `response_model=None` opts out of FastAPI inferring a response model
    # from the return annotation. Without this, a `-> None` annotation
    # would be picked up as `response_model=type(None)`, which trips
    # FastAPI's "204 cannot have a body" assertion at app startup.
    response_model=None,
    summary="Delete a document and all of its chunks.",
)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")

    # `Document.chunks` has cascade="all, delete-orphan", and the FK on
    # `document_chunks.document_id` uses ON DELETE CASCADE, so the chunks
    # disappear with the parent automatically.
    db.delete(doc)
    db.commit()

    logger.info("document_deleted", extra={"document_id": str(document_id)})
