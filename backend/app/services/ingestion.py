"""
Ingestion orchestrator.

Glues the loader, chunker, and database together. Both the HTTP upload
handler and the offline seed script call into here so the flow is in
exactly one place.

A successful ingest is one database transaction:
    INSERT documents (1 row)
    INSERT document_chunks (N rows)

If anything fails partway through, the transaction is rolled back and no
half-ingested document is left behind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk
from app.services.chunking import chunk_document
from app.services.document_loader import LoadedDocument, load_document
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of an ingest. `chunks_created` is the count actually persisted."""

    document: Document
    chunks_created: int
    extraction_metadata: dict[str, Any]


def ingest_bytes(
    db: Session,
    *,
    file_bytes: bytes,
    file_name: str,
    title: str | None = None,
    document_type: str = "policy",
    source_type: str = "upload",
    extra_metadata: dict[str, Any] | None = None,
    embed: bool = True,
) -> IngestResult:
    """
    Load → chunk → persist.

    Args:
        db: an open SQLAlchemy Session. Caller owns commit/rollback. The
            function itself flushes so the new IDs are available, but does
            not commit — the API handler decides the transaction boundary.
        file_bytes: raw file contents.
        file_name: original file name (used for extension detection + the
            Document.file_name column).
        title: human-readable title. Falls back to the file name's stem.
        document_type: domain tag stored on the Document row.
        source_type: provenance tag ("upload" | "seed" | ...).
        extra_metadata: additional metadata copied onto each chunk
            (in addition to the loader's metadata).

    Returns:
        IngestResult — the persisted Document, the number of chunks
        created, and the loader's extraction metadata for diagnostics.
    """
    settings = get_settings()
    effective_title = title or _title_from_filename(file_name)

    # 1) Extract text + format metadata.
    loaded: LoadedDocument = load_document(file_bytes, file_name)

    # 2) Chunk. Per-chunk metadata carries everything a future retriever
    #    needs to attribute a snippet back to its source.
    chunk_metadata = {
        "document_title": effective_title,
        "document_type": document_type,
        "source_file_name": file_name,
        **loaded.metadata,
        **(extra_metadata or {}),
    }
    chunks = chunk_document(
        loaded.text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        metadata=chunk_metadata,
    )

    # 3) Embed chunk text in one batched call. The embedder is a process
    #    singleton; the mock provider is free, the OpenAI provider batches
    #    internally up to its own input cap. The list is typed with
    #    `Optional[list[float]]` because the `embed=False` branch fills
    #    it with Nones — those land as NULL in the embedding column,
    #    which the Phase 3 backfill CLI populates later.
    embeddings: list[list[float] | None]
    if embed and chunks:
        embedder = get_embedder()
        embeddings = list(embedder.embed([c.text for c in chunks]))
    else:
        embeddings = [None] * len(chunks)

    # 4) Persist as one logical unit (caller commits).
    doc = Document(
        title=effective_title,
        source_type=source_type,
        file_name=file_name,
        document_type=document_type,
    )
    db.add(doc)
    db.flush()  # populate doc.id without committing.

    for c, vec in zip(chunks, embeddings, strict=True):
        db.add(
            DocumentChunk(
                document_id=doc.id,
                chunk_text=c.text,
                chunk_index=c.index,
                token_count=c.token_count,
                chunk_metadata=c.metadata,
                embedding=vec,
            )
        )

    logger.info(
        "document_ingested",
        extra={
            "document_id": str(doc.id),
            "title": effective_title,
            "chunks_created": len(chunks),
            "chunks_embedded": sum(1 for v in embeddings if v is not None),
            "source_extension": loaded.metadata.get("source_extension"),
            "raw_byte_size": loaded.metadata.get("raw_byte_size"),
        },
    )

    return IngestResult(
        document=doc,
        chunks_created=len(chunks),
        extraction_metadata=loaded.metadata,
    )


def _title_from_filename(file_name: str) -> str:
    """`prior_authorization_policy.txt` -> `Prior Authorization Policy`."""
    from pathlib import Path

    stem = Path(file_name).stem
    return stem.replace("_", " ").replace("-", " ").strip().title()
