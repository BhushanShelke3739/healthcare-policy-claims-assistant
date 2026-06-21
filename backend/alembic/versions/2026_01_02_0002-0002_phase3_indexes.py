"""phase 3 indexes — tsvector column + HNSW vector index

Adds:
    * `document_chunks.tsv` — a PostgreSQL `tsvector` GENERATED column
      populated automatically from `chunk_text`. Keeps the FTS vector
      perfectly in sync with the source text without a trigger.
    * GIN index on `tsv` — standard full-text-search index.
    * HNSW index on `embedding` using cosine distance — fast approximate
      nearest-neighbor search. HNSW handles NULL embeddings fine (skips
      them) so the index can exist before backfill.

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-02 00:00:00 UTC
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---- Full-text search column + index --------------------------------
    # `GENERATED ALWAYS AS ... STORED` makes Postgres recompute the tsvector
    # on every insert/update of chunk_text. Zero application-side bookkeeping.
    op.execute(
        """
        ALTER TABLE document_chunks
        ADD COLUMN tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
        """
    )
    # GIN is the standard index type for tsvector columns.
    op.execute(
        "CREATE INDEX ix_document_chunks_tsv ON document_chunks USING GIN (tsv)"
    )

    # ---- Vector index ---------------------------------------------------
    # HNSW with cosine distance. Default m=16, ef_construction=64 are fine
    # at our scale; pgvector docs recommend bumping for very large corpora.
    # We use `vector_cosine_ops` because cosine is the dominant choice for
    # sentence/embedding similarity.
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS tsv")
