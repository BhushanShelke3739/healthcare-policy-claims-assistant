"""
Backfill embeddings for chunks that don't have them yet.

Run with:
    python -m app.backfill_embeddings              # only rows where embedding IS NULL
    python -m app.backfill_embeddings --all        # re-embed every chunk
    python -m app.backfill_embeddings --batch-size 64

Necessary because:
    * Documents seeded *before* Phase 3 was implemented have NULL embeddings.
    * Switching `EMBEDDING_PROVIDER` (e.g. mock → openai) means existing
      embeddings need to be recomputed under the new model.
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import DocumentChunk
from app.db.session import SessionLocal
from app.services.embeddings import get_embedder

logger = logging.getLogger(__name__)


def backfill(*, batch_size: int = 32, all_rows: bool = False) -> int:
    """
    Embed and write back chunks in batches.

    Returns: number of chunks updated.
    """
    embedder = get_embedder()
    total = 0

    with SessionLocal() as db:
        while True:
            stmt = select(DocumentChunk.id, DocumentChunk.chunk_text).order_by(DocumentChunk.id)
            if not all_rows:
                stmt = stmt.where(DocumentChunk.embedding.is_(None))
            stmt = stmt.limit(batch_size)

            rows = db.execute(stmt).all()
            if not rows:
                break

            ids = [row.id for row in rows]
            texts = [row.chunk_text for row in rows]
            vectors = embedder.embed(texts)

            # Per-row UPDATE rather than a single bulk statement. pgvector
            # accepts Python lists directly via the SQLAlchemy adapter.
            for chunk_id, vec in zip(ids, vectors, strict=True):
                db.execute(
                    update(DocumentChunk).where(DocumentChunk.id == chunk_id).values(embedding=vec)
                )
            db.commit()
            total += len(rows)
            print(f"  embedded batch of {len(rows):>3d} (running total: {total})")

            # When `all_rows` is true and batch_size happens to equal the
            # remaining rows, the next loop iteration runs an extra query
            # that returns 0 rows and exits. That's fine.
            if all_rows and len(rows) < batch_size:
                break

    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill chunk embeddings.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-embed every chunk, not just rows with NULL embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rows per embedding API call. Defaults to 32.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    print(
        f"Backfilling embeddings  provider={settings.embedding_provider}  "
        f"model={settings.embedding_model}  dim={settings.embedding_dimensions}"
    )
    updated = backfill(batch_size=args.batch_size, all_rows=args.all)
    print(f"\nDone. Updated {updated} chunk(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
