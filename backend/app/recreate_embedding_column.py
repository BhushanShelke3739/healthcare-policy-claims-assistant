"""
Recreate `document_chunks.embedding` at a new vector dimension.

Use when switching to an embedding model with a different output size — e.g.
moving from OpenAI's `text-embedding-3-small` (1536) to Ollama's
`nomic-embed-text` (768) or `all-minilm` (384).

What it does:
    1. DROP the HNSW index on `embedding`.
    2. DROP the `embedding` column.
    3. ADD `embedding` back at the requested dimension.
    4. Recreate the HNSW index.

This is destructive — every existing embedding is wiped. Run
`python -m app.backfill_embeddings --all` afterward to repopulate under
the new model.

Run with:
    python -m app.recreate_embedding_column 768
    python -m app.recreate_embedding_column 1536 --yes
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import text

from app.db.session import engine


def recreate(new_dim: int) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw"))
        conn.execute(text("ALTER TABLE document_chunks DROP COLUMN IF EXISTS embedding"))
        conn.execute(text(f"ALTER TABLE document_chunks ADD COLUMN embedding vector({new_dim})"))
        conn.execute(
            text(
                "CREATE INDEX ix_document_chunks_embedding_hnsw "
                "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recreate document_chunks.embedding at a new dimension."
    )
    parser.add_argument(
        "dim",
        type=int,
        help="New vector dimension (e.g. 384, 768, 1024, 1536).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the destructive-operation prompt.",
    )
    args = parser.parse_args(argv)

    if args.dim < 1 or args.dim > 16000:
        print(f"Error: dim must be in [1, 16000]; got {args.dim}", file=sys.stderr)
        return 2

    if not args.yes:
        print(
            "This will DROP and RECREATE the document_chunks.embedding column.\n"
            "All existing embeddings will be deleted.\n"
            f"New dimension: {args.dim}\n"
        )
        reply = input("Type 'yes' to proceed: ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return 1

    recreate(args.dim)
    print(
        f"\nDone. `embedding` is now vector({args.dim}).\n"
        "Next steps:\n"
        f"  1. Update .env: EMBEDDING_DIMENSIONS={args.dim} "
        "(and EMBEDDING_PROVIDER / EMBEDDING_MODEL if changing them).\n"
        "  2. Restart the backend so the new settings load.\n"
        "  3. python -m app.backfill_embeddings --all"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
