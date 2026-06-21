"""
Seed script — load every file in `sample_data/policies/` into the database.

Run with:
    python -m app.seed_policies

Idempotent: by default, a policy is skipped if a document with the same
`file_name` already exists. Pass `--replace` to delete and re-ingest.

Why a script (and not just curl-loops against the API)?
    * One command for graders / interviewers to bring the system to a
      demo-ready state.
    * Exercises the same ingestion code the upload endpoint uses, so
      regressions in either path surface together.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import Document
from app.db.session import SessionLocal
from app.services.ingestion import ingest_bytes

logger = logging.getLogger(__name__)

# Resolve the repo's sample_data directory from this file's location so
# the script works no matter the current working directory.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
POLICIES_DIR = _REPO_ROOT / "sample_data" / "policies"


def seed_policies(replace: bool = False) -> int:
    """
    Ingest every `.txt`, `.md`, or `.pdf` under sample_data/policies/.

    Returns: number of new documents inserted.
    """
    if not POLICIES_DIR.exists():
        logger.error("policies_directory_missing", extra={"path": str(POLICIES_DIR)})
        return 0

    files = sorted(
        p
        for p in POLICIES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    if not files:
        logger.warning("no_policy_files_found", extra={"path": str(POLICIES_DIR)})
        return 0

    inserted = 0
    with SessionLocal() as db:
        for path in files:
            existing = db.execute(
                select(Document).where(Document.file_name == path.name)
            ).scalar_one_or_none()

            if existing is not None and not replace:
                logger.info(
                    "policy_skip_existing",
                    extra={"file_name": path.name, "document_id": str(existing.id)},
                )
                continue

            if existing is not None and replace:
                logger.info(
                    "policy_replace_existing",
                    extra={"file_name": path.name, "document_id": str(existing.id)},
                )
                db.delete(existing)
                db.flush()

            file_bytes = path.read_bytes()
            result = ingest_bytes(
                db,
                file_bytes=file_bytes,
                file_name=path.name,
                document_type="policy",
                source_type="seed",
            )
            db.commit()
            inserted += 1
            print(
                f"  + {path.name:48s}  "
                f"{result.chunks_created:>3d} chunks  "
                f"(id={result.document.id})"
            )

    return inserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed the synthetic healthcare policies into the database."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and re-ingest documents that already exist (by file_name).",
    )
    args = parser.parse_args(argv)

    configure_logging(get_settings().log_level)

    print(f"Seeding policies from {POLICIES_DIR}")
    inserted = seed_policies(replace=args.replace)
    print(f"\nDone. Inserted {inserted} document(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
