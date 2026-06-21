"""phase 9 — add details JSONB to query_logs

Adds:
    * `query_logs.details` — JSONB blob recording per-question observability
      context that doesn't deserve a typed column: the retrieval mode + top_k,
      per-chunk retrieval scores (similarity + component scores), how many
      citations were kept vs. dropped by server-side validation, the
      confidence label, and whether the answer was a refusal.

      Mirrors the `evaluation_results.details` pattern from migration 0003 —
      JSONB keeps future signal additions migration-free, and it makes the
      QueryLog audit row a complete record for after-the-fact hallucination /
      latency debugging.

Revision ID: 0004
Revises: 0003
Create Date: 2026-01-04 00:00:00 UTC
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "query_logs",
        sa.Column(
            "details",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("query_logs", "details")
