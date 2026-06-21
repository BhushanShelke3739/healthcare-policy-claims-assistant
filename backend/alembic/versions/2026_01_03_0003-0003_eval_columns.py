"""phase 6 — add latency_ms + details to evaluation_results

Adds:
    * `evaluation_results.latency_ms` — wall-clock time for the per-
      question pipeline run.
    * `evaluation_results.details` — JSONB blob for everything that
      doesn't fit a typed column (retrieved chunk IDs, per-question
      retrieval_hit flag, expected_document echo, keyword recall
      breakdown, etc.). Using JSONB instead of carving out N more
      typed columns keeps future metric additions migration-free.

Revision ID: 0003
Revises: 0002
Create Date: 2026-01-03 00:00:00 UTC
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "evaluation_results",
        sa.Column("latency_ms", sa.Integer(), nullable=True),
    )
    op.add_column(
        "evaluation_results",
        sa.Column(
            "details",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("evaluation_results", "details")
    op.drop_column("evaluation_results", "latency_ms")
