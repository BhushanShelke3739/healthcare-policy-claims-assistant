"""
SQLAlchemy declarative base.

We use the 2.x style `DeclarativeBase` so models get full type-hint support
(`Mapped[...]` columns) and play nicely with `mypy`.

Keeping the base in its own module avoids the classic circular import where
`session.py` and `models.py` both want to know about the metadata.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Common declarative base for every ORM model in the project."""
