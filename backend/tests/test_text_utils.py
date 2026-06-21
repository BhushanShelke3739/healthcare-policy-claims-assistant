"""Unit tests for the small text utilities used across the project."""

from __future__ import annotations

from app.utils.ids import new_uuid, new_uuid_str
from app.utils.text import approximate_token_count, normalize_whitespace


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------
def test_normalize_collapses_runs_of_whitespace() -> None:
    assert normalize_whitespace("a  b\t c\n\nd") == "a b c d"


def test_normalize_trims_edges() -> None:
    assert normalize_whitespace("   hello   ") == "hello"


def test_normalize_handles_empty_string() -> None:
    assert normalize_whitespace("") == ""


def test_normalize_handles_only_whitespace() -> None:
    assert normalize_whitespace("   \n\t  ") == ""


# ---------------------------------------------------------------------------
# approximate_token_count
# ---------------------------------------------------------------------------
def test_token_count_grows_with_length() -> None:
    assert approximate_token_count("abc") <= approximate_token_count("abc def ghi jkl")


def test_token_count_floor_is_one() -> None:
    assert approximate_token_count("") >= 1


def test_token_count_uses_four_chars_per_token_heuristic() -> None:
    text = "a" * 40  # 40 chars → ~10 tokens
    assert 8 <= approximate_token_count(text) <= 12


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------
def test_new_uuid_is_unique() -> None:
    seen = {new_uuid() for _ in range(100)}
    assert len(seen) == 100


def test_new_uuid_str_matches_canonical_form() -> None:
    import re

    pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    for _ in range(10):
        assert pattern.match(new_uuid_str())
