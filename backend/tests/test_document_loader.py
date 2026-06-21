"""
Unit tests for the document loader.

PDF parsing is covered with a tiny in-memory PDF generated via pypdf so
the test suite remains hermetic — no fixture files on disk to drift.
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.services.document_loader import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    load_document,
)


# ---------------------------------------------------------------------------
# Plaintext / markdown
# ---------------------------------------------------------------------------
def test_loads_txt_and_normalizes_whitespace() -> None:
    raw = b"  Prior authorization\n\nis required.  "
    loaded = load_document(raw, "policy.txt")

    assert loaded.text == "Prior authorization is required."
    assert loaded.metadata["loader"] == "plaintext"
    assert loaded.metadata["source_extension"] == ".txt"
    assert loaded.metadata["raw_byte_size"] == len(raw)


def test_loads_markdown_the_same_way_as_txt() -> None:
    loaded = load_document(b"# Appeals\n\nFile within 60 days.", "appeals.md")
    assert "Appeals" in loaded.text
    assert "60 days" in loaded.text
    assert loaded.metadata["source_extension"] == ".md"


def test_invalid_utf8_falls_back_to_replacement_char() -> None:
    """Stray non-UTF-8 bytes shouldn't crash the loader."""
    raw = b"valid text \xff\xfe more text"
    loaded = load_document(raw, "messy.txt")

    assert "valid text" in loaded.text
    assert "more text" in loaded.text
    # U+FFFD (replacement) survives normalize_whitespace.
    assert "�" in loaded.text


# ---------------------------------------------------------------------------
# Extension validation
# ---------------------------------------------------------------------------
def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        load_document(b"anything", "policy.docx")


def test_extension_check_is_case_insensitive() -> None:
    loaded = load_document(b"hello", "POLICY.TXT")
    assert loaded.text == "hello"


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _build_pdf_with_text(pages_text: list[str]) -> bytes:
    """
    Build a real (small) PDF in memory.

    pypdf can write blank pages but not text-onto-pages, so we generate a
    minimal valid PDF manually for each piece of text. Using reportlab
    would be cleaner but it's an extra dependency we don't otherwise need.
    """
    # A minimal text PDF for each page. This is hand-crafted but valid;
    # pypdf can read it back.
    pdf_objects = []
    for text in pages_text:
        # Escape parens for the PDF text operator.
        escaped = text.replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET".encode("latin-1")
        pdf_objects.append(content)

    # Assemble a minimal PDF. For test purposes a single-page is enough — we
    # don't need to cover multi-page packing because pypdf itself owns that.
    # If multi-page testing is needed, expand via PdfWriter.
    writer = PdfWriter()
    # Easier: build a one-page PDF per text and merge.
    for content_stream in pdf_objects:
        # Each call to add_blank_page returns the new page; we mutate its
        # /Contents object directly via the writer's underlying objects.
        from pypdf.generic import (
            ArrayObject,
            ContentStream,
            DecodedStreamObject,
            DictionaryObject,
            NameObject,
            NumberObject,
        )

        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(content_stream)
        page[NameObject("/Contents")] = stream

        # Provide a Helvetica font so /F1 in the content stream resolves.
        font_dict = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_dict})}
        )
        page[NameObject("/Resources")] = resources
        page[NameObject("/MediaBox")] = ArrayObject(
            [NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)]
        )

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_loads_pdf_and_extracts_page_text() -> None:
    pdf_bytes = _build_pdf_with_text(["Hello world from a PDF."])

    loaded = load_document(pdf_bytes, "sample.pdf")

    assert "Hello world" in loaded.text
    assert loaded.metadata["loader"] == "pdf"
    assert loaded.metadata["page_count"] == 1


def test_pdf_with_no_extractable_text_raises() -> None:
    # A PDF with zero pages by passing an empty list builds a PDF that has
    # no content; pypdf will refuse to read 0-page PDFs in some versions.
    # Use a single page with no text instead.
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(DocumentLoadError):
        load_document(buf.getvalue(), "blank.pdf")


def test_corrupt_pdf_raises() -> None:
    with pytest.raises(DocumentLoadError):
        load_document(b"%PDF-not-really", "corrupt.pdf")
