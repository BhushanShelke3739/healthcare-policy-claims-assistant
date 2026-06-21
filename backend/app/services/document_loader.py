"""
Document loader.

Turns raw bytes from an upload (or a file on disk) into a single normalized
text string + a small bag of metadata. The chunking service then takes that
string apart.

Supported formats: `.txt`, `.md`, `.pdf`.

We keep this layer dumb on purpose:
    * No DB writes here — that's the ingestion orchestrator's job.
    * No chunking here — also the orchestrator's job.
    * No format-specific Pydantic schemas — caller gets `LoadedDocument`.

That separation makes it trivial to add new formats (e.g. .docx) later
without touching the API layer.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.utils.text import normalize_whitespace

logger = logging.getLogger(__name__)


# Extension → mime label used for diagnostics / metadata. Keep narrow on
# purpose — anything not here is rejected at the API boundary.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf")


@dataclass(frozen=True)
class LoadedDocument:
    """Result of loading one file."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file's extension isn't in SUPPORTED_EXTENSIONS."""


class DocumentLoadError(RuntimeError):
    """Raised when extraction fails (corrupt PDF, encoding issue, ...)."""


# =============================================================================
# Public entry point
# =============================================================================
def load_document(file_bytes: bytes, file_name: str) -> LoadedDocument:
    """
    Detect the file type from `file_name` and extract text.

    Args:
        file_bytes: raw bytes (what FastAPI's UploadFile.read() gives you).
        file_name: original name, used for extension detection and provenance.

    Returns:
        LoadedDocument with whitespace-normalized text + metadata.

    Raises:
        UnsupportedFileTypeError: extension not in SUPPORTED_EXTENSIONS.
        DocumentLoadError: extraction failed (corrupt file, encoding issue).
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension {ext!r}. " f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext in (".txt", ".md"):
        text, extra = _load_plaintext(file_bytes)
    elif ext == ".pdf":
        text, extra = _load_pdf(file_bytes)
    else:  # pragma: no cover — guarded by the check above
        raise UnsupportedFileTypeError(f"No loader for extension {ext!r}")

    return LoadedDocument(
        text=normalize_whitespace(text),
        metadata={
            "source_file_name": file_name,
            "source_extension": ext,
            "raw_byte_size": len(file_bytes),
            **extra,
        },
    )


# =============================================================================
# Format-specific loaders
# =============================================================================
def _load_plaintext(file_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """
    Decode UTF-8 with replacement.

    `errors="replace"` substitutes the U+FFFD replacement character for any
    invalid byte sequence — better than crashing on a stray Windows-1252
    smart quote in an otherwise good document.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    return text, {"loader": "plaintext"}


def _load_pdf(file_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """
    Extract text from every page of the PDF.

    `pypdf` does pure-Python text extraction. It handles selectable text
    well; it doesn't OCR scanned image PDFs. Detecting and routing those
    to OCR (e.g. Tesseract or Document AI) is a Phase 9/10 enhancement.

    Pages are joined with double newlines so the chunker treats page
    breaks as paragraph-equivalent boundaries.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except (PdfReadError, Exception) as exc:
        raise DocumentLoadError(f"could not parse PDF: {exc}") from exc

    pages: list[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "pdf_page_extraction_failed",
                extra={"page_index": idx, "error": str(exc)},
            )
            extracted = ""
        pages.append(extracted)

    if not any(p.strip() for p in pages):
        # All pages empty — likely a scanned/image PDF without OCR.
        raise DocumentLoadError(
            "PDF contained no extractable text — likely an image-only "
            "(scanned) document. OCR support will be added in a later phase."
        )

    full_text = "\n\n".join(pages)
    return full_text, {
        "loader": "pdf",
        "page_count": len(reader.pages),
    }
