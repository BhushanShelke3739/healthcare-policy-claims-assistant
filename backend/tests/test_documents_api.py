"""
Integration tests for the /documents/* HTTP endpoints.

These run against the real Postgres via the `client_with_real_db` fixture
(transactional, rolled back at end of test). Embeddings come from the
mock provider via the autouse fixture in conftest.

What's covered here that the Phase 2 service-level tests don't catch:
    * multipart upload mechanics (FastAPI's UploadFile path)
    * HTTP status codes (415 unsupported, 413 too large, 422 corrupt PDF)
    * query-param plumbing (title, document_type)
    * the LEFT-JOIN aggregation query in list_documents
    * 404 on get/delete of unknown UUIDs
"""

from __future__ import annotations

import io
import uuid


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def test_upload_txt_creates_document_and_chunks(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/documents/upload?title=Test%20Upload&document_type=test_corpus",
        files={"file": ("test.txt", b"Prior authorization is required.", "text/plain")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["document"]["title"] == "Test Upload"
    assert payload["document"]["document_type"] == "test_corpus"
    assert payload["chunks_created"] >= 1
    assert payload["bytes_received"] == len(b"Prior authorization is required.")
    assert payload["extraction_metadata"]["loader"] == "plaintext"


def test_upload_md_uses_filename_as_default_title(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/documents/upload",
        files={"file": ("appeal_policy.md", b"# Appeals\n\nFile within 60 days.", "text/markdown")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    # Falls back to a humanized version of the file stem.
    assert payload["document"]["title"] == "Appeal Policy"


def test_upload_rejects_unsupported_extension(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/documents/upload",
        files={"file": ("policy.docx", b"PK\x03\x04", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert "supported" in response.json()["detail"].lower()


def test_upload_rejects_empty_file(client_with_real_db) -> None:
    response = client_with_real_db.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_oversized_file(client_with_real_db, monkeypatch) -> None:
    # Reach into settings to make the cap tiny for this test only.
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_size_mb", 1, raising=False)

    big = b"x" * (2 * 1024 * 1024)  # 2 MB > 1 MB cap
    response = client_with_real_db.post(
        "/documents/upload",
        files={"file": ("big.txt", big, "text/plain")},
    )
    assert response.status_code == 413
    assert "max" in response.json()["detail"].lower()


def test_upload_rejects_corrupt_pdf(client_with_real_db) -> None:
    """A file that claims to be PDF but isn't parseable returns 422."""
    response = client_with_real_db.post(
        "/documents/upload",
        files={"file": ("bad.pdf", b"%PDF-not-really-a-pdf", "application/pdf")},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------
def test_list_returns_uploaded_documents(client_with_real_db) -> None:
    # Upload two docs so we have something specific to assert against.
    client_with_real_db.post(
        "/documents/upload?title=List%20A&document_type=test_listing",
        files={"file": ("a.txt", b"alpha content one", "text/plain")},
    )
    client_with_real_db.post(
        "/documents/upload?title=List%20B&document_type=test_listing",
        files={"file": ("b.txt", b"bravo content two", "text/plain")},
    )

    response = client_with_real_db.get("/documents?document_type=test_listing")
    assert response.status_code == 200
    body = response.json()
    titles = [item["title"] for item in body["items"]]
    assert "List A" in titles
    assert "List B" in titles
    # Each list row carries a server-computed chunk_count.
    for item in body["items"]:
        if item["title"] in ("List A", "List B"):
            assert item["chunk_count"] >= 1


def test_list_paginates_with_limit_and_offset(client_with_real_db) -> None:
    # Upload three docs.
    for i in range(3):
        client_with_real_db.post(
            "/documents/upload?title=Page" + str(i) + "&document_type=test_paging",
            files={"file": (f"p{i}.txt", f"content {i}".encode(), "text/plain")},
        )

    page1 = client_with_real_db.get("/documents?document_type=test_paging&limit=2&offset=0")
    page2 = client_with_real_db.get("/documents?document_type=test_paging&limit=2&offset=2")
    assert page1.status_code == 200
    assert page2.status_code == 200
    assert len(page1.json()["items"]) == 2
    assert len(page2.json()["items"]) == 1


# ---------------------------------------------------------------------------
# Get / Delete
# ---------------------------------------------------------------------------
def test_get_then_delete_roundtrip(client_with_real_db) -> None:
    upload = client_with_real_db.post(
        "/documents/upload?title=Roundtrip&document_type=test_roundtrip",
        files={"file": ("rt.txt", b"some text for the roundtrip test", "text/plain")},
    )
    doc_id = upload.json()["document"]["id"]

    fetch = client_with_real_db.get(f"/documents/{doc_id}")
    assert fetch.status_code == 200
    body = fetch.json()
    assert body["id"] == doc_id
    assert body["title"] == "Roundtrip"
    assert body["chunk_count"] >= 1

    delete = client_with_real_db.delete(f"/documents/{doc_id}")
    assert delete.status_code == 204

    after = client_with_real_db.get(f"/documents/{doc_id}")
    assert after.status_code == 404


def test_get_nonexistent_returns_404(client_with_real_db) -> None:
    fake = uuid.uuid4()
    response = client_with_real_db.get(f"/documents/{fake}")
    assert response.status_code == 404


def test_delete_nonexistent_returns_404(client_with_real_db) -> None:
    fake = uuid.uuid4()
    response = client_with_real_db.delete(f"/documents/{fake}")
    assert response.status_code == 404
