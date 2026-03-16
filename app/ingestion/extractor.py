import io


def extract_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text page by page from a PDF. Returns list of {text, page_num}."""
    import pymupdf  # PyMuPDF

    try:
        pages = []
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append({"text": text, "page_num": page_num})
        if not pages:
            raise ValueError("PDF contains no extractable text (may be scanned/image-only).")
        return pages
    except pymupdf.mupdf.FzErrorFormat as e:
        raise ValueError(f"Corrupted or invalid PDF: {e}") from e
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to extract PDF: {e}") from e


def extract_docx(file_bytes: bytes) -> list[dict]:
    """Extract text from DOCX. Returns list of {text, page_num: None}."""
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError

    try:
        doc = Document(io.BytesIO(file_bytes))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not full_text.strip():
            raise ValueError("DOCX contains no extractable text.")
        return [{"text": full_text, "page_num": None}]
    except PackageNotFoundError as e:
        raise ValueError(f"Corrupted or invalid DOCX file: {e}") from e
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to extract DOCX: {e}") from e


def extract(filename: str, file_bytes: bytes) -> list[dict]:
    """Dispatch extraction based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_pdf(file_bytes)
    elif ext == "docx":
        return extract_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Only PDF and DOCX are supported.")