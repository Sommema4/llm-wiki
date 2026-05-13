import fitz  # PyMuPDF
from typing import Tuple


def extract_pdf_text(file_path: str, max_chars: int = 500000) -> Tuple[str, int, int]:
    """Extract plain text from a PDF, up to max_chars characters.

    Returns:
        (text, pages_extracted, total_pages)
    """
    doc = fitz.open(file_path)
    total_pages = len(doc)
    parts: list[str] = []
    total = 0
    pages_extracted = 0

    for page in doc:
        page_text = page.get_text("text")
        parts.append(page_text)
        total += len(page_text)
        pages_extracted += 1
        if total >= max_chars:
            break

    doc.close()
    return "\n".join(parts)[:max_chars], pages_extracted, total_pages
