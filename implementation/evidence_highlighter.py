from __future__ import annotations

from pathlib import Path
import re

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source_documents"
OUTPUT_DIR = PROJECT_ROOT / "data" / "highlighted_pdfs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _sentences(text: str) -> list[str]:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) >= 40
    ]


def create_highlighted_pdf(source: dict) -> Path:
    pdf_name = source["source_pdf"]
    page_number = int(source["page"])
    evidence_text = source["text"]

    pdf_path = SOURCE_DIR / pdf_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    try:
        page_index = page_number - 1

        if page_index < 0 or page_index >= len(doc):
            raise ValueError(
                f"Invalid page {page_number} for {pdf_name}"
            )

        page = doc[page_index]
        highlight_count = 0

        for sentence in _sentences(evidence_text):
            for rect in page.search_for(sentence):
                page.add_highlight_annot(rect)
                highlight_count += 1

        if highlight_count == 0:
            raise ValueError(
                f"Could not locate the retrieved evidence on "
                f"page {page_number} of {pdf_name}."
            )

        output_path = OUTPUT_DIR / (
            f"{Path(pdf_name).stem}_p{page_number}_highlighted.pdf"
        )

        doc.save(output_path)

    finally:
        doc.close()

    return output_path
