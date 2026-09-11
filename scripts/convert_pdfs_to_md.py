from pathlib import Path

import fitz  # PyMuPDF


PROJECT_ROOT = Path(__file__).parent.parent

PDF_DIR = PROJECT_ROOT / "source_documents"
MD_DIR = PROJECT_ROOT / "data" / "markdown"

MD_DIR.mkdir(parents=True, exist_ok=True)


pdf_files = sorted(PDF_DIR.glob("*.pdf"))

print(f"PDF files found: {len(pdf_files)}")
print(f"Markdown output directory: {MD_DIR}")
print()


converted = 0
failed = []


for index, pdf_file in enumerate(pdf_files, start=1):

    output_file = MD_DIR / f"{pdf_file.stem}.md"

    print(
        f"[{index}/{len(pdf_files)}] "
        f"Converting {pdf_file.name} ..."
    )

    try:

        document = fitz.open(pdf_file)

        sections = [
            f"# {pdf_file.stem}",
            "",
            f"Source PDF: `{pdf_file.name}`",
            "",
        ]

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            sections.append(
                f"## Page {page_number}"
            )

            sections.append("")

            if text:
                sections.append(text)
            else:
                sections.append(
                    "[No extractable text on this page]"
                )

            sections.append("")

        document.close()

        output_file.write_text(
            "\n".join(sections),
            encoding="utf-8",
        )

        converted += 1

        print(
            f"             [OK] "
            f"{output_file.name}"
        )

    except Exception as exc:

        print(
            f"             [FAIL] {exc}"
        )

        failed.append(
            {
                "pdf": pdf_file.name,
                "error": str(exc),
            }
        )


print()
print("=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)

print(f"PDF files:  {len(pdf_files)}")
print(f"Converted:  {converted}")
print(f"Failed:     {len(failed)}")
