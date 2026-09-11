from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "pcd_type2_diabetes_screened.csv"
)

OUTPUT_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"


# ---------------------------------------------------------
# Chunk settings
# ---------------------------------------------------------

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

metadata_df = pd.read_csv(
    METADATA_FILE,
    dtype=str,
    encoding="utf-8-sig",
)

metadata_df = metadata_df.fillna("")


metadata_by_id = {}

for _, row in metadata_df.iterrows():

    cdc_id = str(
        row.get("cdc_id", "")
    ).strip()

    if not cdc_id:
        continue

    metadata_by_id[cdc_id] = row.to_dict()


print(
    f"Metadata records loaded: "
    f"{len(metadata_by_id)}"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def extract_cdc_id(filename):

    match = re.search(
        r"cdc_(\d+)",
        filename,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return ""


def extract_pages(markdown_text):

    pattern = re.compile(
        r"## Page (\d+)\s*\n(.*?)(?=\n## Page \d+\s*\n|\Z)",
        re.DOTALL,
    )

    pages = []

    for match in pattern.finditer(
        markdown_text
    ):

        page_number = int(
            match.group(1)
        )

        text = (
            match.group(2)
            .strip()
        )

        pages.append(
            (
                page_number,
                text,
            )
        )

    return pages


def chunk_text(text):

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = (
            end - CHUNK_OVERLAP
        )

    return chunks


# ---------------------------------------------------------
# Process Markdown files
# ---------------------------------------------------------

markdown_files = sorted(
    MARKDOWN_DIR.glob("*.md")
)

print(
    f"Markdown files found: "
    f"{len(markdown_files)}"
)

print()


all_chunks = []

documents_processed = 0


for document_index, md_file in enumerate(
    markdown_files,
    start=1,
):

    cdc_id = extract_cdc_id(
        md_file.name
    )

    metadata = metadata_by_id.get(
        cdc_id,
        {}
    )

    markdown_text = md_file.read_text(
        encoding="utf-8",
    )

    pages = extract_pages(
        markdown_text
    )


    print(
        f"[{document_index}/"
        f"{len(markdown_files)}] "
        f"cdc_{cdc_id}: "
        f"{len(pages)} pages"
    )


    for page_number, page_text in pages:

        page_chunks = chunk_text(
            page_text
        )

        for chunk_number, text in enumerate(
            page_chunks,
            start=1,
        ):

            chunk_id = (
                f"cdc_{cdc_id}"
                f"_p{page_number}"
                f"_c{chunk_number}"
            )


            record = {
                "chunk_id": chunk_id,
                "cdc_id": cdc_id,
                "title": metadata.get(
                    "title",
                    metadata.get(
                        "metadata_title",
                        "",
                    ),
                ),
                "year": metadata.get(
                    "year",
                    metadata.get(
                        "mods.sm_dateissued",
                        "",
                    ),
                ),
                "page": page_number,
                "chunk_number": chunk_number,
                "text": text,
                "source_pdf": (
                    f"cdc_{cdc_id}_DS1.pdf"
                ),
                "source_md": md_file.name,
                "landing_page": metadata.get(
                    "landing_page",
                    f"https://stacks.cdc.gov/view/cdc/{cdc_id}",
                ),
                "screening_status": metadata.get(
                    "screening_status",
                    "",
                ),
                "prevention_score": metadata.get(
                    "prevention_score",
                    "",
                ),
                "treatment_score": metadata.get(
                    "treatment_score",
                    "",
                ),
                "screening_reason": metadata.get(
                    "screening_reason",
                    "",
                ),
            }


            all_chunks.append(
                record
            )


    documents_processed += 1


# ---------------------------------------------------------
# Save JSONL
# ---------------------------------------------------------

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    for record in all_chunks:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        file.write("\n")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 60)
print("CHUNKING SUMMARY")
print("=" * 60)

print(
    f"Documents processed: "
    f"{documents_processed}"
)

print(
    f"Chunks created:      "
    f"{len(all_chunks)}"
)

print(
    f"Saved to:            "
    f"{OUTPUT_FILE}"
)
