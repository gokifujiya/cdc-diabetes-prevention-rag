from pathlib import Path
import re
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

RESULTS_HTML = PROJECT_ROOT / "data" / "pcd_type2_diabetes_results.html"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CANDIDATES_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_candidates.csv"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "public_domain_manifest.csv"


# ---------------------------------------------------------
# Load saved CDC search-results HTML
# ---------------------------------------------------------

html_text = RESULTS_HTML.read_text(
    encoding="utf-8",
    errors="ignore",
)


# ---------------------------------------------------------
# Find every CDC record link in document order
# ---------------------------------------------------------

record_pattern = re.compile(
    r'href=["\']https://stacks\.cdc\.gov/view/cdc/(\d+)["\']',
    re.IGNORECASE,
)

matches = list(record_pattern.finditer(html_text))

print(f"CDC records detected in HTML: {len(matches)}")


# ---------------------------------------------------------
# Determine Public Domain status per record
#
# Each record is examined only from its own CDC link
# up to the beginning of the next CDC record.
# ---------------------------------------------------------

public_domain_ids = set()

for index, match in enumerate(matches):

    cdc_id = match.group(1)

    start = match.start()

    if index + 1 < len(matches):
        end = matches[index + 1].start()
    else:
        end = len(html_text)

    record_html = html_text[start:end]

    if re.search(
        r"Public\s+Domain",
        record_html,
        re.IGNORECASE,
    ):
        public_domain_ids.add(cdc_id)


print(
    f"Public Domain IDs detected: "
    f"{len(public_domain_ids)}"
)


# ---------------------------------------------------------
# Load candidate metadata
# ---------------------------------------------------------

candidates_df = pd.read_csv(
    CANDIDATES_FILE,
    dtype=str,
    encoding="utf-8-sig",
).fillna("")


# ---------------------------------------------------------
# Rebuild knowledge_base from scratch
# ---------------------------------------------------------

if KNOWLEDGE_BASE_DIR.exists():
    shutil.rmtree(KNOWLEDGE_BASE_DIR)

KNOWLEDGE_BASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Copy only Public Domain Markdown files
# ---------------------------------------------------------

manifest_rows = []

copied = 0
missing = 0


for _, row in candidates_df.iterrows():

    cdc_id = str(
        row["cdc_id"]
    ).strip()

    title = str(
        row.get("title", "")
    ).strip()

    landing_page = str(
        row.get("landing_page", "")
    ).strip()

    is_public_domain = cdc_id in public_domain_ids

    source_md = (
        MARKDOWN_DIR
        / f"cdc_{cdc_id}_DS1.md"
    )

    destination_md = (
        KNOWLEDGE_BASE_DIR
        / f"cdc_{cdc_id}.md"
    )

    status = "excluded_not_public_domain"

    if is_public_domain:

        if source_md.exists():

            shutil.copy2(
                source_md,
                destination_md,
            )

            copied += 1
            status = "copied"

        else:

            missing += 1
            status = "missing_markdown"


    manifest_rows.append(
        {
            "cdc_id": cdc_id,
            "title": title,
            "public_domain": is_public_domain,
            "source_md": source_md.name,
            "knowledge_base_file": (
                destination_md.name
                if is_public_domain
                else ""
            ),
            "landing_page": landing_page,
            "status": status,
        }
    )


# ---------------------------------------------------------
# Save manifest for all 74 records
# ---------------------------------------------------------

manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    OUTPUT_MANIFEST,
    index=False,
    encoding="utf-8-sig",
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 60)
print("PUBLIC KNOWLEDGE BASE SUMMARY")
print("=" * 60)

print(
    f"Total candidate records: "
    f"{len(candidates_df)}"
)

print(
    f"Public Domain records: "
    f"{len(public_domain_ids)}"
)

print(
    f"Markdown files copied: "
    f"{copied}"
)

print(
    f"Excluded records: "
    f"{len(candidates_df) - len(public_domain_ids)}"
)

print(
    f"Missing Markdown files: "
    f"{missing}"
)

print(
    f"Knowledge base folder: "
    f"{KNOWLEDGE_BASE_DIR}"
)

print(
    f"Manifest saved to: "
    f"{OUTPUT_MANIFEST}"
)
