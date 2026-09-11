import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_results.html"
OUTPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_candidates.csv"


html = INPUT_FILE.read_text(
    encoding="utf-8",
    errors="ignore",
)

pattern = re.compile(
    r'<a[^>]+href="https://stacks\.cdc\.gov/view/cdc/(\d+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

records = []

for cdc_id, title_html in pattern.findall(html):

    # Remove any HTML tags inside the title
    title = re.sub(r"<.*?>", "", title_html)

    # Clean whitespace
    title = " ".join(title.split())

    records.append(
        {
            "cdc_id": cdc_id,
            "title": title,
            "landing_page": f"https://stacks.cdc.gov/view/cdc/{cdc_id}",
        }
    )


# Remove duplicates while preserving order
unique_records = {}

for record in records:
    if record["cdc_id"] not in unique_records:
        unique_records[record["cdc_id"]] = record

records = list(unique_records.values())


with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "cdc_id",
            "title",
            "landing_page",
        ],
    )

    writer.writeheader()
    writer.writerows(records)


print(f"Found {len(records)} CDC records.")
print(f"Saved to: {OUTPUT_FILE}")
