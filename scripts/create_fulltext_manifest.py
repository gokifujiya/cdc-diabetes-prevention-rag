from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_screened.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_fulltext_manifest.csv"


df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    encoding="utf-8-sig",
)


df["pdf_url"] = df["cdc_id"].apply(
    lambda cdc_id:
    f"https://stacks.cdc.gov/view/cdc/{cdc_id}/cdc_{cdc_id}_DS1.pdf"
)

df["local_pdf"] = df["cdc_id"].apply(
    lambda cdc_id:
    f"source_documents/cdc_{cdc_id}.pdf"
)


manifest_columns = [
    "cdc_id",
    "PID",
    "title",
    "authors",
    "publication_date",
    "landing_page",
    "pdf_url",
    "local_pdf",
    "screening_status",
    "prevention_score",
    "treatment_score",
]

manifest_columns = [
    column
    for column in manifest_columns
    if column in df.columns
]


manifest = df[manifest_columns].copy()

manifest.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print(f"Records: {len(manifest)}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("First 5 PDF URLs:")

for url in manifest["pdf_url"].head():
    print(url)
