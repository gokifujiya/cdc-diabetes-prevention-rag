from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

CANDIDATES_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_candidates.csv"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_fulltext_sources.csv"


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

candidates = pd.read_csv(
    CANDIDATES_FILE,
    dtype=str,
    encoding="utf-8-sig",
)

metadata = pd.read_csv(
    METADATA_FILE,
    dtype=str,
    low_memory=False,
)

candidates["PID"] = "cdc:" + candidates["cdc_id"].astype(str)


# ---------------------------------------------------------
# Identify columns that may contain full-text information
# ---------------------------------------------------------

search_terms = [
    "url",
    "download",
    "file",
    "datastream",
    "mime",
    "content",
    "resource",
    "pdf",
    "xml",
    "html",
]

candidate_columns = []

for column in metadata.columns:
    lower = column.lower()

    if any(term in lower for term in search_terms):
        candidate_columns.append(column)


print(f"Potential full-text related columns: {len(candidate_columns)}")
print()

for column in candidate_columns:
    print(column)


# ---------------------------------------------------------
# Restrict metadata to the 74 Type 2 Diabetes records
# ---------------------------------------------------------

subset = metadata[
    metadata["PID"].isin(candidates["PID"])
].copy()


# ---------------------------------------------------------
# Inspect which candidate columns actually contain data
# ---------------------------------------------------------

useful_columns = []

for column in candidate_columns:
    nonempty = (
        subset[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    if nonempty > 0:
        useful_columns.append(column)
        print(f"{column}: {nonempty}/74 non-empty")


# ---------------------------------------------------------
# Save useful full-text-related metadata
# ---------------------------------------------------------

base_columns = [
    "PID",
]

output_columns = base_columns + useful_columns

result = subset[output_columns].copy()

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print()
print(f"Matched records: {len(result)}")
print(f"Useful full-text-related columns: {len(useful_columns)}")
print(f"Saved to: {OUTPUT_FILE}")
