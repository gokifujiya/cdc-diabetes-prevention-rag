import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

CANDIDATES_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_candidates.csv"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_metadata.csv"


# Load the 74 selected Type 2 Diabetes records
candidates = pd.read_csv(
    CANDIDATES_FILE,
    dtype=str,
    encoding="utf-8-sig",
)

# Load the complete PCD metadata export
metadata = pd.read_csv(
    METADATA_FILE,
    dtype=str,
    low_memory=False,
)


# CDC metadata uses IDs such as "cdc:150969"
candidates["PID"] = "cdc:" + candidates["cdc_id"].astype(str)


# Select useful metadata fields that actually exist
wanted_columns = [
    "PID",
    "mods.title",
    "mods.name_personal",
    "mods.sm_dateissued",
    "mods.raw_date",
    "mods.abstract",
    "mods.subject_topic",
    "keywords",
    "mods.journal_title",
    "mods.report_type",
    "mods.sm_publisher",
    "mods.publisher_place",
    "mods.sm_rights",
    "mods.sm_rights_statement",
    "mods.sm_downloadurl",
    "dc.language",
]

available_columns = [
    column
    for column in wanted_columns
    if column in metadata.columns
]

metadata_subset = metadata[available_columns].copy()


# Match the 74 candidates against the full PCD collection
merged = candidates.merge(
    metadata_subset,
    on="PID",
    how="left",
)


# Rename columns into simpler names
rename_map = {
    "mods.title": "metadata_title",
    "mods.name_personal": "authors",
    "mods.sm_dateissued": "publication_date",
    "mods.raw_date": "raw_date",
    "mods.abstract": "abstract",
    "mods.subject_topic": "subjects",
    "keywords": "keywords",
    "mods.journal_title": "journal",
    "mods.report_type": "document_type",
    "mods.sm_publisher": "publisher",
    "mods.publisher_place": "publisher_place",
    "mods.sm_rights": "rights",
    "mods.sm_rights_statement": "rights_statement",
    "mods.sm_downloadurl": "download_url",
    "dc.language": "language",
}

merged = merged.rename(columns=rename_map)


# Remove the temporary PID if you do not need it externally
# We keep it because it is useful for traceability.

# Save clean metadata file
merged.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_MINIMAL,
)


matched = merged["metadata_title"].notna().sum()

print(f"Candidate records: {len(candidates)}")
print(f"Matched metadata records: {matched}")
print(f"Unmatched records: {len(candidates) - matched}")
print(f"Saved to: {OUTPUT_FILE}")
