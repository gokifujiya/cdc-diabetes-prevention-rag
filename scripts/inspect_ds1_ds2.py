from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

METADATA_FILE = PROJECT_ROOT / "data" / "metadata.csv"
CANDIDATES_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_candidates.csv"


metadata = pd.read_csv(
    METADATA_FILE,
    dtype=str,
    low_memory=False,
)

candidates = pd.read_csv(
    CANDIDATES_FILE,
    dtype=str,
    encoding="utf-8-sig",
)

candidates["PID"] = "cdc:" + candidates["cdc_id"].astype(str)

subset = metadata[
    metadata["PID"].isin(candidates["PID"])
].copy()


ds_columns = [
    column
    for column in metadata.columns
    if column.startswith("DS1.") or column.startswith("DS2.")
]


print("DS1 / DS2 columns:")
print()

for column in ds_columns:
    print(column)


print()
print("=" * 80)
print("SAMPLE RECORD")
print("=" * 80)

sample = subset.iloc[0]

print(f"PID: {sample['PID']}")
print()

for column in ds_columns:
    value = sample.get(column)

    if pd.notna(value) and str(value).strip():
        print(f"{column}: {value}")
