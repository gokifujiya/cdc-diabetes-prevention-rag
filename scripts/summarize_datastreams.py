from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "pcd_type2_diabetes_fulltext_sources.csv"
)


df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    encoding="utf-8-sig",
)


# Find DS numbers such as DS1, DS2, DS3...
ds_numbers = sorted(
    {
        re.match(r"(DS\d+)\.", column).group(1)
        for column in df.columns
        if re.match(r"DS\d+\.", column)
    },
    key=lambda x: int(x[2:]),
)


for ds in ds_numbers:

    mime_col = f"{ds}.mimetype_txt_en"
    source_col = f"{ds}.sourceurl"
    size_col = f"{ds}.filesize_tl"

    print()
    print("=" * 70)
    print(ds)
    print("=" * 70)

    # MIME types
    if mime_col in df.columns:
        values = (
            df[mime_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        values = values[values != ""]

        print(f"MIME non-empty: {len(values)}/74")

        if len(values):
            print("MIME types:")
            for value, count in values.value_counts().items():
                print(f"  {value}: {count}")

    # Source URLs
    if source_col in df.columns:
        urls = (
            df[source_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        urls = urls[urls != ""]

        print(f"Source URLs: {len(urls)}/74")

        if len(urls):
            print("Example URLs:")
            for url in urls.head(3):
                print(f"  {url}")

    # File sizes
    if size_col in df.columns:
        sizes = (
            df[size_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        sizes = sizes[sizes != ""]

        print(f"File sizes present: {len(sizes)}/74")
        