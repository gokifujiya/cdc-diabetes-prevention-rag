from pathlib import Path
import html
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "pcd_type2_diabetes_fulltext_manifest.csv"
)

SOURCE_DIR = PROJECT_ROOT / "source_documents"

OUTPUT_FILE = PROJECT_ROOT / "download_pdfs.html"

BATCH_SIZE = 5


# ---------------------------------------------------------
# Load manifest
# ---------------------------------------------------------

df = pd.read_csv(
    MANIFEST_FILE,
    dtype=str,
    encoding="utf-8-sig",
)


# ---------------------------------------------------------
# Detect PDFs already downloaded
# ---------------------------------------------------------

existing_ids = set()

for pdf_file in SOURCE_DIR.glob("*.pdf"):

    match = re.search(
        r"cdc_(\d+)(?:_DS1)?\.pdf$",
        pdf_file.name,
        re.IGNORECASE,
    )

    if match:
        existing_ids.add(match.group(1))


print(f"Existing PDFs: {len(existing_ids)}")


# ---------------------------------------------------------
# Keep only missing records
# ---------------------------------------------------------

records = []

for _, row in df.iterrows():

    cdc_id = str(row["cdc_id"]).strip()

    if cdc_id in existing_ids:
        continue

    title = str(row.get("title", "")).strip()
    pdf_url = str(row["pdf_url"]).strip()

    records.append(
        {
            "cdc_id": cdc_id,
            "title": title,
            "url": pdf_url,
            "filename": f"cdc_{cdc_id}_DS1.pdf",
        }
    )


print(f"Missing PDFs: {len(records)}")


# ---------------------------------------------------------
# Create batches
# ---------------------------------------------------------

batches = []

for start in range(0, len(records), BATCH_SIZE):

    batch = records[
        start:start + BATCH_SIZE
    ]

    batches.append(batch)


# ---------------------------------------------------------
# Create batch buttons
# ---------------------------------------------------------

batch_buttons = []

for index, batch in enumerate(batches, start=1):

    first_number = (
        (index - 1) * BATCH_SIZE + 1
    )

    last_number = (
        first_number + len(batch) - 1
    )

    batch_buttons.append(
        f"""
        <button
            onclick="downloadBatch({index - 1}, this)"
        >
            Batch {index}: PDFs {first_number}–{last_number}
        </button>
        """
    )


# ---------------------------------------------------------
# Create table
# ---------------------------------------------------------

rows = []

for number, record in enumerate(records, start=1):

    rows.append(
        f"""
        <tr>
            <td>{number}</td>
            <td>{html.escape(record["cdc_id"])}</td>
            <td>{html.escape(record["title"])}</td>
            <td>
                <a
                    href="{html.escape(record["url"])}"
                >
                    Download PDF
                </a>
            </td>
        </tr>
        """
    )


batches_json = json.dumps(
    batches,
    ensure_ascii=False,
)


# ---------------------------------------------------------
# HTML
# ---------------------------------------------------------

page = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<title>
CDC Diabetes RAG PDF Downloads
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.5;
}}

.summary {{
    padding: 15px;
    background: #f5f5f5;
    border: 1px solid #ddd;
    margin-bottom: 25px;
}}

.batch-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 20px 0 30px 0;
}}

button {{
    padding: 10px 16px;
    cursor: pointer;
    font-size: 14px;
}}

button.completed {{
    opacity: 0.5;
}}

#status {{
    margin: 15px 0;
    font-weight: bold;
}}

table {{
    border-collapse: collapse;
    width: 100%;
}}

th,
td {{
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
    vertical-align: top;
}}

th {{
    background: #f2f2f2;
}}

</style>

</head>


<body>

<h1>
CDC Type 2 Diabetes PCD Articles
</h1>


<div class="summary">

    <strong>Total corpus:</strong>
    {len(df)}
    <br>

    <strong>Already downloaded:</strong>
    {len(existing_ids)}
    <br>

    <strong>Still missing:</strong>
    {len(records)}
    <br>

    <strong>Batch size:</strong>
    {BATCH_SIZE}

</div>


<h2>
Download missing PDFs
</h2>

<p>
Click one batch at a time. Wait until the five downloads finish,
then click the next batch.
</p>


<div class="batch-container">

{''.join(batch_buttons)}

</div>


<div id="status">
Ready.
</div>


<table>

<thead>

<tr>
    <th>#</th>
    <th>CDC ID</th>
    <th>Title</th>
    <th>PDF</th>
</tr>

</thead>

<tbody>

{''.join(rows)}

</tbody>

</table>


<script>

const batches = {batches_json};


function sleep(ms) {{

    return new Promise(
        resolve => setTimeout(
            resolve,
            ms
        )
    );

}}


async function downloadBatch(
    batchIndex,
    button
) {{

    const batch =
        batches[batchIndex];

    const status =
        document.getElementById(
            "status"
        );


    button.disabled = true;

    status.textContent =
        `Downloading batch ${{batchIndex + 1}}...`;


    for (
        let i = 0;
        i < batch.length;
        i++
    ) {{

        const record =
            batch[i];


        status.textContent =
            `Batch ${{batchIndex + 1}}: ` +
            `${{i + 1}} / ${{batch.length}} — ` +
            record.filename;


        const link =
            document.createElement(
                "a"
            );

        link.href =
            record.url;

        link.download =
            record.filename;

        document.body.appendChild(
            link
        );

        link.click();

        document.body.removeChild(
            link
        );


        await sleep(3000);

    }}


    status.textContent =
        `Batch ${{batchIndex + 1}} finished. ` +
        `Check Downloads before continuing.`;

    button.classList.add(
        "completed"
    );

}}

</script>


</body>

</html>
"""


OUTPUT_FILE.write_text(
    page,
    encoding="utf-8",
)


print()
print(f"Total corpus records: {len(df)}")
print(f"Already downloaded:  {len(existing_ids)}")
print(f"Still missing:        {len(records)}")
print(f"Download batches:     {len(batches)}")
print()
print(f"Saved to: {OUTPUT_FILE}")
