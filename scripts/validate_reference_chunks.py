import json
import re
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = PROJECT_ROOT / "evaluation" / "reference_answers.json"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"


def normalize(text):
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text):
    return set(re.findall(r"[a-z0-9]+", normalize(text)))


with open(REFERENCE_FILE, encoding = "utf-8") as f:
    references = json.load(f)

chunks = []

with open(CHUNKS_FILE, encoding = "utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

chunks_by_id = {
    c["chunk_id"]: c
    for c in chunks
}

for item in references["items"]:
    source = item["reference_source"]
    evidence = normalize(item["reference_evidence"])

    chunk_id = source.get("chunk_id")
    chunk = chunks_by_id.get(chunk_id)

    print("=" * 90)
    print(item["id"])
    print("Reference CDC ID :", source.get("cdc_id"))
    print("Reference title  :", source.get("title"))
    print("Reference page   :", source.get("page"))
    print("Mapped chunk     :", chunk_id)

    if not chunk:
        print("STATUS: CHUNK NOT FOUND")
        continue

    chunk_text = normalize(chunk.get("text", ""))

    ref_tokens = token_set(evidence)
    chunk_tokens = token_set(chunk_text)

    token_coverage = (
        len(ref_tokens & chunk_tokens) / len(ref_tokens)
        if ref_tokens else 0
    )

    sequence_score = SequenceMatcher(
        None,
        evidence,
        chunk_text
    ).ratio()

    exact = evidence in chunk_text

    print("Chunk CDC ID     :", chunk.get("cdc_id"))
    print("Chunk title      :", chunk.get("title"))
    print("Chunk page       :", chunk.get("page"))
    print("Exact substring  :", exact)
    print(f"Token coverage   : {token_coverage:.3f}")
    print(f"Sequence score   : {sequence_score:.3f}")

    id_match = (
        str(source.get("cdc_id")) == str(chunk.get("cdc_id"))
        if source.get("cdc_id")
        else None
    )

    page_match = (
        source.get("page") == chunk.get("page")
        if source.get("page") is not None
        else None
    )

    print("CDC ID match     :", id_match)
    print("Page match       :", page_match)

    print("\nEvidence:")
    print(item["reference_evidence"][:350])

    print("\nMapped chunk:")
    print(chunk.get("text", "")[:500])

print("\nValidation complete.")
