import json
import re
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = PROJECT_ROOT / "evaluation" / "reference_answers.json"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"


def normalize(text):
    if not text:
        return ""

    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Repair PDF line-break hyphenation:
    # "interven-\ntion" -> "intervention"
    text = re.sub(r"-\s+", "", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", normalize(text)))


def token_coverage(reference_text, candidate_text):
    ref = tokens(reference_text)
    cand = tokens(candidate_text)

    if not ref:
        return 0.0

    return len(ref & cand) / len(ref)


with open(REFERENCE_FILE, encoding = "utf-8") as f:
    references = json.load(f)

chunks = []

with open(CHUNKS_FILE, encoding = "utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))


for item in references["items"]:

    source = item["reference_source"]

    reference_title = normalize(source.get("title", ""))
    reference_evidence = item["reference_evidence"]
    reference_page = source.get("page")

    #
    # STEP 1:
    # Find chunks belonging to the reference article by TITLE,
    # not by the existing CDC ID.
    #
    title_candidates = []

    if reference_title:
        for chunk in chunks:
            chunk_title = normalize(chunk.get("title", ""))

            title_score = SequenceMatcher(
                None,
                reference_title,
                chunk_title
            ).ratio()

            if title_score >= 0.90:
                title_candidates.append(chunk)

    #
    # If title matching fails, fall back to the whole corpus.
    #
    candidates = title_candidates if title_candidates else chunks

    best_chunk = None
    best_score = -1

    for chunk in candidates:

        coverage = token_coverage(
            reference_evidence,
            chunk.get("text", "")
        )

        page_bonus = 0.0

        if (
            reference_page is not None
            and chunk.get("page") == reference_page
        ):
            page_bonus = 0.05

        score = coverage + page_bonus

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if not best_chunk:
        print(f'{item["id"]}: NO MATCH')
        continue

    coverage = token_coverage(
        reference_evidence,
        best_chunk.get("text", "")
    )

    #
    # Correct the source metadata from the actual corpus.
    #
    source["cdc_id"] = str(best_chunk.get("cdc_id"))
    source["title"] = best_chunk.get("title")
    source["page"] = best_chunk.get("page")
    source["chunk_id"] = best_chunk.get("chunk_id")

    print(
        f'{item["id"]}: '
        f'{best_chunk.get("chunk_id")} '
        f'coverage={coverage:.3f} | '
        f'{best_chunk.get("title")}'
    )


with open(REFERENCE_FILE, "w", encoding = "utf-8") as f:
    json.dump(
        references,
        f,
        ensure_ascii = False,
        indent = 2
    )

print("\nUpdated:", REFERENCE_FILE)
