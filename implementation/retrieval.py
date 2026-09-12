from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"


def load_chunks(
    chunks_file: Path = CHUNKS_FILE,
) -> list[dict[str, Any]]:
    """
    Load CDC article chunks from JSONL.
    """

    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_file}"
        )

    chunks: list[dict[str, Any]] = []

    with chunks_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            chunks.append(chunk)

    return chunks


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25 retrieval.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


class BM25Retriever:
    """
    BM25 keyword retriever for CDC article chunks.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:

        self.chunks = chunks

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.index = BM25Okapi(
            self.tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:

        query_tokens = tokenize(query)

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):

            chunk = self.chunks[index].copy()

            chunk["bm25_score"] = float(
                scores[index]
            )

            chunk["rank"] = rank

            results.append(chunk)

        return results


def print_results(
    results: list[dict[str, Any]],
) -> None:
    """
    Print retrieval results for testing.
    """

    for result in results:

        print()
        print("=" * 80)

        print(
            f"Rank: {result['rank']}"
        )

        print(
            f"CDC ID: {result.get('cdc_id', '')}"
        )

        print(
            f"Title: {result.get('title', '')}"
        )

        print(
            f"Page: {result.get('page', '')}"
        )

        print(
            f"BM25 score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Source: "
            f"{result.get('landing_page', '')}"
        )

        print()

        text = result.get(
            "text",
            "",
        )

        print(
            text[:700]
        )


def main() -> None:

    print(
        f"Loading chunks from: "
        f"{CHUNKS_FILE}"
    )

    chunks = load_chunks()

    print(
        f"Chunks loaded: "
        f"{len(chunks)}"
    )

    retriever = BM25Retriever(
        chunks
    )

    query = (
        "How can lifestyle interventions "
        "prevent type 2 diabetes?"
    )

    print()
    print(
        f"Query: {query}"
    )

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print_results(
        results
    )


if __name__ == "__main__":
    main()
