from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.jsonl"

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "cdc_embeddings.pt"
CHECKPOINT_FILE = EMBEDDINGS_DIR / "cdc_embeddings_checkpoint.pt"

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

BATCH_SIZE = 32
CHECKPOINT_EVERY = 10


def load_chunks(
    chunks_file: Path = CHUNKS_FILE,
) -> list[dict[str, Any]]:
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


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_name: str = MODEL_NAME,
    ) -> None:
        self.chunks = chunks

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name
        )

        EMBEDDINGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.document_embeddings = (
            self._load_or_create_embeddings()
        )

    def _load_or_create_embeddings(
        self,
    ) -> torch.Tensor:

        if EMBEDDINGS_FILE.exists():
            print("Loading cached embeddings:")
            print(EMBEDDINGS_FILE)

            embeddings = torch.load(
                EMBEDDINGS_FILE,
                map_location="cpu",
                weights_only=True,
            )

            if len(embeddings) != len(self.chunks):
                raise ValueError(
                    "Cached embeddings do not match "
                    "the current number of chunks."
                )

            print(
                f"Cached embeddings loaded: "
                f"{len(embeddings)}"
            )

            return embeddings

        documents = [
            chunk["text"]
            for chunk in self.chunks
        ]

        total_documents = len(documents)

        existing_embeddings: list[torch.Tensor] = []
        start_index = 0

        if CHECKPOINT_FILE.exists():
            checkpoint = torch.load(
                CHECKPOINT_FILE,
                map_location="cpu",
                weights_only=False,
            )

            existing_embeddings = checkpoint[
                "embeddings"
            ]

            start_index = checkpoint[
                "next_index"
            ]

            print(
                f"Resuming from checkpoint at "
                f"document {start_index}/{total_documents}"
            )
        else:
            print(
                f"Creating embeddings for "
                f"{total_documents} chunks..."
            )

        batch_counter = 0

        for batch_start in range(
            start_index,
            total_documents,
            BATCH_SIZE,
        ):
            batch_end = min(
                batch_start + BATCH_SIZE,
                total_documents,
            )

            batch_documents = documents[
                batch_start:batch_end
            ]

            batch_embeddings = (
                self.model.encode_document(
                    batch_documents,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            ).cpu()

            existing_embeddings.append(
                batch_embeddings
            )

            batch_counter += 1

            print(
                f"Embedded "
                f"{batch_end}/{total_documents}"
            )

            if (
                batch_counter % CHECKPOINT_EVERY == 0
                or batch_end == total_documents
            ):
                torch.save(
                    {
                        "embeddings": existing_embeddings,
                        "next_index": batch_end,
                    },
                    CHECKPOINT_FILE,
                )

                print(
                    f"Checkpoint saved at "
                    f"{batch_end}/{total_documents}"
                )

        embeddings = torch.cat(
            existing_embeddings,
            dim=0,
        )

        torch.save(
            embeddings,
            EMBEDDINGS_FILE,
        )

        print("Final embeddings saved:")
        print(EMBEDDINGS_FILE)

        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()

        return embeddings

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:

        query_embedding = (
            self.model.encode_query(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        ).cpu()

        hits = semantic_search(
            query_embedding,
            self.document_embeddings,
            top_k=top_k,
        )[0]

        results = []

        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            index = hit["corpus_id"]

            chunk = self.chunks[index].copy()

            chunk["dense_score"] = float(
                hit["score"]
            )

            chunk["rank"] = rank

            results.append(chunk)

        return results


def print_results(
    results: list[dict[str, Any]],
) -> None:
    for result in results:
        print()
        print("=" * 80)

        print(f"Rank: {result['rank']}")
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
            f"Dense score: "
            f"{result['dense_score']:.4f}"
        )
        print(
            f"Source: "
            f"{result.get('landing_page', '')}"
        )

        print()
        print(
            result.get("text", "")[:700]
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

    retriever = DenseRetriever(
        chunks
    )

    query = (
        "How can lifestyle interventions "
        "prevent type 2 diabetes?"
    )

    print()
    print(f"Query: {query}")

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print_results(results)


if __name__ == "__main__":
    main()
