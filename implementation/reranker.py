from __future__ import annotations

from typing import Any

import torch
from sentence_transformers import CrossEncoder

from implementation.hybrid_retrieval import HybridRetriever
from implementation.embeddings import load_chunks


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker:
    """
    Cross-encoder reranker for hybrid retrieval candidates.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(
            model_name,
            activation_fn=torch.nn.Sigmoid(),
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        pairs = [
            (
                query,
                candidate["text"],
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            result = candidate.copy()

            result["reranker_score"] = float(
                score
            )

            reranked.append(result)

        reranked.sort(
            key=lambda item: item[
                "reranker_score"
            ],
            reverse=True,
        )

        reranked = reranked[:top_k]

        for rank, result in enumerate(
            reranked,
            start=1,
        ):
            result["rank"] = rank

        return reranked


def print_results(
    results: list[dict[str, Any]],
) -> None:

    for result in results:

        print()
        print("=" * 80)

        print(
            f"Final rank: {result['rank']}"
        )

        print(
            f"CDC ID: "
            f"{result.get('cdc_id', '')}"
        )

        print(
            f"Title: "
            f"{result.get('title', '')}"
        )

        print(
            f"Page: "
            f"{result.get('page', '')}"
        )

        print(
            f"BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Dense rank: "
            f"{result.get('dense_rank')}"
        )

        print(
            f"RRF score: "
            f"{result.get('rrf_score', 0.0):.6f}"
        )

        print(
            f"Reranker score: "
            f"{result['reranker_score']:.6f}"
        )

        print(
            f"Source: "
            f"{result.get('landing_page', '')}"
        )

        print()

        print(
            result.get(
                "text",
                "",
            )[:700]
        )


def main() -> None:

    chunks = load_chunks()

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    hybrid_retriever = HybridRetriever(
        chunks
    )

    reranker = Reranker()

    query = (
        "How can lifestyle interventions "
        "prevent type 2 diabetes?"
    )

    print()
    print(
        f"Query: {query}"
    )

    candidates = hybrid_retriever.search(
        query=query,
        top_k=20,
        candidate_k=40,
    )

    print(
        f"Hybrid candidates: "
        f"{len(candidates)}"
    )

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=5,
    )

    print_results(
        results
    )


if __name__ == "__main__":
    main()
