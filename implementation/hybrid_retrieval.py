from __future__ import annotations

from typing import Any

from implementation.embeddings import DenseRetriever, load_chunks
from implementation.retrieval import BM25Retriever


RRF_K = 60


class HybridRetriever:
    """
    Combine BM25 and dense semantic retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        self.chunks = chunks

        print("Initializing BM25 retriever...")
        self.bm25 = BM25Retriever(chunks)

        print("Initializing dense retriever...")
        self.dense = DenseRetriever(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 30,
    ) -> list[dict[str, Any]]:

        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

        dense_results = self.dense.search(
            query=query,
            top_k=candidate_k,
        )

        fused: dict[str, dict[str, Any]] = {}

        for result in bm25_results:
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = result.copy()
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["bm25_rank"] = None
                fused[chunk_id]["dense_rank"] = None

            rank = result["rank"]

            fused[chunk_id]["bm25_rank"] = rank

            fused[chunk_id]["rrf_score"] += (
                1.0 / (RRF_K + rank)
            )

        for result in dense_results:
            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = result.copy()
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["bm25_rank"] = None
                fused[chunk_id]["dense_rank"] = None

            rank = result["rank"]

            fused[chunk_id]["dense_rank"] = rank

            fused[chunk_id]["rrf_score"] += (
                1.0 / (RRF_K + rank)
            )

            if "dense_score" in result:
                fused[chunk_id]["dense_score"] = (
                    result["dense_score"]
                )

        results = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )[:top_k]

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result["rank"] = rank

        return results


def print_results(
    results: list[dict[str, Any]],
) -> None:

    for result in results:
        print()
        print("=" * 80)

        print(
            f"Hybrid rank: {result['rank']}"
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
            f"BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Dense rank: "
            f"{result.get('dense_rank')}"
        )

        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
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

    chunks = load_chunks()

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    retriever = HybridRetriever(
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
        candidate_k=30,
    )

    print_results(results)


if __name__ == "__main__":
    main()
