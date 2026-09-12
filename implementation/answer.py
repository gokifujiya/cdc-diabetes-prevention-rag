from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from implementation.embeddings import load_chunks
from implementation.hybrid_retrieval import HybridRetriever
from implementation.prompts import SYSTEM_PROMPT, build_user_prompt
from implementation.reranker import Reranker


DEFAULT_MODEL = "gpt-5.6-luna"


class CDCAnswerPipeline:
    """
    Complete CDC diabetes prevention RAG pipeline.

    Question
        -> hybrid retrieval
        -> reranking
        -> evidence prompt
        -> OpenAI model
        -> grounded answer
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ) -> None:

        load_dotenv()

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = model

        print("Loading CDC chunks...")

        self.chunks = load_chunks()

        print(
            f"Chunks loaded: "
            f"{len(self.chunks)}"
        )

        print(
            "Initializing hybrid retriever..."
        )

        self.retriever = HybridRetriever(
            self.chunks
        )

        print(
            "Initializing reranker..."
        )

        self.reranker = Reranker()

    def retrieve(
        self,
        question: str,
        hybrid_top_k: int = 20,
        final_top_k: int = 5,
    ) -> list[dict]:

        candidates = (
            self.retriever.search(
                query=question,
                top_k=hybrid_top_k,
                candidate_k=40,
            )
        )

        passages = (
            self.reranker.rerank(
                query=question,
                candidates=candidates,
                top_k=final_top_k,
            )
        )

        return passages

    def answer(
        self,
        question: str,
    ) -> dict:

        passages = self.retrieve(
            question
        )

        user_prompt = build_user_prompt(
            question=question,
            passages=passages,
        )

        response = (
            self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
            )
        )

        return {
            "question": question,
            "answer": response.output_text,
            "sources": passages,
        }


def print_sources(
    sources: list[dict],
) -> None:

    print()
    print("=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    for index, source in enumerate(
        sources,
        start=1,
    ):

        print()

        print(
            f"[Source {index}]"
        )

        print(
            f"Title: "
            f"{source.get('title', '')}"
        )

        print(
            f"CDC ID: "
            f"{source.get('cdc_id', '')}"
        )

        print(
            f"Page: "
            f"{source.get('page', '')}"
        )

        print(
            f"URL: "
            f"{source.get('landing_page', '')}"
        )

        print(
            f"Reranker score: "
            f"{source.get('reranker_score', 0):.6f}"
        )


def main() -> None:

    pipeline = CDCAnswerPipeline()

    question = (
        "How can lifestyle interventions "
        "prevent type 2 diabetes?"
    )

    print()
    print(
        f"Question: {question}"
    )

    print()
    print(
        "Generating evidence-grounded answer..."
    )

    result = pipeline.answer(
        question
    )

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print()

    print(
        result["answer"]
    )

    print_sources(
        result["sources"]
    )


if __name__ == "__main__":
    main()
