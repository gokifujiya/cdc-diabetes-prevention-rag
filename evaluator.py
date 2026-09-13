import json
from pathlib import Path

from implementation.embeddings import DenseRetriever, load_chunks
from implementation.hybrid_retrieval import HybridRetriever
from implementation.reranker import Reranker
from implementation.retrieval import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parent

QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.json"

TOP_K = 10
HYBRID_CANDIDATE_K = 40
RERANK_CANDIDATE_K = 20
RERANK_TOP_K = 10


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "questions" in data:
        return data["questions"]

    raise ValueError(
        "questions.json must contain either a list of questions "
        "or an object with a 'questions' list."
    )


def get_question_id(item, index):
    return (
        item.get("id")
        or item.get("question_id")
        or f"Q{index:02d}"
    )


def get_question_text(item):
    return (
        item.get("question")
        or item.get("query")
        or item.get("text")
    )


def get_category(item):
    return item.get("category", "")


def simplify_result(result):
    return {
        "chunk_id": result.get("chunk_id"),
        "cdc_id": result.get("cdc_id"),
        "title": result.get("title"),
        "page": result.get("page"),
        "text": result.get("text"),
        "bm25_score": result.get("bm25_score"),
        "dense_score": result.get("dense_score"),
        "rrf_score": result.get("rrf_score"),
        "reranker_score": result.get("reranker_score"),
        "bm25_rank": result.get("bm25_rank"),
        "dense_rank": result.get("dense_rank"),
        "rank": result.get("rank"),
        "landing_page": result.get("landing_page"),
    }


def add_ranks(results):
    ranked = []

    for rank, result in enumerate(results, start=1):
        item = dict(result)
        item["rank"] = rank
        ranked.append(item)

    return ranked


def evaluate():
    print("Loading questions...")
    questions = load_questions()

    print(f"Questions loaded: {len(questions)}")

    print("\nLoading CDC chunks...")
    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks)}")

    print("\nInitializing BM25 retriever...")
    bm25 = BM25Retriever(chunks)

    print("\nInitializing dense retriever...")
    dense = DenseRetriever(chunks)

    print("\nInitializing hybrid retriever...")
    hybrid = HybridRetriever(chunks)

    print("\nInitializing reranker...")
    reranker = Reranker()

    all_results = []

    for index, item in enumerate(questions, start=1):
        question_id = get_question_id(item, index)
        question = get_question_text(item)
        category = get_category(item)

        if not question:
            print(f"\nSkipping {question_id}: no question text.")
            continue

        print("\n" + "=" * 80)
        print(f"{question_id}: {question}")
        print("=" * 80)

        # ---------------------------------------------------------
        # 1. BM25
        # ---------------------------------------------------------
        print("Running BM25...")

        bm25_results = bm25.search(
            question,
            top_k=TOP_K,
        )

        bm25_results = add_ranks(bm25_results)

        # ---------------------------------------------------------
        # 2. Dense retrieval
        # ---------------------------------------------------------
        print("Running dense retrieval...")

        dense_results = dense.search(
            question,
            top_k=TOP_K,
        )

        dense_results = add_ranks(dense_results)

        # ---------------------------------------------------------
        # 3. Hybrid retrieval
        # ---------------------------------------------------------
        print("Running hybrid retrieval...")

        hybrid_results = hybrid.search(
            question,
            top_k=TOP_K,
            candidate_k=HYBRID_CANDIDATE_K,
        )

        hybrid_results = add_ranks(hybrid_results)

        # ---------------------------------------------------------
        # 4. Hybrid + CrossEncoder reranking
        # ---------------------------------------------------------
        print("Running CrossEncoder reranking...")

        rerank_candidates = hybrid.search(
            question,
            top_k = RERANK_CANDIDATE_K,
            candidate_k = HYBRID_CANDIDATE_K,
        )

        reranked_results = reranker.rerank(
            question,
            rerank_candidates,
            top_k = RERANK_TOP_K,
        )

        reranked_results = add_ranks(reranked_results)

        question_result = {
            "id": question_id,
            "category": category,
            "question": question,
            "retrieval": {
                "bm25": [
                    simplify_result(result)
                    for result in bm25_results
                ],
                "dense": [
                    simplify_result(result)
                    for result in dense_results
                ],
                "hybrid": [
                    simplify_result(result)
                    for result in hybrid_results
                ],
                "hybrid_reranked": [
                    simplify_result(result)
                    for result in reranked_results
                ],
            },
        }

        all_results.append(question_result)

        print("\nTop reranked result:")

        if reranked_results:
            top = reranked_results[0]

            print(
                f"CDC {top.get('cdc_id')} | "
                f"Page {top.get('page')} | "
                f"{top.get('title')}"
            )

    output = {
        "evaluation": {
            "question_count": len(all_results),
            "chunk_count": len(chunks),
            "top_k": TOP_K,
            "hybrid_candidate_k": HYBRID_CANDIDATE_K,
            "rerank_candidate_k": RERANK_CANDIDATE_K,
            "rerank_top_k": RERANK_TOP_K,
            "methods": [
                "bm25",
                "dense",
                "hybrid",
                "hybrid_reranked",
            ],
        },
        "results": all_results,
    }

    RESULTS_FILE.parent.mkdir(
        parents = True,
        exist_ok = True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding = "utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent = 2,
            ensure_ascii = False,
        )

    print("\n" + "=" * 80)
    print("Evaluation retrieval run complete.")
    print(f"Questions processed: {len(all_results)}")
    print(f"Results saved to:")
    print(RESULTS_FILE)
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
