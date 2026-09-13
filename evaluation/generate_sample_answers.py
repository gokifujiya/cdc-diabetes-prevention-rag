import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the project root when this script is run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation.answer import CDCAnswerPipeline

QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "generated_answers.json"


def load_questions():
    with open(QUESTIONS_FILE, "r", encoding = "utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        return data["questions"]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported questions.json format")


def get_id(item, index):
    return item.get("id") or item.get("question_id") or f"Q{index:02d}"


def get_text(item):
    return item.get("question") or item.get("query") or item.get("text")


def main():
    questions = load_questions()

    if len(questions) != 20:
        raise ValueError(f"Expected 20 benchmark questions, found {len(questions)}")

    for item in questions:
        if not get_text(item):
            raise ValueError(f"Question text is missing for {item}")

    selected = []

    for index, item in enumerate(questions, start = 1):
        qid = get_id(item, index)

        selected.append({
            "id": qid,
            "category": item.get("category", ""),
            "question": get_text(item),
        })

    print(f"Selected questions: {len(selected)}")

    pipeline = CDCAnswerPipeline()

    results = []

    for item in selected:
        print("\n" + "=" * 80)
        print(f"{item['id']}: {item['question']}")
        print("=" * 80)

        result = pipeline.answer(item["question"])

        record = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": result["answer"],
            "sources": result["sources"],
        }

        results.append(record)

        print(result["answer"])

    with open(OUTPUT_FILE, "w", encoding = "utf-8") as f:
        json.dump(
            results,
            f,
            indent = 2,
            ensure_ascii = False,
        )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
