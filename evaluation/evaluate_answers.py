import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

GENERATED_FILE = PROJECT_ROOT / "evaluation" / "generated_answers.json"
REFERENCE_FILE = PROJECT_ROOT / "evaluation" / "reference_answers.json"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "answer_evaluation.json"


client = OpenAI()


def load_json(path):
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)


def judge_answer(question, reference_answer, generated_answer):
    prompt = f"""
You are evaluating a medical/public-health RAG answer.

Compare the GENERATED ANSWER with the REFERENCE ANSWER.

Important:
- The reference answer is a benchmark target, not necessarily the only valid wording.
- Do not require exact wording.
- Give credit when the generated answer expresses the same medically relevant meaning.
- Extra information is acceptable if it is relevant and does not contradict the reference.
- Do not penalize the answer merely for being more detailed.

Scoring:

2 = FULLY CORRECT
The generated answer contains all or nearly all important information in the reference answer and does not materially contradict it.

1 = PARTIALLY CORRECT
The generated answer is substantially correct but omits one or more important reference points, or contains a minor substantive problem.

0 = INCORRECT
The generated answer materially contradicts the reference, misses the central answer, or is largely unsupported/off-topic.

Question:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{generated_answer}

Return JSON only in this exact format:

{{
  "score": 0,
  "reason": "short explanation",
  "missing_reference_points": [],
  "contradictions": []
}}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
    )

    text = response.output_text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return json.loads(text.strip())


def main():
    generated = load_json(GENERATED_FILE)
    references = load_json(REFERENCE_FILE)["items"]

    generated_by_id = {
        item["id"]: item
        for item in generated
    }

    results = []

    for ref in references:
        qid = ref["id"]

        if qid not in generated_by_id:
            raise ValueError(f"Missing generated answer for {qid}")

        gen = generated_by_id[qid]

        print("=" * 80)
        print(qid)
        print(ref["question"])

        judgment = judge_answer(
            question = ref["question"],
            reference_answer = ref["reference_answer"],
            generated_answer = gen["answer"],
        )

        record = {
            "id": qid,
            "question": ref["question"],
            "score": judgment["score"],
            "reason": judgment["reason"],
            "missing_reference_points": judgment.get(
                "missing_reference_points", []
            ),
            "contradictions": judgment.get(
                "contradictions", []
            ),
        }

        results.append(record)

        print("Score:", record["score"])
        print("Reason:", record["reason"])

    scores = [x["score"] for x in results]

    full = sum(score == 2 for score in scores)
    partial = sum(score == 1 for score in scores)
    incorrect = sum(score == 0 for score in scores)

    total_points = sum(scores)
    max_points = len(scores) * 2
    percentage = (
        total_points / max_points * 100
        if max_points
        else 0
    )

    output = {
        "evaluation_method": "LLM semantic comparison against reference answers",
        "scoring": {
            "2": "fully correct",
            "1": "partially correct",
            "0": "incorrect",
        },
        "n_questions": len(results),
        "summary": {
            "fully_correct": full,
            "partially_correct": partial,
            "incorrect": incorrect,
            "total_points": total_points,
            "maximum_points": max_points,
            "percentage": round(percentage, 1),
        },
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding = "utf-8") as f:
        json.dump(
            output,
            f,
            indent = 2,
            ensure_ascii = False,
        )

    print("\n" + "=" * 80)
    print("ANSWER EVALUATION COMPLETE")
    print("=" * 80)

    print("Fully correct:", full)
    print("Partially correct:", partial)
    print("Incorrect:", incorrect)
    print(f"Score: {total_points}/{max_points}")
    print(f"Percentage: {percentage:.1f}%")

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
