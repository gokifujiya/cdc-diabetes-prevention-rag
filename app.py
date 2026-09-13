from __future__ import annotations

import gradio as gr

from implementation.answer import CDCAnswerPipeline


print("Initializing CDC Diabetes Prevention RAG...")
pipeline = CDCAnswerPipeline()


def chat(
    message: str,
    history: list,
) -> str:
    """
    Generate an evidence-grounded answer from the CDC corpus.
    """

    if not message.strip():
        return "Please enter a question."

    try:
        result = pipeline.answer(message)
        return result["answer"]

    except Exception as exc:
        return (
            "An error occurred while generating the answer.\n\n"
            f"Details: {exc}"
        )


DESCRIPTION = """
Ask questions about **type 2 diabetes prevention** using a corpus of
CDC *Preventing Chronic Disease* publications.

The system uses hybrid BM25 + semantic retrieval, reciprocal rank
fusion, CrossEncoder reranking, and an evidence-grounded language
model response.

Answers are intended for educational and public-health information
and are not a substitute for individualized medical advice.
"""


demo = gr.ChatInterface(
    fn = chat,
    title = "CDC Diabetes Prevention RAG",
    description = DESCRIPTION,
    examples = [
        "How can lifestyle interventions prevent type 2 diabetes?",
        "How does physical activity help prevent type 2 diabetes?",
        "What is the role of weight loss in diabetes prevention?",
        "How can diabetes prevention programs help people with prediabetes?",
    ],
    save_history = False,
)


if __name__ == "__main__":
    demo.launch()
