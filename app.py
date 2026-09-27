from __future__ import annotations

import re

import gradio as gr

from implementation.answer import CDCAnswerPipeline
from implementation.evidence_highlighter import create_highlighted_pdf


print("Initializing CDC Diabetes Prevention RAG...")
pipeline = CDCAnswerPipeline()


def is_in_scope(query: str) -> bool:
    q = query.lower()

    diabetes_terms = {
        "diabetes",
        "type 2 diabetes",
        "type ii diabetes",
        "prediabetes",
        "pre-diabetes",
        "diabetes prevention",
        "prevent diabetes",
        "diabetes risk",
        "blood glucose",
        "blood sugar",
        "glycemic",
        "glycaemic",
        "a1c",
        "hba1c",
        "insulin resistance",
        "lifestyle intervention",
        "lifestyle change",
        "physical activity",
        "exercise",
        "weight loss",
        "weight management",
        "obesity",
        "overweight",
        "diet",
        "nutrition",
        "diabetes prevention program",
        "diabetes prevention programme",
        "dpp",
        "cdc diabetes",
    }

    return any(term in q for term in diabetes_terms)


def get_cited_source_numbers(answer: str) -> list[int]:
    """
    Extract source numbers used in citations such as:

    [Source 1]
    [Source 2]
    [Sources 1, 3]
    """

    numbers: set[int] = set()

    pattern = r"\[Sources?\s+([0-9,\s]+)\]"

    for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
        for number in re.findall(r"\d+", match.group(1)):
            numbers.add(int(number))

    return sorted(numbers)


def chat(
    message: str,
    history: list,
):
    """
    Generate an evidence-grounded answer and highlighted source PDFs.
    """

    if not message.strip():
        return "Please enter a question.", [], ""

    # Reject questions outside the CDC diabetes-prevention scope
    # before running retrieval or calling the OpenAI API.
    if not is_in_scope(message):
        return (
            "This system is limited to questions about type 2 diabetes "
            "prevention based on the CDC corpus. Please ask a question "
            "within that scope.",
            [],
            "",
        )

    try:
        result = pipeline.answer(message)

        answer = result["answer"]
        sources = result["sources"]

        cited_numbers = get_cited_source_numbers(answer)

        highlighted_files = []
        evidence_lines = []

        for source_number in cited_numbers:

            if not 1 <= source_number <= len(sources):
                continue

            source = sources[source_number - 1]

            title = source.get("title", "Untitled source")
            page = source.get("page", "")
            landing_page = source.get("landing_page", "")

            evidence_lines.append(
                f"**Source {source_number}:** {title}, page {page}  \n"
                f"[CDC original]({landing_page})"
            )

            try:
                highlighted_pdf = create_highlighted_pdf(source)
                highlighted_files.append(str(highlighted_pdf))

            except Exception as exc:
                evidence_lines.append(
                    f"*Highlight unavailable for Source {source_number}: {exc}*"
                )

        evidence_markdown = "\n\n".join(evidence_lines)

        return answer, highlighted_files, evidence_markdown

    except Exception as exc:
        return (
            "An error occurred while generating the answer.\n\n"
            f"Details: {exc}",
            [],
            "",
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


with gr.Blocks() as demo:

    gr.Markdown("# CDC Diabetes Prevention RAG")
    gr.Markdown(DESCRIPTION)

    chatbot = gr.Chatbot()

    message = gr.Textbox(
        label="Question",
        placeholder="Ask a question about type 2 diabetes prevention...",
    )

    submit = gr.Button("Submit")

    gr.Markdown("## Evidence")

    evidence_info = gr.Markdown()

    highlighted_files = gr.Files(
        label="Highlighted supporting evidence",
        interactive=False,
    )

    gr.Examples(
        examples=[
            ["How can lifestyle interventions prevent type 2 diabetes?"],
            ["How does physical activity help prevent type 2 diabetes?"],
            ["What is the role of weight loss in diabetes prevention?"],
            ["How can diabetes prevention programs help people with prediabetes?"],
        ],
        inputs=message,
    )

    def respond(
        user_message: str,
        chat_history: list,
    ):
        answer, files, evidence = chat(
            user_message,
            chat_history,
        )

        chat_history = chat_history or []

        chat_history.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return (
            "",
            chat_history,
            evidence,
            files,
        )

    submit.click(
        fn=respond,
        inputs=[
            message,
            chatbot,
        ],
        outputs=[
            message,
            chatbot,
            evidence_info,
            highlighted_files,
        ],
    )

    message.submit(
        fn=respond,
        inputs=[
            message,
            chatbot,
        ],
        outputs=[
            message,
            chatbot,
            evidence_info,
            highlighted_files,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
