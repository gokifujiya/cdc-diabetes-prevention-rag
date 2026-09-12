from __future__ import annotations


SYSTEM_PROMPT = """
You are an evidence-grounded assistant for type 2 diabetes
prevention and public-health education.

You answer questions using only the CDC evidence passages
provided to you.

Rules:

1. Ground factual claims in the supplied CDC evidence.
2. Do not invent facts, statistics, recommendations, or sources.
3. If the supplied evidence is insufficient, clearly say so.
4. Cite supporting evidence using the source labels provided,
   for example [Source 1] or [Source 2].
5. Distinguish research findings from general recommendations.
6. Do not diagnose an individual.
7. Do not recommend starting, stopping, or changing medication.
8. Do not replace individualized medical advice from a qualified
   health professional.
9. When evidence concerns a particular population, setting, or
   study, preserve that limitation rather than generalizing it
   to everybody.
10. Prefer clear, concise language suitable for patient education
    while preserving important scientific qualifications.
"""


def build_context(
    passages: list[dict],
) -> str:
    """
    Convert retrieved CDC passages into labeled evidence
    for the language model.
    """

    sections: list[str] = []

    for index, passage in enumerate(
        passages,
        start=1,
    ):
        title = passage.get(
            "title",
            "Unknown title",
        )

        cdc_id = passage.get(
            "cdc_id",
            "",
        )

        page = passage.get(
            "page",
            "",
        )

        source = passage.get(
            "landing_page",
            "",
        )

        text = passage.get(
            "text",
            "",
        )

        section = f"""
[Source {index}]
Title: {title}
CDC ID: {cdc_id}
Page: {page}
URL: {source}

{text}
""".strip()

        sections.append(section)

    return "\n\n".join(sections)


def build_user_prompt(
    question: str,
    passages: list[dict],
) -> str:
    """
    Build the evidence-grounded user prompt.
    """

    context = build_context(
        passages
    )

    return f"""
Answer the question using only the CDC evidence below.

Question:
{question}

CDC evidence:
{context}

Requirements:
- Give a direct answer first.
- Support important factual statements with source labels such
  as [Source 1].
- Do not cite a source unless it supports the statement.
- Do not make claims beyond the supplied evidence.
- At the end, provide a short "Sources" section listing the
  CDC article title, page, and URL for each source actually used.
""".strip()
