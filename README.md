```python
from langchain_core.messages import HumanMessage, SystemMessage

from graph.graph_state import FinancialGraphState
from models import ExtractionResponse
from config import (
    VALIDATION_SCORE_THRESHOLD,
    MAX_REFINEMENT_ATTEMPTS,
)


REFINEMENT_PROMPT = """
Review the existing financial extraction against the provided
annual report context.

The existing answer has low confidence.

Try to improve or correct the existing answer using the document.
Do not extract unrelated metrics.
Do not invent information.

Return the corrected answer, confidence, page number,
source fields, and formula.
"""


def confidence_refinement(
    state: FinancialGraphState,
) -> dict:

    answers = state.get("answers", [])
    attempts = state.get("refinement_attempts", 0)

    # Check confidence
    low_confidence = [
        answer
        for answer in answers
        if float(answer.get("confidence", 0.0))
        < VALIDATION_SCORE_THRESHOLD
    ]

    # Nothing needs refinement
    if not low_confidence:
        return {
            "refinement_attempts": attempts
        }

    llm = state.get("llm")

    if llm is None:
        raise ValueError("LLM is missing from graph state")

    selected_pages = state.get("selected_pages", [])

    structured_llm = llm.with_structured_output(
        ExtractionResponse
    )

    full_context = "\n\n".join(
        [
            f"--- Page {i + 1} ---\n{page.get('text', '')}"
            for i, page in enumerate(selected_pages)
        ]
    )

    refined_answers = []

    for answer in low_confidence:

        messages = [
            SystemMessage(
                content=REFINEMENT_PROMPT
            ),
            HumanMessage(
                content=f"""
Existing extraction:

{answer}

Annual report context:

{full_context}
"""
            ),
        ]

        response = structured_llm.invoke(messages)

        refined_answers.extend(response.answers)

    # Replace only the low-confidence answers
    refined_by_variable = {
        answer.variable: answer.model_dump()
        for answer in refined_answers
    }

    updated_answers = []

    for answer in answers:

        variable = answer.get("variable")

        if variable in refined_by_variable:
            updated_answers.append(
                refined_by_variable[variable]
            )
        else:
            updated_answers.append(answer)

    # Return STATE update
    return {
        "answers": updated_answers,
        "refinement_attempts": attempts + 1,
    }


def route_after_refinement(
    state: FinancialGraphState,
) -> str:

    answers = state.get("answers", [])
    attempts = state.get("refinement_attempts", 0)

    # Check remaining low-confidence answers
    low_confidence = [
        answer
        for answer in answers
        if float(answer.get("confidence", 0.0))
        < VALIDATION_SCORE_THRESHOLD
    ]

    # Everything is good
    if not low_confidence:
        return "continue"

    # Maximum attempts reached
    if attempts >= MAX_REFINEMENT_ATTEMPTS:
        return "continue"

    # Try refinement again
    return "refine"
```
