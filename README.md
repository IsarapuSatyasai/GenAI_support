```python
feature/confidence_refinement_verification
```

```python
REFINEMENT_CONFIDENCE_THRESHOLD = 0.70
MAX_REFINEMENT_ATTEMPTS = 3
```

```python
class VerificationResult(BaseModel):
    variable: str = Field(
        description="The variable being verified."
    )
    is_correct: bool = Field(
        description="Whether the extracted answer is supported by the PDF."
    )
    verification_confidence: float = Field(
        description="Confidence from 0.0 to 1.0 that the PDF supports the answer."
    )
    verified_value: str = Field(
        description="The value supported by the PDF. Use 'N/A' if not found."
    )
    page_number: int = Field(
        description="The relevant PDF page number. Use -1 if not found."
    )
    source_fields: List[str] = Field(
        default_factory=list,
        description="PDF fields supporting the answer."
    )
    formula: Optional[str] = Field(
        default=None,
        description="Formula used when the answer requires calculation."
    )
    reason: str = Field(
        description="Short explanation of why the answer is correct or incorrect."
    )


class VerificationResponse(BaseModel):
    results: List[VerificationResult] = Field(
        default_factory=list
    )
```


```python
original_answers: List[Dict[str, Any]]
refined_answers: List[Dict[str, Any]]
verification_results: List[Dict[str, Any]]
```

```python
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    MAX_REFINEMENT_ATTEMPTS,
    REFINEMENT_CONFIDENCE_THRESHOLD,
)
from graph.graph_state import FinancialGraphState
from models import ExtractionResponse, VerificationResponse


REFINEMENT_PROMPT = """
You are refining a financial extraction from an annual report.

Use only the provided PDF evidence.

Review the existing candidate and correct it only when the
document provides sufficient evidence.

Check:
- correct metric or variable
- correct numerical value
- correct financial year
- correct unit or scale
- correct page
- correct source fields
- formula or calculation when applicable

Do not invent information.
Do not use information outside the provided PDF evidence.

Return the best supported candidate extraction.
"""


VERIFICATION_PROMPT = """
You are verifying a financial extraction against an annual report.

The PDF evidence is the source of truth.

For the provided candidate, verify:

1. Whether the numerical value is actually supported.
2. Whether it belongs to the requested metric.
3. Whether the financial year is correct.
4. Whether the unit or scale is correct.
5. Whether the cited page is correct.
6. Whether the source fields support the extraction.
7. Whether the formula/calculation is correct when applicable.
8. Whether the surrounding PDF context supports the interpretation.

Do not assume that a high confidence score means the answer is correct.

Return is_correct=true only when the PDF evidence supports the
candidate sufficiently.

If the candidate is incorrect, identify the value supported by
the PDF evidence when it can be determined.

Do not invent information.
"""


def _build_pdf_context(selected_pages):
    context = []

    for page in selected_pages:
        page_number = page.get("page_number")

        if page_number is None:
            page_number = page.get("page", -1)

        text = page.get("text", "")

        context.append(
            f"--- PDF Page {page_number} ---\n{text}"
        )

    return "\n\n".join(context)


def _refine_answer(llm, answer, pdf_context):
    structured_llm = llm.with_structured_output(
        ExtractionResponse
    )

    messages = [
        SystemMessage(content=REFINEMENT_PROMPT),
        HumanMessage(
            content=f"""
Existing candidate:

{answer}

PDF evidence:

{pdf_context}
"""
        ),
    ]

    response = structured_llm.invoke(messages)

    if not response.answers:
        return answer

    return response.answers[0].model_dump()


def _verify_answer(llm, answer, pdf_context):
    structured_llm = llm.with_structured_output(
        VerificationResponse
    )

    messages = [
        SystemMessage(content=VERIFICATION_PROMPT),
        HumanMessage(
            content=f"""
Candidate extraction:

{answer}

PDF evidence:

{pdf_context}
"""
        ),
    ]

    response = structured_llm.invoke(messages)

    if not response.results:
        return {
            "variable": answer.get("variable", ""),
            "is_correct": False,
            "verification_confidence": 0.0,
            "verified_value": "N/A",
            "page_number": -1,
            "source_fields": [],
            "formula": None,
            "reason": "Verification returned no result.",
        }

    return response.results[0].model_dump()


def confidence_refinement(
    state: FinancialGraphState,
) -> dict:
    answers = state.get("answers", [])
    attempts = state.get("refinement_attempts", 0)
    original_answers = state.get("original_answers")

    if original_answers is None:
        original_answers = [answer.copy() for answer in answers]

    llm = state.get("llm")

    if llm is None:
        raise ValueError("LLM is missing from graph state")

    selected_pages = state.get("selected_pages", [])
    pdf_context = _build_pdf_context(selected_pages)

    refined_answers = state.get("refined_answers", [])
    verification_results = state.get("verification_results", [])

    refined_by_variable = {
        answer["variable"]: answer
        for answer in refined_answers
    }

    verification_by_variable = {
        result["variable"]: result
        for result in verification_results
    }

    current_answers = []

    for answer in answers:
        variable = answer.get("variable", "")

        previous_verification = verification_by_variable.get(
            variable
        )

        if (
            previous_verification
            and previous_verification.get("is_correct")
        ):
            current_answers.append(answer)
            continue

        confidence = float(
            answer.get("confidence", 0.0)
        )

        candidate = answer

        if confidence < REFINEMENT_CONFIDENCE_THRESHOLD:
            candidate = _refine_answer(
                llm,
                answer,
                pdf_context,
            )

        elif variable in refined_by_variable:
            candidate = refined_by_variable[variable]

        verification = _verify_answer(
            llm,
            candidate,
            pdf_context,
        )

        refined_by_variable[variable] = candidate
        verification_by_variable[variable] = verification

        if verification.get("is_correct"):
            final_answer = candidate.copy()

            final_answer["verification_status"] = "verified"
            final_answer["verification_confidence"] = (
                verification.get(
                    "verification_confidence",
                    0.0,
                )
            )
            final_answer["verification_reason"] = (
                verification.get("reason", "")
            )

            current_answers.append(final_answer)

        else:
            candidate = candidate.copy()

            candidate["verification_status"] = (
                "verification_failed"
            )
            candidate["verification_confidence"] = (
                verification.get(
                    "verification_confidence",
                    0.0,
                )
            )
            candidate["verification_reason"] = (
                verification.get("reason", "")
            )

            current_answers.append(candidate)

    return {
        "answers": current_answers,
        "original_answers": original_answers,
        "refined_answers": list(refined_by_variable.values()),
        "verification_results": list(
            verification_by_variable.values()
        ),
        "refinement_attempts": attempts + 1,
    }


def route_after_refinement(
    state: FinancialGraphState,
) -> str:
    answers = state.get("answers", [])
    attempts = state.get("refinement_attempts", 0)

    if not answers:
        return "continue"

    all_verified = all(
        answer.get("verification_status") == "verified"
        for answer in answers
    )

    if all_verified:
        return "continue"

    if attempts >= MAX_REFINEMENT_ATTEMPTS:
        return "continue"

    return "refine"
```

```python
f"--- Page {i + 1} ---\n{page.get('text', '')}"


page_number = page.get("page_number")

if page_number is None:
    page_number = page.get("page", -1)
```

```python
add source-based verification to confidence refinement
```

```python
- Enhanced confidence refinement to verify every extracted financial value against the relevant annual report PDF.
- Preserved the original extraction separately and generated refined candidates for low-confidence or verification-failed values.
- Added source-based verification for both high- and low-confidence extractions instead of relying only on the confidence score.
- Verification checks the extracted value, metric/variable, financial year, page number, source fields, surrounding context, and formula where applicable.
- Implemented a maximum of 3 refinement and verification cycles; values that remain unsupported are marked as verification_failed.
- Kept the existing graph structure unchanged by implementing the refinement and verification loop entirely within the existing confidence_refinement node.
```
