### 1. `config.py`

Change:

```python
MAX_REFINEMENT_ATTEMPTS = 3
```

to:

```python
MAX_REFINEMENT_ATTEMPTS = 2
```

Your current file still has `3`.

---

### 2. Fix `models.py`

Add `worksheet` to `VerificationResult`:

```python
class VerificationResult(BaseModel):
    worksheet: str = Field(
        description="The worksheet containing the variable."
    )

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
```

Your current model doesn't contain `worksheet`.

---

### 3. Fix `answer_questions.py`


```python
def answer_questions(state: FinancialGraphState) -> dict:
    """Extract metrics via LLM and enrich each answer with an Excel hyperlink."""
    llm = state.get("llm")

    if llm is None:
        return {
            "answers": [],
            "errors": state.get(
                "errors",
                [],
            ) + ["LLM not available"],
        }

    structured_llm = llm.with_structured_output(
        ExtractionResponse
    )

    excel_data = state.get(
        "excel_data",
        [],
    )
    errors = state.get(
        "errors",
        [],
    )
    selected_pages = state.get(
        "selected_pages",
        [],
    )
    pdf_file_name = state.get(
        "pdf_file_name",
        "",
    )

    settings = get_settings()

    source_link = (
        settings.sharepoint_link
        + pdf_file_name.replace(" ", "%20")
    )

    user_prompt = get_user_prompt(
        excel_data
    )

    full_context = "\n\n".join(
        [
            f"--- Page {i + 1} ---\n"
            f"{page.get('text', '')}"
            for i, page in enumerate(
                selected_pages
            )
        ]
    )

    user_content = (
        "Please extract the following variables:\n"
        f"{user_prompt}\n\n"
        "Document context:\n"
        f"{full_context}\n"
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_content
        ),
    ]

    try:
        response: ExtractionResponse = (
            structured_llm.invoke(messages)
        )

        if errors:
            response.errors.extend(errors)

        template_lookup = {
            (
                item["worksheet"],
                item["variable"],
            ): item
            for item in excel_data
        }

        final_answers = []

        for index, answer in enumerate(
            response.answers
        ):
            template_metric = None

            # Prefer template order.
            # This prevents the LLM from changing
            # worksheet or variable names.
            if index < len(excel_data):
                template_metric = (
                    excel_data[index]
                )

            # Fallback to worksheet + variable match.
            if template_metric is None:
                template_metric = (
                    template_lookup.get(
                        (
                            answer.worksheet,
                            answer.variable,
                        )
                    )
                )

            # Ignore metrics that are not part of
            # the Excel template.
            if template_metric is None:
                continue

            # Always preserve the template schema.
            answer.worksheet = (
                template_metric["worksheet"]
            )
            answer.variable = (
                template_metric["variable"]
            )

            answer.source_link = build_hyperlink(
                answer.page_number,
                source_link,
            )

            final_answers.append(
                answer.model_dump()
            )

        return {
            "answers": final_answers,
            "errors": response.errors,
        }

    except Exception as exc:
        error_msg = (
            "Bulk LLM extraction error: "
            f"{exc}"
        )

        return {
            "answers": [],
            "errors": errors + [error_msg],
        }
```


# 4. Replace `confidence_refinement.py`


```python
from langchain_core.messages import HumanMessage, SystemMessage

from config import MAX_REFINEMENT_ATTEMPTS
from graph.graph_state import FinancialGraphState
from models import ExtractionResponse, VerificationResponse


REFINEMENT_PROMPT = """
You are refining an existing financial extraction.

The Excel template schema is fixed.

Do NOT:
- create worksheets
- create variables
- rename worksheets
- rename variables
- move metrics between worksheets
- extract unrelated metrics

Review the existing answer using only the provided PDF evidence.

Check:
- metric
- value
- financial year
- unit or scale
- page number
- source fields
- formula
- surrounding PDF context

If the existing answer is correct, keep it unchanged.

If the PDF supports a better answer, return the corrected answer.

Do not guess or invent information.

The worksheet and variable must remain exactly unchanged.
"""


VERIFICATION_PROMPT = """
You are verifying an existing financial extraction against an annual report.

The PDF evidence is the source of truth.

Verify:

1. Whether the answer is supported by the PDF.
2. Whether it belongs to the requested metric.
3. Whether the financial year is correct.
4. Whether the unit or scale is correct.
5. Whether the cited page supports the answer.
6. Whether the source fields support the extraction.
7. Whether the formula is correct when applicable.
8. Whether the surrounding context supports the interpretation.

Do not assume that a high confidence score means the answer is correct.

Return is_correct=true only when the PDF evidence sufficiently
supports the candidate.

If incorrect, identify the supported value when it can be
determined from the PDF.

Do not invent information.

Do not change the worksheet or variable.
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
        return answer.copy()

    candidate = response.answers[0].model_dump()

    # Template schema is immutable.
    candidate["worksheet"] = answer["worksheet"]
    candidate["variable"] = answer["variable"]

    return candidate


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
            "worksheet": answer.get("worksheet", ""),
            "variable": answer.get("variable", ""),
            "is_correct": False,
            "verification_confidence": 0.0,
            "verified_value": "N/A",
            "page_number": -1,
            "source_fields": [],
            "formula": None,
            "reason": "Verification returned no result.",
        }

    result = response.results[0].model_dump()

    result["worksheet"] = answer["worksheet"]
    result["variable"] = answer["variable"]

    return result


def confidence_refinement(
    state: FinancialGraphState,
) -> dict:

    answers = state.get("answers", [])
    attempts = state.get("refinement_attempts", 0)

    original_answers = state.get("original_answers")

    if original_answers is None:
        original_answers = [
            answer.copy()
            for answer in answers
        ]

    llm = state.get("llm")

    if llm is None:
        raise ValueError(
            "LLM is missing from graph state"
        )

    selected_pages = state.get(
        "selected_pages",
        [],
    )

    pdf_context = _build_pdf_context(
        selected_pages
    )

    verification_results = []

    current_answers = []

    for original_answer in answers:

        # Always verify the current answer first.
        verification = _verify_answer(
            llm,
            original_answer,
            pdf_context,
        )

        if verification.get("is_correct"):
            current_answers.append(
                original_answer.copy()
            )

            verification_results.append(
                verification
            )

            continue

        # Verification failed, so refine the
        # existing answer using the same PDF evidence.
        refined_answer = _refine_answer(
            llm,
            original_answer,
            pdf_context,
        )

        refined_verification = _verify_answer(
            llm,
            refined_answer,
            pdf_context,
        )

        verification_results.append(
            refined_verification
        )

        if refined_verification.get("is_correct"):
            current_answers.append(
                refined_answer
            )
        else:
            # Keep the original answer if refinement
            # cannot be verified.
            current_answers.append(
                original_answer.copy()
            )

    return {
        "answers": current_answers,
        "original_answers": original_answers,
        "refined_answers": current_answers,
        "verification_results": verification_results,
        "refinement_attempts": attempts + 1,
    }


def route_after_refinement(
    state: FinancialGraphState,
) -> str:

    attempts = state.get(
        "refinement_attempts",
        0,
    )

    verification_results = state.get(
        "verification_results",
        [],
    )

    if not verification_results:
        return "continue"

    all_verified = all(
        result.get("is_correct", False)
        for result in verification_results
    )

    if all_verified:
        return "continue"

    if attempts >= MAX_REFINEMENT_ATTEMPTS:
        return "continue"

    return "refine"
```

The important workflow is now:

```text
Original Extraction
        ↓
Verify Original
        ↓
   Correct?
   /      \
 Yes       No
 ↓         ↓
Keep     Refine
           ↓
        Verify
           ↓
       Correct?
       /     \
     Yes      No
      ↓        ↓
    Keep    Keep Original
```

---

# `create_excel_output.py`


Replace the file with:

```python
"""Node for creating the final Excel output."""

import pandas as pd

from graph.graph_state import FinancialGraphState
from template.excel_writer import excel_write


OUTPUT_COLUMNS = [
    "variable",
    "answer",
    "confidence",
    "source_fields",
    "formula",
    "source_link",
]


def clean_llm_output(
    answers: list[dict],
) -> list[dict]:
    """Clean extracted answers before Excel generation."""

    cleaned_answers = []

    for answer in answers:
        row = {
            column: answer.get(column)
            for column in OUTPUT_COLUMNS
        }

        if str(row.get("answer")).strip().upper() == "N/A":
            row["answer"] = None

        if str(row.get("source_link")).strip().upper() == "N/A":
            row["source_link"] = None

        if row.get("formula") in ("", "null", "None"):
            row["formula"] = None

        if row.get("source_fields") in ("[]", []):
            row["source_fields"] = None

        cleaned_answers.append(row)

    return cleaned_answers


def validate_template_schema(
    answers: list[dict],
    template_metrics: list[dict],
) -> None:
    """Ensure LLM answers belong to the Excel template."""

    template_keys = {
        (
            metric["worksheet"],
            metric["variable"],
        )
        for metric in template_metrics
    }

    answer_keys = {
        (
            answer["worksheet"],
            answer["variable"],
        )
        for answer in answers
    }

    unexpected = answer_keys - template_keys

    if unexpected:
        raise ValueError(
            "LLM produced metrics outside the "
            f"template schema: {sorted(unexpected)}"
        )


def build_excel_dateframes(
    answers: list[dict],
    template_metrics: list[dict],
) -> dict[str, pd.DataFrame]:
    """
    Build Excel worksheets from the template.

    The template determines:
    - worksheet names
    - variables
    - row structure

    LLM output only supplies values.
    """

    answer_lookup = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in answers
    }

    worksheet_dict = {}

    for metric in template_metrics:
        worksheet = metric["worksheet"]
        variable = metric["variable"]

        answer = answer_lookup.get(
            (worksheet, variable),
            {},
        )

        row = {
            "variable": variable,
            "answer": answer.get("answer"),
            "confidence": answer.get(
                "confidence",
                0.0,
            ),
            "source_fields": answer.get(
                "source_fields"
            ),
            "formula": answer.get("formula"),
            "source_link": answer.get(
                "source_link"
            ),
        }

        worksheet_dict.setdefault(
            worksheet,
            [],
        ).append(row)

    return {
        worksheet: pd.DataFrame(
            rows,
            columns=OUTPUT_COLUMNS,
        )
        for worksheet, rows in worksheet_dict.items()
    }


def create_excel_output(
    state: FinancialGraphState,
):
    """Create the final Excel file."""

    answers = state.get(
        "answers",
        []
    )

    template_metrics = state.get(
        "excel_data",
        []
    )

    output_path = state.get(
        "output_path",
        ""
    )

    pdf_file_name = state.get(
        "pdf_file_name",
        ""
    )

    if not output_path:
        raise ValueError(
            "output_path is missing from graph state"
        )

    if not pdf_file_name:
        raise ValueError(
            "pdf_file_name is missing from graph state"
        )

    validate_template_schema(
        answers,
        template_metrics,
    )

    cleaned_answers = clean_llm_output(
        answers
    )

    worksheet_dict = build_excel_dateframes(
        answers=cleaned_answers,
        template_metrics=template_metrics,
    )

    if output_path.startswith("/mnt/"):
        output_excel = (
            "dbfs:"
            + output_path
            + pdf_file_name.removeprefix(
                "InputData"
            ).removesuffix(".pdf")
            + ".xlsx"
        )
    else:
        output_excel = (
            output_path
            + pdf_file_name.removeprefix(
                "InputData"
            ).removesuffix(".pdf")
            + ".xlsx"
        )

    try:
        status = excel_write(
            dataframes=worksheet_dict,
            output_path=output_excel,
        )

        return {
            "output_excel": status,
            "errors": state.get(
                "errors",
                [],
            ),
        }

    except Exception as exc:
        error = (
            "Error writing Excel file: "
            f"{exc}"
        )

        return {
            "output_excel": "",
            "errors": (
                state.get("errors", [])
                + [error]
            ),
        }
```

**Commit Message:**

```text
fix: preserve template schema during refinement
```

**Commit description:**

```text
Update the refinement workflow to improve existing LLM answers
without changing the Excel template schema.

- Verify the original answer before refinement
- Limit refinement to two attempts
- Preserve worksheet and variable names
- Refine only when verification or confidence indicates an issue
- Validate refined answers against PDF evidence
- Prevent creation or renaming of worksheets and variables
- Keep verification details internal
- Generate final Excel worksheets from the original template
```
