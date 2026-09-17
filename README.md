### 1. `config.py`

Find:

```python
MAX_REFINEMENT_ATTEMPTS = 3
```

Replace with:

```python
REFINEMENT_CONFIDENCE_THRESHOLD = 0.70
MAX_REFINEMENT_ATTEMPTS = 2
```

The threshold already exists in your branch, so don't duplicate it.

---

### 2. `nodes/answer_questions.py`

**Purpose:** Keep the worksheet and variable from the Excel template fixed.

Find this existing section:

```python
final_answers = []

for answer in response.answers:
    answer.source_link = build_hyperlink(
        answer.page_number,
        source_link,
    )
    final_answers.append(answer.model_dump())
```

Replace it with:

```python
def answer_questions(state: FinancialGraphState) -> dict:
    """Extract metrics via LLM and enrich each answer with an Excel hyperlink."""
    llm = state.get("llm")
    structured_llm = llm.with_structured_output(ExtractionResponse)

    excel_data = state.get("excel_data", [])
    errors = state.get("errors", [])
    selected_pages = state.get("selected_pages", [])
    pdf_file_name = state.get("pdf_file_name", "")
  
    settings = get_settings()
    source_link = settings.sharepoint_link + pdf_file_name.replace(" ", "%20")


    USER_PROMPT = get_user_prompt(excel_data)
    
    full_context = "\n\n".join(
        [f"--- Page {i+1} ---\n{page.get('text', '')}" for i, page in enumerate(selected_pages)]
    )

    user_content = (
        f"Please extract the following variables:\n{USER_PROMPT}\n\n"
        f"Document context:\n{full_context}\n\n"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=user_content)
    ]
    
    try:
        response: ExtractionResponse = structured_llm.invoke(messages)
        
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

        for answer in response.answers:
            answer_data = answer.model_dump()

            key = (
                answer_data.get("worksheet"),
                answer_data.get("variable"),
            )

            template_metric = template_lookup.get(key)

            if template_metric is None:
                continue

            # Preserve the original template schema
            answer_data["worksheet"] = template_metric["worksheet"]
            answer_data["variable"] = template_metric["variable"]

            answer_data["source_link"] = build_hyperlink(
                answer_data.get("page_number"),
                source_link,
            )

            final_answers.append(answer_data)
                    
                return {
                    "answers": final_answers, 
                    "errors": response.errors
                }
        
    except Exception as e:
        error_msg = f"Bulk LLM extraction error: {str(e)}"
        errors.append(error_msg)
        return {
            "answers": [],
            "errors": errors
        }
```

This prevents the initial LLM output from introducing different worksheet/variable names. Your current code directly accepts the LLM's `worksheet` and `variable`.

---

### 3. `nodes/confidence_refinement.py`

**Purpose:** Verify the existing answer first, refine only when needed, and verify the refinement.

#### Step 3.1 — Replace `REFINEMENT_PROMPT`

```python
REFINEMENT_PROMPT = """
You are refining an existing financial extraction.

The Excel template schema is fixed.

Do NOT create new worksheets.
Do NOT create new variables.
Do NOT rename worksheets or variables.
Do NOT move metrics between worksheets.
Do NOT extract unrelated metrics.

Review the existing answer using the provided PDF evidence.

Check:
- existing answer
- page number
- financial year
- unit or scale
- source fields
- formula when applicable
- surrounding PDF context

If the existing answer is correct, keep it unchanged.
If the PDF supports a better answer, return the improved answer.

Do not guess or invent information.

The worksheet and variable must remain exactly unchanged.
"""
```

#### Step 3.2 — Replace `VERIFICATION_PROMPT`

```python
VERIFICATION_PROMPT = """
You are verifying an existing financial extraction against an annual report.

The PDF evidence is the source of truth.

Verify:

1. Whether the answer is supported by the PDF.
2. Whether the source fields semantically match the requested metric.
3. Whether the financial year is correct.
4. Whether the unit or scale is correct.
5. Whether the cited page supports the answer.
6. Whether the formula or calculation is correct when applicable.
7. Whether the surrounding PDF context supports the interpretation.

The source wording does not need to exactly match the requested
variable. Evaluate the financial meaning and context.

Do not change the worksheet or variable.
Do not assume high confidence means the answer is correct.

Return is_correct=true only when the PDF evidence supports the
candidate sufficiently.

If the candidate is incorrect and the correct value can be determined
from the PDF, return the supported value.

Do not invent information.
"""
```

#### Step 3.3 — Replace `_refine_answer()`

```python
def _refine_answer(
    llm,
    answer,
    pdf_context,
):
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

    # Preserve the original template schema
    candidate["worksheet"] = answer["worksheet"]
    candidate["variable"] = answer["variable"]

    return candidate
```

#### Step 3.4 — Replace `confidence_refinement()`

Replace the **entire existing `confidence_refinement()` function** with:

```python
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

    refined_answers = state.get(
        "refined_answers",
        [],
    )

    verification_results = state.get(
        "verification_results",
        [],
    )

    refined_by_metric = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in refined_answers
    }

    verification_by_metric = {
        (
            result.get("worksheet"),
            result["variable"],
        ): result
        for result in verification_results
    }

    current_answers = []

    for answer in answers:
        variable = answer.get(
            "variable",
            "",
        )

        original_verification = _verify_answer(
            llm,
            answer,
            pdf_context,
        )

        verification_by_metric[
            (
                answer["worksheet"],
                variable,
            )
        ] = original_verification

        if original_verification.get(
            "is_correct",
            False,
        ):
            current_answers.append(
                answer.copy()
            )
            continue

        confidence = float(
            answer.get(
                "confidence",
                0.0,
            )
        )

        if (
            confidence < REFINEMENT_CONFIDENCE_THRESHOLD
            or not original_verification.get(
                "is_correct",
                False,
            )
        ):
            candidate = _refine_answer(
                llm,
                answer,
                pdf_context,
            )

            # Preserve the original template schema
            candidate["worksheet"] = answer["worksheet"]
            candidate["variable"] = answer["variable"]

            refined_by_metric[
                (
                    answer["worksheet"],
                    variable,
                )
            ] = candidate

            refined_verification = _verify_answer(
                llm,
                candidate,
                pdf_context,
            )

            verification_by_metric[
                (
                    answer["worksheet"],
                    variable,
                )
            ] = refined_verification

            if refined_verification.get(
                "is_correct",
                False,
            ):
                current_answers.append(
                    candidate
                )
            else:
                current_answers.append(
                    answer.copy()
                )
        else:
            current_answers.append(
                answer.copy()
            )

    return {
        "answers": current_answers,
        "original_answers": original_answers,
        "refined_answers": list(
            refined_by_metric.values()
        ),
        "verification_results": list(
            verification_by_metric.values()
        ),
        "refinement_attempts": attempts + 1,
    }
```

**Important:** Don't add these fields to `answers`:

```python
verification_status
verification_confidence
verification_reason
```

Keep them inside `verification_results`.

#### Step 3.5 — Replace `route_after_refinement()`

```python
def route_after_refinement(
    state: FinancialGraphState,
) -> str:
    verification_results = state.get(
        "verification_results",
        []
    )

    attempts = state.get(
        "refinement_attempts",
        0,
    )

    if not verification_results:
        return "refine"

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

**One correction:** with the above `confidence_refinement()` implementation, each graph iteration verifies the current answer and may refine it. Therefore `MAX_REFINEMENT_ATTEMPTS = 2` means the refinement node gets at most two graph passes.

---

### 4. `nodes/create_excel_output.py`

**Purpose:** The Excel template, not the LLM, defines the final worksheets.

#### Step 4.1 — Replace `build_excel_dateframes()`

```python
def build_excel_dateframes(
    answers: list[dict],
    template_metrics: list[dict],
) -> dict[str, pd.DataFrame]:
    worksheet_dict = {}

    answer_lookup = {
        (
            answer["worksheet"],
            answer["variable"],
        ): answer
        for answer in answers
    }

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
            "formula": answer.get(
                "formula"
            ),
            "source_link": answer.get(
                "source_link"
            ),
        }

        worksheet_dict.setdefault(
            worksheet,
            [],
        ).append(row)

    return {
        worksheet: pd.DataFrame(rows)
        for worksheet, rows
        in worksheet_dict.items()
    }
```

#### Step 4.2 — Add schema validation

Add this function before `create_excel_output()`:

```python
def validate_template_schema(
    answers: list[dict],
    template_metrics: list[dict],
) -> None:
    template_keys = {
        (
            item["worksheet"],
            item["variable"],
        )
        for item in template_metrics
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
            "LLM produced metrics outside "
            f"the template schema: {sorted(unexpected)}"
        )
```

#### Step 4.3 — Modify `create_excel_output()`

Find:

```python
answers = state.get("answers")
```

Change to:

```python
answers = state.get("answers", [])
template_metrics = state.get("excel_data", [])
```

Then find:

```python
worksheet_dict = excel_preparation_chain.invoke(
    answers
)
```

Replace with:

```python
cleaned_answers = clean_llm_output(
    answers
)

validate_template_schema(
    answers=cleaned_answers,
    template_metrics=template_metrics,
)

worksheet_dict = build_excel_dateframes(
    answers=cleaned_answers,
    template_metrics=template_metrics,
)
```

The current `VerificationResult` model contains only `variable`, not `worksheet`.  Therefore, if you use `(worksheet, variable)` in `verification_by_metric`, you should also make this small change in `models.py`:

```python
class VerificationResult(BaseModel):
    worksheet: str = Field(
        description="The worksheet containing the variable."
    )
    variable: str = Field(
        description="The variable being verified."
    )
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

```python
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    MAX_REFINEMENT_ATTEMPTS,
    REFINEMENT_CONFIDENCE_THRESHOLD,
)
from graph.graph_state import FinancialGraphState
from models import ExtractionResponse, VerificationResponse


REFINEMENT_PROMPT = """
You are refining an existing financial extraction from an annual report.

Use only the provided PDF evidence.

Review the existing candidate and improve it only when the document
provides sufficient evidence.

Check:
- correct metric or variable
- correct numerical value
- correct financial year
- correct unit or scale
- correct page
- correct source fields
- formula or calculation when applicable
- surrounding PDF context

Do not invent information.
Do not use information outside the provided PDF evidence.

Do not change the worksheet or variable name.
Return the best supported candidate extraction.
"""


VERIFICATION_PROMPT = """
You are verifying an existing financial extraction against an annual report.

The PDF evidence is the source of truth.

Verify:

1. Whether the numerical value is supported by the PDF.
2. Whether it belongs to the requested metric.
3. Whether the financial year is correct.
4. Whether the unit or scale is correct.
5. Whether the cited page supports the answer.
6. Whether the source fields support the extraction.
7. Whether the formula or calculation is correct when applicable.
8. Whether the surrounding PDF context supports the interpretation.

Do not assume that a high confidence score means the answer is correct.

Return is_correct=true only when the PDF evidence sufficiently supports
the candidate.

If the candidate is incorrect and the correct value can be determined
from the PDF, return the supported value.

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
        return answer.copy()

    candidate = response.answers[0].model_dump()

    # Preserve the original template schema.
    candidate["worksheet"] = answer.get(
        "worksheet",
        "",
    )
    candidate["variable"] = answer.get(
        "variable",
        "",
    )

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

    # Keep worksheet context even if VerificationResult
    # does not define a worksheet field.
    result["worksheet"] = answer.get(
        "worksheet",
        "",
    )
    result["variable"] = answer.get(
        "variable",
        result.get("variable", ""),
    )

    return result


def confidence_refinement(
    state: FinancialGraphState,
) -> dict:
    answers = state.get("answers", [])
    attempts = state.get(
        "refinement_attempts",
        0,
    )

    original_answers = state.get(
        "original_answers"
    )

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

    # Keep previous refined candidates,
    # but verify only the current pass.
    refined_answers = state.get(
        "refined_answers",
        [],
    )

    refined_by_metric = {
        (
            answer.get("worksheet", ""),
            answer.get("variable", ""),
        ): answer
        for answer in refined_answers
    }

    # IMPORTANT:
    # Start fresh for every refinement pass.
    verification_by_metric = {}

    current_answers = []

    for answer in answers:
        worksheet = answer.get(
            "worksheet",
            "",
        )
        variable = answer.get(
            "variable",
            "",
        )

        metric_key = (
            worksheet,
            variable,
        )

        # First verify the current answer.
        original_verification = _verify_answer(
            llm,
            answer,
            pdf_context,
        )

        if original_verification.get(
            "is_correct",
            False,
        ):
            verification_by_metric[
                metric_key
            ] = original_verification

            current_answers.append(
                answer.copy()
            )
            continue

        # Verification failed.
        # Refinement is now required regardless of confidence.
        confidence = float(
            answer.get(
                "confidence",
                0.0,
            )
        )

        should_refine = (
            confidence < REFINEMENT_CONFIDENCE_THRESHOLD
            or not original_verification.get(
                "is_correct",
                False,
            )
        )

        if should_refine:
            candidate = _refine_answer(
                llm,
                answer,
                pdf_context,
            )
        else:
            candidate = answer.copy()

        # Preserve template identity.
        candidate["worksheet"] = worksheet
        candidate["variable"] = variable

        refined_by_metric[
            metric_key
        ] = candidate

        # Verify the refined candidate.
        refined_verification = _verify_answer(
            llm,
            candidate,
            pdf_context,
        )

        verification_by_metric[
            metric_key
        ] = refined_verification

        if refined_verification.get(
            "is_correct",
            False,
        ):
            # Use refined answer only when verified.
            current_answers.append(
                candidate.copy()
            )
        else:
            # Keep the current/original answer when
            # refinement is not verified.
            current_answers.append(
                answer.copy()
            )

    return {
        "answers": current_answers,
        "original_answers": original_answers,
        "refined_answers": list(
            refined_by_metric.values()
        ),
        "verification_results": list(
            verification_by_metric.values()
        ),
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

    # Always stop when maximum attempts are reached.
    if attempts >= MAX_REFINEMENT_ATTEMPTS:
        return "continue"

    # Nothing to verify.
    if not verification_results:
        return "continue"

    all_verified = all(
        result.get("is_correct", False)
        for result in verification_results
    )

    if all_verified:
        return "continue"

    return "refine"
```
