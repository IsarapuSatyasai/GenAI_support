### Refinement changes

Modify `config.py` so refinement stops after 2 attempts:

```python
REFINEMENT_CONFIDENCE_THRESHOLD = 0.70
MAX_REFINEMENT_ATTEMPTS = 2
```

Modify `nodes/confidence_refinement.py` so the original answer is verified first and refinement only improves the existing answer:

```python
original_verification = _verify_answer(
    llm,
    answer,
    pdf_context,
)

if original_verification.get("is_correct"):
    current_answers.append(answer.copy())
    continue

confidence = float(
    answer.get("confidence", 0.0)
)

if (
    confidence < REFINEMENT_CONFIDENCE_THRESHOLD
    or not original_verification.get("is_correct")
):
    candidate = _refine_answer(
        llm,
        answer,
        pdf_context,
    )

    # Preserve the original template schema
    candidate["worksheet"] = answer["worksheet"]
    candidate["variable"] = answer["variable"]

    refined_verification = _verify_answer(
        llm,
        candidate,
        pdf_context,
    )

    if refined_verification.get("is_correct"):
        current_answers.append(candidate)
    else:
        current_answers.append(answer.copy())
```

Update `_refine_answer()`:

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

Update the refinement prompt:

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

Keep verification details internal. Do not add them to the final answer:

```python
verification_results.append(verification)
```

Do not add:

```python
candidate["verification_status"] = "verified"
candidate["verification_confidence"] = ...
candidate["verification_reason"] = ...
```

Update `route_after_refinement()`:

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

Use `(worksheet, variable)` as the refinement/verification key:

```python
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
```

### Answer schema changes

Modify `nodes/answer_questions.py` so the LLM output cannot change the template identity:

```python
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

    answer_data["worksheet"] = template_metric["worksheet"]
    answer_data["variable"] = template_metric["variable"]

    answer_data["source_link"] = build_hyperlink(
        answer_data.get("page_number"),
        source_link,
    )

    final_answers.append(answer_data)
```

### Excel output changes

Modify `nodes/create_excel_output.py` so the original template defines the worksheets:

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
            "confidence": answer.get("confidence", 0.0),
            "source_fields": answer.get("source_fields"),
            "formula": answer.get("formula"),
            "source_link": answer.get("source_link"),
        }

        worksheet_dict.setdefault(
            worksheet,
            [],
        ).append(row)

    return {
        worksheet: pd.DataFrame(rows)
        for worksheet, rows in worksheet_dict.items()
    }
```

Update `create_excel_output()`:

```python
answers = state.get("answers", [])
template_metrics = state.get("excel_data", [])

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

Add schema validation:

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
